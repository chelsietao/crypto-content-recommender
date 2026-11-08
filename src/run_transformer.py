"""Train, evaluate and benchmark a CPU Transformer reranker using frozen V7 recall."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data_pipeline import build_dataset
from src.evaluation import chronological_holdout, evaluate_all
from src.neural import NeuralSystem, seed_cpu
from src.transformer_reranker import TransformerConfig, TransformerReranker

BASE = ARTIFACTS_DIR / 'neural/neural_recommender.pt'
DEFAULT_OUTPUT = REPORTS_DIR / 'transformer_reranking'
EXPORTS = {'transformer.pt', 'training_history.csv', 'training_audit.json', 'training_curves.png',
           'model_metrics.csv', 'evaluation_details.csv', 'recommendations.csv', 'latency_samples.csv',
           'benchmark_workers.json', 'performance.csv', 'quality_deltas.json', 'run_summary.json'}


def performance(samples, workers):
    rows = []
    for model, frame in samples.groupby('model', sort=True):
        runs = [row for row in workers if row['model'] == model]
        values = frame.wall_ms.to_numpy()
        if not len(runs) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Invalid benchmark samples')
        rows.append({'model': model, 'requests': len(frame), 'errors': int(frame.error.sum()),
            'p50_ms': float(np.percentile(values, 50)), 'p95_ms': float(np.percentile(values, 95)),
            'p99_ms': float(np.percentile(values, 99)),
            'requests_per_second': len(frame) / sum(row['window_wall_seconds'] for row in runs),
            'cpu_seconds_per_1000_requests': float(frame.cpu_ms.mean()),
            'peak_worker_rss_mib': max(row['peak_rss_bytes'] for row in runs) / 2**20,
            'parameter_bytes': runs[0]['parameter_bytes'], 'resident_parameter_bytes': runs[0]['resident_parameter_bytes'],
            'checkpoint_bytes': runs[0]['checkpoint_bytes']})
    return pd.DataFrame(rows)


def benchmark_worker(output, name, count, repetition):
    seed_cpu(42)
    tick = time.perf_counter()
    if name == 'transformer':
        model = TransformerReranker.load(output / 'transformer.pt', BASE)
    else:
        system = NeuralSystem.load(BASE)
        model = system.recommenders()['v8_pytorch_neural_ranker']
    load_seconds = time.perf_counter() - tick
    system = model.system
    users = [*system.users.user_id, 'new-user']
    for index in range(10):
        model.recommend(users[index % len(users)])
    process = psutil.Process()
    rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.001):
            rss.append(process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    records = []
    started = time.perf_counter()
    try:
        for index in range(count):
            user = users[index % len(users)]
            cpu, tick = time.process_time(), time.perf_counter()
            result = model.recommend(user)
            elapsed, cpu_elapsed = time.perf_counter() - tick, time.process_time() - cpu
            if not len(result) or not np.isfinite(result.score).all():
                raise ValueError('Invalid benchmark recommendation')
            records.append({'model': model.name, 'round': repetition, 'request': index, 'user_id': user,
                'wall_ms': elapsed * 1000, 'cpu_ms': cpu_elapsed * 1000, 'error': 0})
        wall = time.perf_counter() - started
    finally:
        stop.set()
        thread.join()
    rss.append(process.memory_info().rss)
    modules = [system.tower, system.ranker] if name != 'transformer' else [system.tower, model.network]
    parameter_bytes = sum(p.numel() * p.element_size() for module in modules for p in module.parameters())
    # Transformer worker retains the base NeuralSystem, including its unused V8 ranker.
    resident_parameter_bytes = parameter_bytes + (sum(p.numel()*p.element_size() for p in system.ranker.parameters()) if name == 'transformer' else 0)
    result = {'model': model.name, 'round': repetition, 'requests': count, 'warmups': 10,
        'threads': torch.get_num_threads(), 'window_wall_seconds': wall, 'load_seconds': load_seconds,
        'peak_rss_bytes': max(rss), 'rss_sample_interval_seconds': 0.001,
        'parameter_bytes': parameter_bytes, 'resident_parameter_bytes': resident_parameter_bytes,
        'checkpoint_bytes': BASE.stat().st_size + ((output / 'transformer.pt').stat().st_size if name == 'transformer' else 0),
        'samples': records}
    print(json.dumps(result))


def quality_deltas(details):
    pivot = details.pivot(index='user_id', columns='model', values='ndcg_at_k')
    names = ['v8_pytorch_neural_ranker', TransformerReranker.name]
    delta = (pivot[names[1]] - pivot[names[0]]).to_numpy()
    if not len(delta) or not np.isfinite(delta).all():
        raise ValueError('Quality comparison needs complete paired users')
    generator = np.random.default_rng(42)
    bootstrap = delta[generator.integers(0, len(delta), size=(2000, len(delta)))].mean(axis=1)
    return {'baseline': names[0], 'model': names[1], 'metric': 'ndcg@10', 'users': len(delta),
        'mean_paired_delta': float(delta.mean()), 'bootstrap_repetitions': 2000,
        'percentile_95_interval': np.percentile(bootstrap, [2.5, 97.5]).tolist(), 'seed': 42}


def run(output, config, requests=100, repetitions=4):
    if requests < 1 or repetitions < 1:
        raise ValueError('Request and repetition counts must be positive')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use an empty output directory to preserve experiment evidence')
    seed_cpu(config.seed)
    system = NeuralSystem.load(BASE)
    model = TransformerReranker(system, config)
    cpu, tick = time.process_time(), time.perf_counter()
    model.fit_transformer()
    train_wall, train_cpu = time.perf_counter()-tick, time.process_time()-cpu
    model.save(output / 'transformer.pt', BASE)
    restored = TransformerReranker.load(output / 'transformer.pt', BASE)
    max_error = 0.0
    frames = []
    for user in [*system.users.user_id, 'new-user']:
        left, right = model.rank_scores(user), restored.rank_scores(user)
        np.testing.assert_array_equal(np.isfinite(left), np.isfinite(right))
        finite = np.isfinite(left)
        np.testing.assert_allclose(left[finite], right[finite], atol=1e-7, rtol=0)
        max_error = max(max_error, float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else 0.0)
        result = model.recommend(user)
        assert result.content_id.tolist() == restored.recommend(user).content_id.tolist()
        result.insert(0, 'user_id', user)
        result.insert(1, 'rank', range(1, len(result)+1))
        frames.append(result)
    pd.concat(frames, ignore_index=True).to_csv(output / 'recommendations.csv', index=False)
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    models = system.recommenders()
    models[model.name] = model
    metrics, details = evaluate_all(models, split.test)
    metrics.to_csv(output / 'model_metrics.csv', index=False)
    details.to_csv(output / 'evaluation_details.csv', index=False)
    pd.DataFrame(model.history).to_csv(output / 'training_history.csv', index=False)
    (output / 'training_audit.json').write_text(json.dumps(model.training_audit, indent=2)+'\n')
    (output / 'quality_deltas.json').write_text(json.dumps(quality_deltas(details), indent=2)+'\n')
    from src.analytics import plt
    history = pd.DataFrame(model.history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.train_loss, label='Training')
    ax.plot(history.epoch, history.validation_loss, label='Validation')
    ax.axvline(model.training_summary['selected_epoch'], linestyle=':', color='black', label='Selected')
    ax.set(xlabel='Epoch', ylabel='Weighted BCE loss', title='Transformer content reranker')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=160)
    plt.close(fig)
    workers, samples = [], []
    for repetition in range(repetitions):
        order = ('neural', 'transformer') if repetition % 2 == 0 else ('transformer', 'neural')
        for name in order:
            command = [sys.executable, '-m', 'src.run_transformer', '--output', str(output.resolve()),
                '--benchmark-worker', name, '--requests', str(requests), '--round', str(repetition)]
            result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'Benchmark worker failed ({result.returncode}): {result.stderr[-3000:]}')
            worker = json.loads(result.stdout)
            samples.extend(worker.pop('samples'))
            workers.append(worker)
            print(json.dumps({'completed_round': repetition, 'model': name}), flush=True)
    frame = pd.DataFrame(samples)
    frame.to_csv(output / 'latency_samples.csv', index=False)
    performance(frame, workers).to_csv(output / 'performance.csv', index=False)
    (output / 'benchmark_workers.json').write_text(json.dumps(workers, indent=2)+'\n')
    summary = {'format_version': 1, 'device': 'cpu', 'dtype': 'float32', 'threads': 1, 'concurrency': 1,
        'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(),
        'cpu_model': platform.processor(), 'logical_cpus': psutil.cpu_count(), 'physical_memory_bytes': psutil.virtual_memory().total,
        'config': asdict(config), 'requests_per_worker': requests, 'repetitions': repetitions,
        'training_wall_seconds': train_wall, 'training_cpu_seconds': train_cpu, 'training': model.training_summary,
        'training_cutoff': split.cutoff.isoformat(), 'evaluated_users': len(split.test),
        'checkpoint_reload_users': len(system.users)+1, 'checkpoint_reload_max_score_error': max_error,
        'base_checkpoint_sha256': hashlib.sha256(BASE.read_bytes()).hexdigest(),
        'measurement_scope': 'in-process warm recommendation, frozen retrieval + reranking + top-10 formatting, cached catalog tokenization',
        'memory_scope': 'isolated worker absolute sampled RSS after warmup, includes Python, PyTorch and base NeuralSystem',
        'cpu_scope': 'process CPU time per request, including RSS sampler thread',
        'training_scope': 'example construction and Transformer fit, excludes base checkpoint training and exports',
        'source_sha256': {name: hashlib.sha256((PROJECT_ROOT/'src'/name).read_bytes()).hexdigest()
                           for name in ('transformer_reranker.py','run_transformer.py','neural.py','evaluation.py')}}
    (output/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                                                   for name in sorted(EXPORTS)}, indent=2)+'\n')
    verify(output)
    print(metrics.to_string(index=False))


def verify(output):
    hashes = json.loads((output/'manifest.json').read_text())
    if set(hashes) != EXPORTS:
        raise ValueError('Transformer manifest must cover every required export')
    for name, expected in hashes.items():
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Transformer artifact hash mismatch: {name}')
    summary = json.loads((output/'run_summary.json').read_text())
    if summary['device'] != 'cpu' or summary['threads'] != 1 or summary['concurrency'] != 1:
        raise ValueError('Unexpected benchmark hardware settings')
    model = TransformerReranker.load(output/'transformer.pt', BASE)
    if (summary['config'] != asdict(model.config) or summary['training'] != model.training_summary
            or summary['base_checkpoint_sha256'] != hashlib.sha256(BASE.read_bytes()).hexdigest()):
        raise ValueError('Transformer checkpoint metadata differs')
    pd.testing.assert_frame_equal(pd.read_csv(output/'training_history.csv'), pd.DataFrame(model.history),
                                  check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    if json.loads((output/'training_audit.json').read_text()) != model.training_audit:
        raise ValueError('Training audit differs from checkpoint')
    audit = model.training_audit
    windows = np.asarray(audit['windows'])
    stamps, ends = np.asarray(audit['timestamps_ns']), np.asarray(audit['label_ends_ns'])
    bounds = np.array([pd.Timestamp(value).value for value in audit['boundaries']])
    if (set(windows) != {2,3} or not (stamps > bounds[windows-1]).all()
            or not (ends <= bounds[windows]).all() or not (ends > stamps).all()
            or len(set(audit['indices'])) != len(windows)
            or hashlib.sha256(np.asarray(audit['indices'], dtype=np.int64).tobytes()).hexdigest() != audit['example_indices_sha256']
            or int((windows==2).sum()) != model.training_summary['training_examples']
            or int((windows==3).sum()) != model.training_summary['validation_examples']):
        raise ValueError('Training temporal window audit is inconsistent')
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    models = model.system.recommenders()
    models[model.name] = model
    metrics, details = evaluate_all(models, split.test)
    for name, computed in [('model_metrics.csv', metrics), ('evaluation_details.csv', details)]:
        pd.testing.assert_frame_equal(pd.read_csv(output/name), computed, check_dtype=False,
                                      check_exact=False, atol=1e-10, rtol=1e-10)
    saved = pd.read_csv(output/'recommendations.csv')
    if set(saved.user_id) != {*model.system.users.user_id, 'new-user'} or set(saved.model) != {model.name}:
        raise ValueError('Incomplete recommendation export')
    for user, group in saved.groupby('user_id'):
        actual = model.recommend(user)
        group = group.sort_values('rank')
        if group.content_id.tolist() != actual.content_id.tolist() or group['rank'].tolist() != list(range(1,len(actual)+1)):
            raise ValueError('Recommendation identity or rank mismatch')
        np.testing.assert_allclose(group.score, actual.score, atol=1e-6, rtol=1e-5)
    if json.loads((output/'quality_deltas.json').read_text()) != quality_deltas(details):
        raise ValueError('Paired quality statistics differ')
    samples = pd.read_csv(output/'latency_samples.csv')
    workers = json.loads((output/'benchmark_workers.json').read_text())
    expected_models = {'v8_pytorch_neural_ranker', model.name}
    expected = {(name, round_index, index) for name in expected_models for round_index in range(summary['repetitions'])
                for index in range(summary['requests_per_worker'])}
    if (samples.duplicated(['model','round','request']).any()
            or set(zip(samples.model,samples['round'],samples.request)) != expected
            or not samples.error.eq(0).all() or not np.isfinite(samples[['wall_ms','cpu_ms']]).all().all()
            or (samples[['wall_ms','cpu_ms']] < 0).any().any()):
        raise ValueError('Incomplete or invalid benchmark request grid')
    if len(workers) != 2*summary['repetitions'] or {(r['model'],r['round']) for r in workers} != {
            (name, index) for name in expected_models for index in range(summary['repetitions'])}:
        raise ValueError('Incomplete benchmark worker grid')
    for worker in workers:
        rows = samples.loc[(samples.model==worker['model']) & (samples['round']==worker['round'])]
        numeric = [worker[key] for key in ('window_wall_seconds','load_seconds','peak_rss_bytes',
                   'parameter_bytes','resident_parameter_bytes','checkpoint_bytes')]
        serving_users = [*model.system.users.user_id, 'new-user']
        expected_users = [serving_users[index%len(serving_users)] for index in range(len(rows))]
        modules = [model.system.tower, model.network if worker['model']==model.name else model.system.ranker]
        expected_parameters = sum(p.numel()*p.element_size() for module in modules for p in module.parameters())
        unused_ranker = sum(p.numel()*p.element_size() for p in model.system.ranker.parameters()) if worker['model']==model.name else 0
        expected_checkpoint = BASE.stat().st_size + ((output/'transformer.pt').stat().st_size if worker['model']==model.name else 0)
        if (worker['threads'] != 1 or worker['requests'] != len(rows) or worker['warmups'] != 10
                or not np.isfinite(numeric).all() or (np.asarray(numeric)<=0).any()
                or worker['parameter_bytes'] != expected_parameters
                or worker['resident_parameter_bytes'] != expected_parameters+unused_ranker
                or worker['checkpoint_bytes'] != expected_checkpoint
                or rows.sort_values('request').user_id.tolist() != expected_users
                or worker['peak_rss_bytes'] <= 0 or worker['window_wall_seconds'] <= 0
                or worker['window_wall_seconds']*1000 < rows.wall_ms.sum()):
            raise ValueError('Benchmark worker metadata differs from requests')
    pd.testing.assert_frame_equal(pd.read_csv(output/'performance.csv'), performance(samples, workers),
                                  check_dtype=False, check_exact=False, atol=1e-9, rtol=1e-9)
    return {'files':len(hashes), 'recommendation_users':saved.user_id.nunique(), 'benchmark_requests':len(samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--requests', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=4)
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--benchmark-worker', choices=['neural','transformer'])
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.benchmark_worker:
        benchmark_worker(args.output, args.benchmark_worker, args.requests, args.round)
    elif args.verify_only:
        seed_cpu(42)
        print(json.dumps(verify(args.output)))
    else:
        run(args.output, TransformerConfig(epochs=args.epochs), args.requests, args.repetitions)


if __name__ == '__main__':
    main()
