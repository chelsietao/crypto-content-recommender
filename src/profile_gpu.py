"""Synchronized MPS GPU stage profiling and batched/precision experiments."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.gpu_inference import BatchedNeuralInference, MODEL_NAMES
from src.neural import NeuralSystem


def synchronize(device):
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()


def memory_sample():
    return {'tensor_allocated_bytes': torch.mps.current_allocated_memory(),
            'driver_allocated_bytes': torch.mps.driver_allocated_memory()}


def execute(engine, batch, model, mode):
    if mode in ('batched', 'batched_baseline'):
        return engine.forward(batch, model, optimized=mode == 'batched')
    rows = [engine.forward(batch.row(index), model) for index in range(len(batch.users))]
    return tuple(torch.cat([row[index] for row in rows], dim=0) for index in (0, 1))


def time_variant(engine, batch, model, mode, iterations, repetitions):
    for _ in range(2):
        execute(engine, batch, model, mode)
    synchronize(engine.device.type)
    records = []
    memory = []
    for repetition in range(repetitions):
        for iteration in range(iterations):
            synchronize(engine.device.type)
            started = time.perf_counter_ns()
            result = execute(engine, batch, model, mode)
            synchronize(engine.device.type)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            records.append({'device': engine.device.type, 'precision': str(engine.dtype).split('.')[-1],
                'model': model, 'mode': mode, 'batch_size': len(batch.users),
                'repetition': repetition, 'iteration': iteration, 'batch_latency_ms': elapsed_ms})
            del result
        if engine.device.type == 'mps':
            memory.append(memory_sample())
    return records, memory


@torch.inference_mode()
def stage_probe(engine, batch, model, iterations, optimized=True):
    records, samples = [], []
    variant = 'optimized' if optimized else 'baseline'
    for iteration in range(iterations):
        synchronize('mps')
        started = time.perf_counter_ns()
        with torch.profiler.record_function(f'stage/{model}/{variant}/user_tower'):
            users = engine.encode(batch)
            synchronize('mps')
        records.append({'stage': 'user_tower', 'iteration': iteration,
                        'latency_ms': (time.perf_counter_ns() - started) / 1e6})
        started = time.perf_counter_ns()
        with torch.profiler.record_function(f'stage/{model}/{variant}/retrieval'):
            candidates, scores = engine.retrieve(batch, users)
            synchronize('mps')
        records.append({'stage': 'retrieval', 'iteration': iteration,
                        'latency_ms': (time.perf_counter_ns() - started) / 1e6})
        if model == MODEL_NAMES[1]:
            started = time.perf_counter_ns()
            with torch.profiler.record_function(f'stage/{model}/{variant}/neural_ranker'):
                scores = engine.rank(batch, users, candidates, scores)
                synchronize('mps')
            records.append({'stage': 'neural_ranker', 'iteration': iteration,
                            'latency_ms': (time.perf_counter_ns() - started) / 1e6})
        started = time.perf_counter_ns()
        with torch.profiler.record_function(f'stage/{model}/{variant}/final_sort'):
            output = engine.finish(candidates, scores, 10, presorted=model == MODEL_NAMES[0], optimized=optimized)
            synchronize('mps')
        records.append({'stage': 'final_sort', 'iteration': iteration,
                        'latency_ms': (time.perf_counter_ns() - started) / 1e6})
        samples.append(memory_sample())
        del output, users, candidates, scores
    return records, samples


def validate_precision(engine, user_ids, model, references):
    rows = engine.decode(*engine.forward(engine.prepare(user_ids), model))
    records = []
    for user, actual in zip(user_ids, rows):
        expected_ids, reference_scores = references[(model, user)]
        actual_ids = [item for item, _ in actual]
        errors = [abs(score - reference_scores[item]) for item, score in actual if item in reference_scores]
        records.append({'device': engine.device.type, 'precision': str(engine.dtype).split('.')[-1],
            'model': model, 'user_id': user, 'exact_id_order': actual_ids == expected_ids,
            'top10_overlap': len(set(actual_ids) & set(expected_ids)) / max(len(expected_ids), 1),
            'max_common_score_error': max(errors, default=0),
            'items_outside_cpu_candidates': sum(item not in reference_scores for item, _ in actual)})
    return records


def profile(output: Path, batch_sizes=(1, 8, 32, 128), iterations=10, repetitions=3):
    if not torch.backends.mps.is_available():
        raise RuntimeError('An available Apple MPS GPU is required, CPU fallback is not used')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Set PYTORCH_ENABLE_MPS_FALLBACK=0 for this GPU experiment')
    if not batch_sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in batch_sizes):
        raise ValueError('Batch sizes must be positive integers')
    if iterations < 1 or repetitions < 1:
        raise ValueError('Iterations and repetitions must be positive')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    checkpoint = ARTIFACTS_DIR / 'neural' / 'neural_recommender.pt'
    system = NeuralSystem.load(checkpoint)
    user_ids = [*system.users.user_id, 'new-user']
    references = {}
    for model, recommender in system.recommenders().items():
        for user in user_ids:
            expected = recommender.recommend(user)
            all_scores = recommender.score_all(user)
            references[(model, user)] = (expected.content_id.tolist(),
                {content: float(score) for content, score in zip(system.encoder.content_ids, all_scores) if np.isfinite(score)})
    records, validation, memory, transfers = [], [], [], []
    # CPU vectorized FP32 is measured on the same inputs as a hardware reference.
    cpu = BatchedNeuralInference(system)
    for model in MODEL_NAMES:
        validation.extend(validate_precision(cpu, user_ids, model, references))
        for size in batch_sizes:
            users = [user_ids[index % len(user_ids)] for index in range(size)]
            measured, _ = time_variant(cpu, cpu.prepare(users), model, 'batched', iterations, repetitions)
            records.extend(measured)
    del cpu
    for dtype in (torch.float32, torch.float16):
        engine = BatchedNeuralInference(system, 'mps', dtype)
        for model in MODEL_NAMES:
            validation.extend(validate_precision(engine, user_ids, model, references))
            for size in batch_sizes:
                users = [user_ids[index % len(user_ids)] for index in range(size)]
                started = time.perf_counter_ns()
                batch = engine.prepare(users)
                synchronize('mps')
                transfers.append({'precision': str(dtype).split('.')[-1], 'model': model, 'batch_size': size,
                    'prepare_and_transfer_ms': (time.perf_counter_ns() - started) / 1e6})
                for mode in ('per_user_loop', 'batched_baseline', 'batched'):
                    measured, samples = time_variant(engine, batch, model, mode, iterations, repetitions)
                    records.extend(measured)
                    memory.extend({'precision': str(dtype).split('.')[-1], 'model': model,
                        'batch_size': size, 'mode': mode, 'sample': index, **sample}
                        for index, sample in enumerate(samples))
                print(f'{str(dtype)} {model} batch={size} complete', flush=True)
                del batch
        del engine
        gc.collect()
        torch.mps.empty_cache()
    # Stage timings and CPU dispatch trace are separate from throughput measurements.
    engine = BatchedNeuralInference(system, 'mps', torch.float32)
    batch = engine.prepare([user_ids[index % len(user_ids)] for index in range(max(batch_sizes))])
    stages = []
    for model in MODEL_NAMES:
        for optimized in (False, True):
            engine.forward(batch, model, optimized=optimized)
            synchronize('mps')
            measured, samples = stage_probe(engine, batch, model, iterations, optimized)
            stages.extend({'model': model, 'batch_size': max(batch_sizes),
                'variant': 'optimized' if optimized else 'baseline', **row} for row in measured)
            memory.extend({'precision': 'float32', 'model': model, 'batch_size': max(batch_sizes),
                'mode': 'stage_probe', 'sample': index, **sample} for index, sample in enumerate(samples))
    # Capture dispatch evidence in a separate pass, avoiding profiler overhead in timings.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as trace:
        with torch.mps.profiler.profile(mode='interval', wait_until_completed=False):
            for model in MODEL_NAMES:
                for optimized in (False, True):
                    stage_probe(engine, batch, model, 1, optimized)
    trace.export_chrome_trace(str(output / 'cpu_dispatch_trace.json'))
    sort_counts = {}
    for event in trace.events():
        if event.name not in ('aten::sort', 'aten::argsort'):
            continue
        parent = event.cpu_parent
        while parent is not None and not parent.name.startswith('stage/'):
            parent = parent.cpu_parent
        if parent is not None:
            _, model, variant, stage = parent.name.split('/')
            key = (model, variant, stage, event.name)
            sort_counts[key] = sort_counts.get(key, 0) + 1
    pd.DataFrame([{'model': model, 'variant': variant, 'stage': stage, 'operator': operator, 'calls': count}
        for (model, variant, stage, operator), count in sort_counts.items()]).to_csv(output / 'sort_calls.csv', index=False)
    operators = [{'operator': row.key, 'calls': row.count, 'self_cpu_time_us': row.self_cpu_time_total,
                  'cpu_time_us': row.cpu_time_total} for row in trace.key_averages()]
    frame = pd.DataFrame(records)
    summary = []
    for keys, group in frame.groupby(['device', 'precision', 'model', 'mode', 'batch_size'], sort=False):
        device, precision, model, mode, size = keys
        elapsed = group.batch_latency_ms.to_numpy()
        p50, p95 = np.percentile(elapsed, [50, 95])
        summary.append({'device': device, 'precision': precision, 'model': model, 'mode': mode,
            'batch_size': int(size), 'samples': len(group), 'p50_batch_ms': p50, 'p95_batch_ms': p95,
            'requests_per_second': int(size) * len(group) * 1000 / elapsed.sum()})
    validation_frame = pd.DataFrame(validation)
    fp32 = validation_frame.loc[validation_frame.precision == 'float32']
    if not fp32.exact_id_order.all() or fp32.max_common_score_error.max() > 1e-5:
        raise AssertionError('FP32 batched inference differs from checkpoint recommendations')
    metadata = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'device': 'Apple GPU via MPS', 'soc': subprocess.check_output(
            ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
        'os': platform.platform(), 'ram_gib': psutil.virtual_memory().total / 2**30,
        'python': platform.python_version(), 'torch': torch.__version__, 'cpu_threads': 1,
        'mps_cpu_fallback': False, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'iterations': iterations, 'repetitions': repetitions, 'batch_sizes': list(batch_sizes),
        'warmup_batches': 2, 'validation_users': len(user_ids),
        'timing_scope': 'prepared device inputs through GPU-resident top10 outputs, synchronized before and after each sample',
        'stage_timing_scope': 'host dispatch plus GPU completion, with a barrier per stage, separate from throughput samples',
        'trace_scope': 'CPU dispatch operators, not a GPU kernel timeline',
        'trace_iterations_per_model_per_variant': 1,
        'mps_os_signposts_emitted': True,
        'memory_scope': 'MPS tensor and Metal driver bytes sampled after repetitions and stage probes, not a peak counter',
        'recommended_max_memory_bytes': torch.mps.recommended_max_memory(),
        'source_sha256': {name: hashlib.sha256((PROJECT_ROOT / 'src' / name).read_bytes()).hexdigest()
                          for name in ('gpu_inference.py', 'profile_gpu.py', 'neural.py')},
        'summary': summary}
    for name, data in [('timings.csv', frame), ('summary.csv', pd.DataFrame(summary)),
                       ('precision_validation.csv', validation_frame), ('stages.csv', pd.DataFrame(stages)),
                       ('memory_samples.csv', pd.DataFrame(memory)), ('input_preparation.csv', pd.DataFrame(transfers)),
                       ('cpu_operators.csv', pd.DataFrame(operators))]:
        data.to_csv(output / name, index=False)
    (output / 'results.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    names = ['timings.csv', 'summary.csv', 'precision_validation.csv', 'stages.csv',
             'memory_samples.csv', 'input_preparation.csv', 'cpu_operators.csv', 'cpu_dispatch_trace.json', 'sort_calls.csv', 'results.json']
    (output / 'manifest.json').write_text(json.dumps({name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in names}, indent=2) + '\n')
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'gpu_profile')
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 8, 32, 128])
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    profile(args.output, args.batch_sizes, args.iterations, args.repetitions)


if __name__ == '__main__':
    main()
