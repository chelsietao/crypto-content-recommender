"""Measure a separate Uvicorn process over loopback HTTP with result validation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import psutil
import torch

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.neural import NeuralSystem
from src.serve_neural import DEFAULT_CHECKPOINT, MODEL_NAMES, RecommendationResponse


def validate_response(response, expected, user, model, checkpoint_hash, training_cutoff):
    response.raise_for_status()
    if response.status_code != 200:
        raise AssertionError('Unexpected recommendation status')
    payload = RecommendationResponse.model_validate(response.json())
    if (payload.user_id, payload.model, payload.checkpoint_sha256, payload.training_cutoff) != (
            user, model, checkpoint_hash, training_cutoff):
        raise AssertionError('Response identity or cutoff mismatch')
    if [row.content_id for row in payload.items] != expected[0]:
        raise AssertionError('Recommendation IDs or ordering mismatch')
    scores = np.array([row.score for row in payload.items])
    expected_scores = np.asarray(expected[1])
    if scores.shape != expected_scores.shape or not np.isfinite(scores).all() or not np.allclose(
            scores, expected_scores, rtol=1e-5, atol=1e-6):
        raise AssertionError('Recommendation scores mismatch')


async def measure(base_url, model, users, expected, checkpoint_hash, training_cutoff, concurrency, count, repetition):
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(base_url=base_url, limits=limits, timeout=10, trust_env=False) as client:
        # Warm the connection pool and each worker path outside the measured interval.
        async def warm(worker):
            user = users[worker % len(users)]
            for _ in range(5):
                response = await client.post('/recommend', json={'user_id': user, 'model': model})
                validate_response(response, expected[(model, user)], user, model, checkpoint_hash, training_cutoff)
        await asyncio.gather(*(warm(worker) for worker in range(concurrency)))
        records = []
        async def worker(worker_id):
            for index in range(worker_id, count, concurrency):
                user = users[(index + repetition * count) % len(users)]
                started = time.perf_counter()
                error = ''
                status = 0
                try:
                    response = await client.post('/recommend', json={'user_id': user, 'model': model})
                    status = response.status_code
                    validate_response(response, expected[(model, user)], user, model, checkpoint_hash, training_cutoff)
                except (httpx.HTTPError, AssertionError, KeyError, ValueError, TypeError) as exc:
                    error = type(exc).__name__
                records.append({'model': model, 'concurrency': concurrency, 'repetition': repetition,
                    'request_index': index, 'user_id': user, 'status_code': status,
                    'latency_ms': (time.perf_counter() - started) * 1000, 'error': error})
        started = time.perf_counter()
        await asyncio.gather(*(worker(index) for index in range(concurrency)))
        return records, time.perf_counter() - started


def summarize(records, elapsed_seconds):
    successful = [row['latency_ms'] for row in records if not row['error']]
    quantiles = np.percentile(successful, [50, 95, 99]).tolist() if successful else [None] * 3
    return {'requests': len(records), 'successful_requests': len(successful),
        'errors': len(records) - len(successful),
        'error_rate': (len(records) - len(successful)) / len(records),
        'p50_ms': quantiles[0], 'p95_ms': quantiles[1], 'p99_ms': quantiles[2],
        'elapsed_seconds': elapsed_seconds, 'successful_requests_per_second': len(successful) / elapsed_seconds}


def run_benchmark(output: Path, checkpoint: Path = DEFAULT_CHECKPOINT, requests: int = 500,
                  concurrencies=(1, 4, 16), repetitions: int = 3):
    if requests < 1 or repetitions < 1 or not concurrencies or any(c < 1 or c > requests for c in concurrencies):
        raise ValueError('Requests and repetitions must be positive, with concurrency between 1 and requests')
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint.resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    torch.set_num_threads(1)
    system = NeuralSystem.load(checkpoint)
    users = [*system.users.user_id, 'new-user']
    expected = {}
    for model, recommender in system.recommenders().items():
        for user in users:
            result = recommender.recommend(user)
            expected[(model, user)] = (result.content_id.tolist(), result.score.to_numpy())
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base_url = f'http://127.0.0.1:{port}'
    command = [sys.executable, '-m', 'src.serve_neural', '--port', str(port),
               '--checkpoint', str(checkpoint), '--torch-threads', '1']
    all_records, summaries, repeats = [], [], []
    with (output / 'server.log').open('w') as log:
        started = time.perf_counter()
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=log)
        try:
            with httpx.Client(base_url=base_url, timeout=1, trust_env=False) as client:
                deadline = time.monotonic() + 60
                while True:
                    if process.poll() is not None:
                        raise RuntimeError('Inference server exited during startup, see server.log')
                    try:
                        response = client.get('/ready')
                        if response.status_code == 200:
                            ready = response.json()
                            if ready['pid'] != process.pid or ready['checkpoint_sha256'] != checkpoint_hash:
                                raise RuntimeError('Readiness identity mismatch')
                            break
                    except httpx.TransportError:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Inference server readiness timed out')
                    time.sleep(0.05)
            startup_seconds = time.perf_counter() - started
            server = psutil.Process(process.pid)
            for model in MODEL_NAMES:
                for concurrency in concurrencies:
                    combined, total = [], 0.0
                    rss_before = server.memory_info().rss
                    for repetition in range(repetitions):
                        records, elapsed = asyncio.run(measure(base_url, model, users, expected,
                            checkpoint_hash, system.reference.isoformat(), concurrency, requests, repetition))
                        combined.extend(records)
                        total += elapsed
                        repeats.append({'model': model, 'concurrency': concurrency,
                            'repetition': repetition, **summarize(records, elapsed)})
                    summaries.append({'model': model, 'concurrency': concurrency,
                        **summarize(combined, total), 'server_rss_before_mib': rss_before / 2**20,
                        'server_rss_after_mib': server.memory_info().rss / 2**20})
                    all_records.extend(combined)
                    print(json.dumps(summaries[-1]), flush=True)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    cpu_name = platform.processor()
    if sys.platform == 'darwin':
        cpu_name = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    report = {'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': 'HTTP/1.1 over loopback, closed-loop persistent connections, no retries',
        'latency_scope': 'client request construction through HTTP parsing, strict response schema and result validation',
        'percentile_method': 'numpy.percentile linear, successful measured requests only',
        'throughput_scope': 'successful requests divided by total measured wall time',
        'warmup_requests_per_worker_per_repetition': 5,
        'requests_per_repetition': requests, 'repetitions': repetitions,
        'users': len(users), 'known_users': len(system.users), 'new_users': 1,
        'k': 10, 'exclude_seen': True, 'readiness': ready, 'startup_seconds': startup_seconds,
        'hardware': {'cpu': cpu_name, 'machine': platform.machine(), 'os': platform.platform(),
            'physical_cores': psutil.cpu_count(logical=False), 'logical_cores': psutil.cpu_count(),
            'ram_gib': psutil.virtual_memory().total / 2**30, 'device': 'cpu',
            'client_and_server_same_host': True},
        'versions': {name: importlib.metadata.version(name) for name in
                     ('torch', 'fastapi', 'uvicorn', 'httpx', 'pydantic', 'numpy', 'pandas')},
        'python': platform.python_version(),
        'source_sha256': {name: hashlib.sha256((PROJECT_ROOT / 'src' / name).read_bytes()).hexdigest()
                          for name in ('serve_neural.py', 'benchmark_service.py', 'neural.py')},
        'scenarios': summaries, 'repetition_results': repeats}
    pd.DataFrame(all_records).to_csv(output / 'requests.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
              for name in ('requests.csv', 'summary.csv', 'results.json')}
    (output / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    if any(row['errors'] for row in summaries):
        raise RuntimeError('Benchmark recorded request or result validation errors, see results.json')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'service_benchmark')
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--requests', type=int, default=500)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 4, 16])
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    run_benchmark(args.output, args.checkpoint, args.requests, args.concurrency, args.repetitions)


if __name__ == '__main__':
    main()
