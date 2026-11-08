"""Recompute GPU experiment statistics and validate exported report integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ['device', 'precision', 'model', 'mode', 'batch_size']
MODES = ('per_user_loop', 'batched_baseline', 'batched')
MODELS = ('v7_pytorch_two_tower', 'v8_pytorch_neural_ranker')
BASE_FILES = {'timings.csv', 'summary.csv', 'precision_validation.csv', 'stages.csv',
    'memory_samples.csv', 'input_preparation.csv', 'cpu_operators.csv',
    'cpu_dispatch_trace.json', 'sort_calls.csv', 'results.json'}


def summarize_timings(frame):
    rows = []
    for keys, group in frame.groupby(KEYS, sort=False):
        device, precision, model, mode, size = keys
        elapsed = group.batch_latency_ms.to_numpy()
        p50, p95 = np.percentile(elapsed, [50, 95])
        rows.append({'device': device, 'precision': precision, 'model': model, 'mode': mode,
            'batch_size': int(size), 'samples': len(group), 'p50_batch_ms': p50, 'p95_batch_ms': p95,
            'requests_per_second': int(size) * len(group) * 1000 / elapsed.sum()})
    return pd.DataFrame(rows)


def paired_comparisons(frame):
    frame = frame.loc[frame.device == 'mps']
    rows = []
    for keys, group in frame.groupby(['precision', 'model', 'batch_size', 'repetition'], sort=False):
        precision, model, size, repetition = keys
        rates = {mode: size * len(values) * 1000 / values.batch_latency_ms.sum()
                 for mode, values in group.groupby('mode')}
        for baseline in ('per_user_loop', 'batched_baseline'):
            rows.append({'precision': precision, 'model': model, 'batch_size': int(size),
                'repetition': int(repetition), 'baseline_mode': baseline,
                'baseline_requests_per_second': rates[baseline],
                'optimized_requests_per_second': rates['batched'],
                'throughput_ratio': rates['batched'] / rates[baseline]})
    return pd.DataFrame(rows)


def _same_table(expected, actual, keys):
    try:
        pd.testing.assert_frame_equal(expected.sort_values(keys).reset_index(drop=True),
            actual[expected.columns].sort_values(keys).reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    except (AssertionError, KeyError) as exc:
        raise ValueError('Exported statistics or coverage differ from raw measurements') from exc


def verify_report(output: Path):
    output = Path(output)
    hashes = json.loads((output / 'manifest.json').read_text())
    if not isinstance(hashes, dict) or not BASE_FILES <= set(hashes):
        raise ValueError('GPU manifest is missing required files')
    # Check every declared hash before consuming statistics or metadata.
    for name, expected in hashes.items():
        if Path(name).name != name or not isinstance(expected, str):
            raise ValueError('Invalid GPU manifest entry')
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'GPU artifact hash mismatch: {name}')
    meta = json.loads((output / 'results.json').read_text())
    required = set(BASE_FILES)
    if 'variant_validation_rows' in meta:
        required.add('variant_validation.csv')
    if meta.get('format_version', 1) >= 2:
        required.update({'mode_schedule.csv', 'paired_comparison.csv'})
    if set(hashes) not in (required, required | {'throughput.png'}):
        raise ValueError('GPU manifest file coverage does not match the report format')
    iterations, repetitions = meta['iterations'], meta['repetitions']
    sizes = meta['batch_sizes']
    if (isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1
            or isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 1):
        raise ValueError('Invalid experiment iteration counts')
    if (not sizes or any(isinstance(size, bool) or not isinstance(size, int) or size < 1 for size in sizes)
            or len(set(sizes)) != len(sizes)):
        raise ValueError('Invalid experiment batch sizes')
    expected_groups = {('cpu', 'float32', model, 'batched', size) for model in MODELS for size in sizes}
    expected_groups.update(('mps', precision, model, mode, size)
        for precision in ('float32', 'float16') for model in MODELS for mode in MODES for size in sizes)
    frame = pd.read_csv(output / 'timings.csv')
    if set(map(tuple, frame[KEYS].drop_duplicates().to_numpy())) != expected_groups:
        raise ValueError('Timing data is missing or adds experiment groups')
    if not np.isfinite(frame.batch_latency_ms).all() or not frame.batch_latency_ms.gt(0).all():
        raise ValueError('Timing data must contain positive finite durations')
    if frame.duplicated(KEYS + ['repetition', 'iteration']).any():
        raise ValueError('Timing data contains duplicate sample coordinates')
    for _, group in frame.groupby(KEYS):
        actual = set(zip(group.repetition, group.iteration))
        expected = {(rep, index) for rep in range(repetitions) for index in range(iterations)}
        if actual != expected:
            raise ValueError('Timing samples do not cover every repetition and iteration')
    computed = summarize_timings(frame)
    _same_table(computed, pd.read_csv(output / 'summary.csv'), KEYS)
    _same_table(computed, pd.DataFrame(meta['summary']), KEYS)
    precision = pd.read_csv(output / 'precision_validation.csv')
    expected_precision = {(device, dtype, model) for device, dtype, model, _, _ in expected_groups}
    if set(map(tuple, precision[['device', 'precision', 'model']].drop_duplicates().to_numpy())) != expected_precision:
        raise ValueError('Precision audit group coverage is incomplete')
    if precision.duplicated(['device', 'precision', 'model', 'user_id']).any():
        raise ValueError('Precision audit contains duplicate user rows')
    for _, group in precision.groupby(['device', 'precision', 'model']):
        if len(group) != meta['validation_users']:
            raise ValueError('Precision audit does not cover all users')
        if 'validation_user_ids' in meta and set(group.user_id) != set(meta['validation_user_ids']):
            raise ValueError('Precision audit contains the wrong users')
    if 'precision_validation_rows' in meta and len(precision) != meta['precision_validation_rows']:
        raise ValueError('Precision audit row count is inconsistent')
    if 'timed_variants_validated' in meta and len(computed) != meta['timed_variants_validated']:
        raise ValueError('Declared scenario coverage is inconsistent')
    checked = [precision]
    if 'variant_validation.csv' in required:
        variants = pd.read_csv(output / 'variant_validation.csv')
        if len(variants) != meta['variant_validation_rows']:
            raise ValueError('Variant audit row count is inconsistent')
        if set(map(tuple, variants[KEYS].drop_duplicates().to_numpy())) != expected_groups:
            raise ValueError('Variant audit group coverage is incomplete')
        for keys, group in variants.groupby(KEYS):
            if len(group) != keys[-1] or sorted(group.request_index) != list(range(keys[-1])):
                raise ValueError('Variant audit does not cover each request exactly once')
            if 'validation_user_ids' in meta:
                users = meta['validation_user_ids']
                expected_users = [users[index % len(users)] for index in range(keys[-1])]
                if group.sort_values('request_index').user_id.tolist() != expected_users:
                    raise ValueError('Variant audit request users differ from experiment inputs')
        checked.append(variants)
    for table in checked:
        fp32 = table.loc[table.precision == 'float32']
        if 'common_score_items' in fp32 and not fp32.common_score_items.gt(0).all():
            raise ValueError('FP32 scores were not compared against reference items')
        if (not fp32.exact_id_order.eq(True).all() or not np.isfinite(fp32.max_common_score_error).all()
                or fp32.max_common_score_error.gt(1e-5).any()):
            raise ValueError('FP32 output equivalence failed')
    if meta.get('format_version', 1) >= 2:
        users = meta.get('validation_user_ids', [])
        if len(users) != meta['validation_users'] or len(set(users)) != len(users):
            raise ValueError('Experiment user inventory is incomplete')
        schedule = pd.read_csv(output / 'mode_schedule.csv')
        expected = []
        for device, dtype, model, mode, size in expected_groups:
            for rep in range(repetitions):
                position = 0 if device == 'cpu' else (MODES.index(mode) - rep % len(MODES)) % len(MODES)
                expected.append(dict(zip(KEYS, (device, dtype, model, mode, size)),
                                     repetition=rep, order_position=position))
        _same_table(pd.DataFrame(expected), schedule, KEYS + ['repetition'])
        if not set(KEYS + ['repetition', 'order_position']) <= set(frame.columns):
            raise ValueError('Timing records do not identify measurement order')
        _same_table(schedule, frame[KEYS + ['repetition', 'order_position']].drop_duplicates(), KEYS + ['repetition'])
        _same_table(paired_comparisons(frame), pd.read_csv(output / 'paired_comparison.csv'),
                    ['precision', 'model', 'batch_size', 'repetition', 'baseline_mode'])
    return {'format_version': meta.get('format_version', 1), 'files': len(hashes),
            'scenarios': len(computed), 'timing_samples': len(frame),
            'precision_rows': len(precision), 'variant_rows': len(checked[-1]) if len(checked) > 1 else 0}
