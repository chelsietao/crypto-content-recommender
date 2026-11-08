import hashlib
import json
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')
from src.config import REPORTS_DIR
from src.gpu_report import BASE_FILES, KEYS, MODELS, MODES, paired_comparisons, summarize_timings, verify_report
from src.profile_gpu import rotating_schedule, time_rotating_variants, verify_checkpoint


def test_rotating_schedule_balances_every_mode_position():
    schedule = rotating_schedule(6)
    counts = Counter((mode, position) for _, modes in schedule for position, mode in enumerate(modes))
    assert len(counts) == 9 and set(counts.values()) == {2}
    assert [rep for rep, _ in schedule] == list(range(6))
    assert all(set(modes) == set(MODES) for _, modes in schedule)


def test_rotating_timing_preserves_round_coordinates():
    engine = SimpleNamespace(device=torch.device('cpu'), dtype=torch.float32)
    batch = SimpleNamespace(users=torch.zeros(2, 1))
    def timed(engine, batch, model, mode, iterations, repetitions, **settings):
        assert repetitions == 1 and settings['warmup'] == 0
        return [{'mode': mode, 'repetition': settings['repetition_start'],
                 'order_position': settings['order_position']}], []
    with patch('src.profile_gpu.execute') as execute:
        with patch('src.profile_gpu.time_variant', side_effect=timed):
            records, _, schedule = time_rotating_variants(engine, batch, MODELS[0], 2, 6)
    assert execute.call_count == 6
    assert len(records) == len(schedule) == 18
    assert Counter((row['mode'], row['order_position']) for row in records) == {
        (mode, position): 2 for mode in MODES for position in range(3)}


@pytest.fixture
def report(tmp_path):
    # Synthetic statistics fixture, separate from published device measurements.
    rows = []
    groups = [('cpu', 'float32', model, 'batched', 1) for model in MODELS]
    groups += [('mps', precision, model, mode, 1) for precision in ['float32', 'float16']
               for model in MODELS for mode in MODES]
    schedule = []
    for device, precision, model, mode, size in groups:
        for rep in range(6):
            position = 0 if device == 'cpu' else (MODES.index(mode) - rep % 3) % 3
            schedule.append(dict(zip(KEYS, [device, precision, model, mode, size]),
                                 repetition=rep, order_position=position))
            for iteration in range(2):
                duration = {'per_user_loop': 40, 'batched_baseline': 10, 'batched': 5}[mode] * (1 + rep / 10)
                rows.append(dict(zip(KEYS, [device, precision, model, mode, size]), repetition=rep,
                    iteration=iteration, order_position=position, batch_latency_ms=duration))
    timings = pd.DataFrame(rows)
    summary = summarize_timings(timings)
    precision = pd.DataFrame([{'device': device, 'precision': dtype, 'model': model, 'user_id': 'test-user',
        'exact_id_order': True, 'max_common_score_error': 0.0, 'common_score_items': 10}
        for device, dtype, model in sorted(set((row[0], row[1], row[2]) for row in groups))])
    variants = pd.DataFrame([dict(zip(KEYS, group), request_index=0, user_id='test-user',
        exact_id_order=True, max_common_score_error=0.0, common_score_items=10) for group in groups])
    meta = {'format_version': 2, 'iterations': 2, 'repetitions': 6, 'batch_sizes': [1],
        'validation_users': 1, 'validation_user_ids': ['test-user'], 'precision_validation_rows': len(precision),
        'variant_validation_rows': len(variants), 'timed_variants_validated': len(summary),
        'summary': summary.to_dict('records')}
    files = BASE_FILES | {'variant_validation.csv', 'mode_schedule.csv', 'paired_comparison.csv'}
    for name in files:
        (tmp_path / name).write_text('{}')
    for name, frame in [('timings.csv', timings), ('summary.csv', summary), ('precision_validation.csv', precision),
        ('variant_validation.csv', variants), ('mode_schedule.csv', pd.DataFrame(schedule)),
        ('paired_comparison.csv', paired_comparisons(timings))]:
        frame.to_csv(tmp_path / name, index=False)
    (tmp_path / 'results.json').write_text(json.dumps(meta))
    update_hashes(tmp_path, files)
    return tmp_path


def update_hashes(folder, names=None):
    if names is None:
        names = json.loads((folder / 'manifest.json').read_text()).keys()
    (folder / 'manifest.json').write_text(json.dumps({name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                                                    for name in names}))


def test_paired_comparison_uses_same_round_and_valid_report(report):
    pairs = pd.read_csv(report / 'paired_comparison.csv')
    np.testing.assert_allclose(pairs.loc[pairs.baseline_mode == 'batched_baseline', 'throughput_ratio'], 2)
    np.testing.assert_allclose(pairs.loc[pairs.baseline_mode == 'per_user_loop', 'throughput_ratio'], 8)
    summary = pd.read_csv(report / 'summary.csv').iloc[0]
    assert summary['requests_per_second'] == 160
    assert summary['p50_batch_ms'] == 6.25
    assert summary['p95_batch_ms'] == 7.5
    assert verify_report(report)['timing_samples'] == 168


@pytest.mark.parametrize('mutation', ['summary', 'paired_ratio', 'schedule', 'missing_variant',
                                    'missing_timing', 'nan_timing', 'fp32_failure', 'no_score_comparison', 'wrong_request_user'])
def test_report_rejects_semantic_errors_even_with_updated_hashes(report, mutation):
    if mutation == 'summary':
        name = 'summary.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'requests_per_second'] *= 2
    elif mutation == 'paired_ratio':
        name = 'paired_comparison.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'throughput_ratio'] *= 2
    elif mutation == 'schedule':
        name = 'mode_schedule.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'order_position'] = 9
    elif mutation == 'missing_variant':
        name = 'variant_validation.csv'; frame = pd.read_csv(report / name).iloc[1:]
        meta = json.loads((report / 'results.json').read_text())
        meta['variant_validation_rows'] = len(frame)
        (report / 'results.json').write_text(json.dumps(meta))
    elif mutation == 'missing_timing':
        name = 'timings.csv'; frame = pd.read_csv(report / name).iloc[1:]
    elif mutation == 'nan_timing':
        name = 'timings.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'batch_latency_ms'] = np.nan
    elif mutation == 'fp32_failure':
        name = 'precision_validation.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'exact_id_order'] = False
    elif mutation == 'wrong_request_user':
        name = 'variant_validation.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'user_id'] = 'other-user'
    else:
        name = 'variant_validation.csv'; frame = pd.read_csv(report / name); frame.loc[0, 'common_score_items'] = 0
    frame.to_csv(report / name, index=False)
    update_hashes(report)
    with pytest.raises(ValueError):
        verify_report(report)


def test_plot_refuses_hash_mismatch_before_writing(report):
    from src.plot_gpu_profile import plot
    path = report / 'summary.csv'
    path.write_text(path.read_text().replace('200.0', '201.0') + '\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        plot(report)
    assert not (report / 'throughput.png').exists()


@pytest.mark.parametrize('mutation', ['empty', 'missing'])
def test_report_rejects_incomplete_manifests(report, mutation):
    hashes = json.loads((report / 'manifest.json').read_text())
    if mutation == 'empty':
        hashes = {}
    else:
        hashes.pop('timings.csv')
    (report / 'manifest.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError):
        verify_report(report)


def test_existing_published_report_still_verifies():
    result = verify_report(REPORTS_DIR / 'gpu_profile_review')
    assert result['scenarios'] == 56 and result['variant_rows'] == 2366


@pytest.mark.parametrize('valid', [True, False])
def test_checkpoint_gate_checks_file_hash(tmp_path, valid):
    path = tmp_path / 'checkpoint.pt'; path.write_bytes(b'checkpoint fixture')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({path.name: digest if valid else '0' * 64}))
    if valid:
        assert verify_checkpoint(path) == digest
    else:
        with pytest.raises(ValueError, match='checkpoint hash'):
            verify_checkpoint(path)
