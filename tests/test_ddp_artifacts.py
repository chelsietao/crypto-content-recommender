import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest

torch = pytest.importorskip('torch')
from src.config import PROJECT_ROOT, REPORTS_DIR
from src.ddp_artifacts import run_files, write_manifest
from src.process_runner import run_training_process
from src.validate_distributed import verify_exports


@pytest.fixture
def report(tmp_path):
    target = tmp_path / 'report'
    shutil.copytree(REPORTS_DIR / 'ddp_cpu_reference', target)
    return target


def refresh(report):
    for world in (1, 2):
        folder = report / f'round_0_world_{world}'
        files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in sorted(run_files(world))}
        (folder / 'manifest.json').write_text(json.dumps(files))
    write_manifest(report)


def test_manifest_covers_all_artifacts_and_batch_grid(report):
    result = verify_exports(report)
    assert result['files'] == 23
    assert result['verified_global_steps'] == 1440
    assert result['gpu_result'] is False


@pytest.mark.parametrize('mutation', ['comparison_only', 'missing_child_manifest', 'bad_child_hash', 'extra_file'])
def test_complete_manifest_inventory_is_required(report, mutation):
    hashes = json.loads((report / 'manifest.json').read_text())
    if mutation == 'comparison_only':
        hashes = {'comparison.json': hashes['comparison.json']}
    elif mutation == 'missing_child_manifest':
        hashes.pop('round_0_world_2/manifest.json')
    elif mutation == 'bad_child_hash':
        path = report / 'round_0_world_2/manifest.json'
        child = json.loads(path.read_text())
        child['neural_recommender.pt'] = '0' * 64
        path.write_text(json.dumps(child))
        write_manifest(report)
        hashes = json.loads((report / 'manifest.json').read_text())
    else:
        (report / 'unexpected.csv').write_text('untracked artifact')
    (report / 'manifest.json').write_text(json.dumps(hashes))
    with pytest.raises(ValueError):
        verify_exports(report)


@pytest.mark.parametrize('mutation', ['whole_global_step', 'whole_epoch', 'one_rank_step', 'duplicate_step',
    'incorrect_global_size', 'shifted_local_counts', 'fractional_counts', 'nan_duration', 'negative_duration'])
def test_step_grid_and_exact_shards_reject_modified_logs(report, mutation):
    folder = report / 'round_0_world_2'
    for rank in (0, 1):
        path = folder / f'steps_rank_{rank}.csv'
        frame = pd.read_csv(path)
        first = (frame.stage == 'two_tower') & (frame.epoch == 1) & (frame.batch_index == 0)
        if mutation == 'whole_global_step' or (mutation == 'one_rank_step' and rank == 1):
            frame = frame.loc[~first]
        elif mutation == 'whole_epoch':
            frame = frame.loc[~((frame.stage == 'two_tower') & (frame.epoch == 1))]
        elif mutation == 'duplicate_step' and rank == 0:
            frame = pd.concat([frame, frame.loc[first]])
        elif mutation == 'incorrect_global_size':
            frame.loc[first, 'global_examples'] = 126
            frame.loc[first, 'local_examples'] = 63
        elif mutation == 'shifted_local_counts':
            frame.loc[first, 'local_examples'] = 63 if rank == 0 else 65
        elif mutation == 'fractional_counts':
            frame['local_examples'] = frame.local_examples.astype(float)
            frame.loc[first, 'local_examples'] = 63.5 if rank == 0 else 64.5
        elif mutation == 'nan_duration':
            frame.loc[first, 'forward_seconds'] = np.nan
        elif mutation == 'negative_duration':
            frame.loc[first, 'optimizer_seconds'] = -1
        frame.to_csv(path, index=False)
    refresh(report)
    with pytest.raises(ValueError):
        verify_exports(report)


@pytest.mark.parametrize('mutation', ['selected_epoch', 'communication_row', 'communication_payload',
    'rank_device', 'missing_user', 'both_score_files', 'both_metric_files', 'evaluation_subset'])
def test_export_binding_rejects_inconsistent_evidence_with_valid_hashes(report, mutation):
    folder = report / 'round_0_world_2'
    if mutation == 'selected_epoch':
        path = folder / 'results.json'; record = json.loads(path.read_text())
        record['training']['two_tower']['selected_epoch'] = 0
        path.write_text(json.dumps(record))
    elif mutation in ['communication_row', 'communication_payload']:
        path = folder / 'communication.csv'; frame = pd.read_csv(path)
        if mutation == 'communication_row':
            frame = frame.iloc[1:]
        else:
            frame.loc[0, 'payload_bytes_per_rank'] += 4
        frame.to_csv(path, index=False)
    elif mutation == 'rank_device':
        path = folder / 'rank_1.json'; record = json.loads(path.read_text())
        record['device'] = 'cuda:1'
        path.write_text(json.dumps(record))
    elif mutation == 'missing_user':
        path = folder / 'recommendations.csv'; frame = pd.read_csv(path)
        frame.loc[frame.user_id != 'U0001'].to_csv(path, index=False)
    else:
        for world in (1, 2):
            selected = report / f'round_0_world_{world}'
            if mutation == 'evaluation_subset':
                from src.neural import NeuralSystem
                from src.evaluation import evaluate_all
                details = pd.read_csv(selected / 'evaluation_details.csv')
                test = details[['user_id', 'target_content_id']].drop_duplicates().iloc[1:]
                models = NeuralSystem.load(selected / 'neural_recommender.pt').recommenders()
                metrics, subset = evaluate_all(models, test.rename(columns={'target_content_id': 'content_id'}))
                metrics.to_csv(selected / 'model_metrics.csv', index=False)
                subset.to_csv(selected / 'evaluation_details.csv', index=False)
                continue
            name = 'recommendations.csv' if mutation == 'both_score_files' else 'model_metrics.csv'
            path = selected / name; frame = pd.read_csv(path)
            if mutation == 'both_score_files':
                frame.loc[0, 'score'] += 0.1
            else:
                frame.loc[0, 'ndcg@10'] += 0.1
            frame.to_csv(path, index=False)
    refresh(report)
    with pytest.raises(ValueError):
        verify_exports(report)


@pytest.mark.parametrize('separate_session', [False, True])
def test_timeout_reaps_launcher_and_worker(tmp_path, separate_session):
    marker = tmp_path / 'worker.pid'
    worker = "import time; time.sleep(60)"
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c',sys.argv[2],'--ddp-timeout-worker'],"
        f"stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session={separate_session}); "
        "Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
    )
    with (tmp_path / 'process.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_training_process([sys.executable, '-c', parent, str(marker), worker],
                                 PROJECT_ROOT, os.environ.copy(), log, timeout=2)
    assert marker.exists()
    pid = int(marker.read_text())
    try:
        process = psutil.Process(pid)
        assert not process.is_running() or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_normal_launcher_exit_code_is_preserved(tmp_path):
    with (tmp_path / 'process.log').open('w') as log:
        assert run_training_process([sys.executable, '-c', 'raise SystemExit(7)'],
                                    PROJECT_ROOT, os.environ.copy(), log, timeout=5) == 7
