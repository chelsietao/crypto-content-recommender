"""Launch one/two-rank training and compare checkpoint parameters and outputs."""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.neural import NeuralSystem
from src.ddp_artifacts import write_manifest, verify_all
from src.process_runner import run_training_process


def compare_runs(first: Path, second: Path, comparison='numerical'):
    if comparison not in ('numerical', 'ranking'):
        raise ValueError('Unknown distributed comparison mode')
    summaries = [json.loads((folder / 'results.json').read_text()) for folder in (first, second)]
    systems = [NeuralSystem.load(folder / 'neural_recommender.pt') for folder in (first, second)]
    if summaries[0]['ranker_example_index_sha256'] != summaries[1]['ranker_example_index_sha256']:
        raise AssertionError('Ranker candidate example identities differ')
    maximum_weight_error = {}
    for name, stage in [('tower', 'two_tower'), ('ranker', 'neural_ranker')]:
        if summaries[0]['training'][stage]['label_partition_sha256'] != summaries[1]['training'][stage]['label_partition_sha256']:
            raise AssertionError('Single and distributed training label partitions differ')
        left, right = [getattr(system, name).state_dict() for system in systems]
        if any(not torch.isfinite(value).all().item() for state in (left, right) for value in state.values()):
            raise AssertionError('Nonfinite distributed checkpoint parameters')
        maximum_weight_error[stage] = max(float(torch.max(torch.abs(left[key] - right[key]))) for key in left)
        if comparison == 'numerical' and maximum_weight_error[stage] > 1e-4:
            raise AssertionError('Single and distributed weights differ beyond tolerance')
    expected = pd.read_csv(first / 'recommendations.csv')
    actual = pd.read_csv(second / 'recommendations.csv')
    keys = ['model', 'user_id', 'rank']
    expected, actual = [frame.sort_values(keys).reset_index(drop=True) for frame in (expected, actual)]
    assert expected[keys + ['content_id']].equals(actual[keys + ['content_id']]), 'Recommendation identities changed'
    if not np.isfinite(expected.score).all() or not np.isfinite(actual.score).all():
        raise AssertionError('Nonfinite recommendation scores')
    if comparison == 'numerical':
        np.testing.assert_allclose(expected.score, actual.score, atol=1e-5, rtol=0)
    pd.testing.assert_frame_equal(pd.read_csv(first / 'model_metrics.csv'), pd.read_csv(second / 'model_metrics.csv'),
                                 check_exact=False, atol=1e-12, rtol=0)
    difference = float(np.max(np.abs(expected.score - actual.score)))
    return {'device': summaries[0]['device'], 'comparison': comparison,
        'training_precision': summaries[0]['training_precision'], 'single_world_size': summaries[0]['world_size'],
        'distributed_world_size': summaries[1]['world_size'],
        'maximum_parameter_error': maximum_weight_error, 'maximum_recommendation_score_error': difference,
        'matched_recommendation_groups': expected.groupby(['model', 'user_id']).ngroups,
        'selected_epochs': [{stage: summary['training'][stage]['selected_epoch'] for stage in ['two_tower', 'neural_ranker']}
                            for summary in summaries],
        'single_fit_seconds': summaries[0]['fit_seconds_max_rank'],
        'distributed_fit_seconds': summaries[1]['fit_seconds_max_rank'],
        'fit_time_ratio': summaries[0]['fit_seconds_max_rank'] / summaries[1]['fit_seconds_max_rank'],
        'scaling_efficiency': (summaries[0]['fit_seconds_max_rank'] / summaries[1]['fit_seconds_max_rank'])
            / (summaries[1]['world_size'] / summaries[0]['world_size'])}


def launch(output: Path, device='cpu', epochs=30, global_batch_size=128, repetitions=1, precision='float32', comparison='numerical'):
    if device not in ('cpu', 'cuda') or epochs < 1 or repetitions < 1:
        raise ValueError('Device, epochs and repetitions must be valid')
    if device == 'cuda' and torch.cuda.device_count() < 2:
        raise RuntimeError('A two-CUDA-GPU host is required for this comparison')
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if device == 'cpu':
        env['GLOO_SOCKET_IFNAME'] = 'lo0' if sys.platform == 'darwin' else 'lo'
    records = []
    for repetition in range(repetitions):
        paths = {}
        for world_size in (1, 2):
            folder = output / f'round_{repetition}_world_{world_size}'
            folder.mkdir(parents=True, exist_ok=True)
            with socket.socket() as probe:
                probe.bind(('127.0.0.1', 0))
                port = probe.getsockname()[1]
            command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
                '--master-addr=127.0.0.1', f'--master-port={port}',
                f'--nproc-per-node={world_size}', '-m', 'src.train_distributed', '--device', device,
                '--precision', precision, '--epochs', str(epochs), '--global-batch-size', str(global_batch_size), '--output', str(folder.resolve())]
            with (folder / 'launcher.log').open('w') as log:
                returncode = run_training_process(command, PROJECT_ROOT, env, log, timeout=300)
            if returncode:
                raise RuntimeError(f'Distributed run failed, see {folder / "launcher.log"}')
            paths[world_size] = folder
        measured = compare_runs(paths[1], paths[2], comparison)
        measured['repetition'] = repetition
        records.append(measured)
        print(json.dumps(measured), flush=True)
    result = {'format_version': 2, 'device': device, 'training_precision': precision, 'comparison': comparison, 'epochs': epochs, 'global_batch_size': global_batch_size,
        'repetitions': repetitions, 'comparisons': records,
        'measurement_scope': 'one vs two ranks on the explicitly recorded device type',
        'gpu_result': device == 'cuda'}
    (output / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    write_manifest(output)
    verify_exports(output)
    return result


def verify_exports(output):
    return verify_all(Path(output), compare_runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--comparison', choices=['numerical', 'ranking'], default='numerical')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--global-batch-size', type=int, default=128)
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'ddp_cpu_validation')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(verify_exports(args.output), indent=2))
    elif args.device is None:
        parser.error('--device is required for training')
    else:
        launch(args.output, args.device, args.epochs, args.global_batch_size, args.repetitions, args.precision, args.comparison)


if __name__ == '__main__':
    main()
