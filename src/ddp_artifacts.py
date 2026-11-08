"""Complete DDP artifact inventory and per-rank training evidence validation."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ('two_tower', 'neural_ranker')
RUN_FILES = {'communication.csv', 'evaluation_details.csv', 'model_metrics.csv',
             'neural_recommender.pt', 'recommendations.csv', 'results.json', 'training_history.csv'}


def run_files(world):
    return RUN_FILES | {f'{name}_{rank}.{extension}' for rank in range(world)
                        for name, extension in [('rank', 'json'), ('steps_rank', 'csv')]}


def required_files(recorded):
    count = recorded.get('repetitions')
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('Invalid distributed repetition count')
    return {'comparison.json'} | {
        f'round_{rep}_world_{world}/{name}' for rep in range(count) for world in (1, 2)
        for name in run_files(world) | {'manifest.json'}}


def verify_hashes(folder, manifest, expected):
    if not isinstance(manifest, dict) or set(manifest) != expected:
        raise ValueError('Distributed manifest must cover the complete artifact inventory')
    root = folder.resolve()
    for name, digest in manifest.items():
        relative = Path(name)
        path = folder / relative
        if (relative.is_absolute() or '..' in relative.parts or not isinstance(digest, str)
                or root not in path.resolve().parents):
            raise ValueError('Invalid distributed artifact path or hash')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Distributed artifact hash mismatch: {name}')


def write_manifest(output):
    recorded = json.loads((output / 'comparison.json').read_text())
    expected = required_files(recorded)
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in sorted(expected)}
    (output / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def verify_steps(frames, summary):
    world = _integer(summary['world_size'], 'world size')
    epochs = _integer(summary['epochs'], 'epochs')
    batch_size = _integer(summary['global_batch_size'], 'global batch size')
    if len(frames) != world:
        raise ValueError('Incomplete rank step inventory')
    required_columns = {'rank', 'stage', 'epoch', 'batch_index', 'global_examples', 'local_examples',
                        'forward_seconds', 'backward_with_ddp_seconds', 'optimizer_seconds'}
    training = summary['training']
    grids = {}
    for stage in STAGES:
        total = _integer(training[stage]['training_examples'], 'training examples')
        _integer(training[stage]['validation_examples'], 'validation examples')
        if training[stage]['world_size'] != world or training[stage]['global_batch_size'] != batch_size:
            raise ValueError('Stage settings differ from run settings')
        if training[stage]['replica_max_parameter_error'] != 0:
            raise ValueError('DDP replicas differ')
        grids.update({(stage, epoch, index): min(batch_size, total - index * batch_size)
                      for epoch in range(1, epochs + 1) for index in range(math.ceil(total / batch_size))})
    for rank, frame in enumerate(frames):
        if not required_columns <= set(frame):
            raise ValueError('Step log is missing required columns')
        if set(zip(frame.stage, frame.epoch, frame.batch_index)) != set(grids):
            raise ValueError('Step log does not cover every stage, epoch and global batch')
        if frame.duplicated(['stage', 'epoch', 'batch_index']).any() or not frame['rank'].eq(rank).all():
            raise ValueError('Step log has duplicated coordinates or the wrong rank')
        counts = frame[['epoch', 'batch_index', 'global_examples', 'local_examples', 'rank']].to_numpy(dtype=float)
        if not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all():
            raise ValueError('Step coordinates and sample counts must be finite integers')
        durations = frame[['forward_seconds', 'backward_with_ddp_seconds', 'optimizer_seconds']].to_numpy(dtype=float)
        if not np.isfinite(durations).all() or (durations < 0).any():
            raise ValueError('Step durations must be finite and nonnegative')
        for row in frame.itertuples():
            expected_global = grids[(row.stage, row.epoch, row.batch_index)]
            expected_local = max(0, (expected_global + world - 1 - rank) // world)
            if row.global_examples != expected_global or row.local_examples != expected_local:
                raise ValueError('Step sample counts differ from the exact rank partition')
    return len(grids)


def verify_history(folder, summary):
    history = pd.read_csv(folder / 'training_history.csv')
    epochs = summary['epochs']
    if (history.duplicated(['stage', 'epoch']).any()
            or set(zip(history.stage, history.epoch)) != {(stage, epoch) for stage in STAGES for epoch in range(epochs + 1)}):
        raise ValueError('Training history epoch coverage is incomplete')
    values = history[['train_loss', 'validation_loss']].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Training history has invalid loss values')
    for stage in STAGES:
        rows = history.loc[history.stage == stage].sort_values('epoch')
        selected = rows.loc[rows.validation_loss.idxmin()]
        declared = summary['training'][stage]
        if (declared['selected_epoch'] != selected.epoch
                or not np.isclose(declared['validation_loss'], selected.validation_loss, atol=1e-12, rtol=0)):
            raise ValueError('Selected checkpoint differs from minimum validation loss')


def verify_communication(folder, summary):
    frame = pd.read_csv(folder / 'communication.csv')
    if (frame.duplicated(['stage', 'iteration']).any()
            or set(zip(frame.stage, frame.iteration)) != {(stage, index) for stage in STAGES for index in range(10)}):
        raise ValueError('Communication sample coverage is incomplete')
    if not frame.world_size.eq(summary['world_size']).all():
        raise ValueError('Communication world size mismatch')
    durations = frame.all_reduce_seconds_max_rank.to_numpy(dtype=float)
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError('Communication times are invalid')
    for row in frame.itertuples():
        if row.payload_bytes_per_rank != summary['training'][row.stage]['parameter_bytes']:
            raise ValueError('Communication payload differs from model parameter bytes')


def verify_all(output, compare):
    output = output.resolve()
    recorded = json.loads((output / 'comparison.json').read_text())
    _integer(recorded['epochs'], 'epochs')
    _integer(recorded['global_batch_size'], 'global batch size')
    if recorded['training_precision'] not in ('float32', 'float64'):
        raise ValueError('Invalid training precision')
    expected_files = required_files(recorded)
    manifest = json.loads((output / 'manifest.json').read_text())
    verify_hashes(output, manifest, expected_files)
    actual = {str(path.relative_to(output)) for path in output.rglob('*')
              if path.is_file() and path.name not in ('launcher.log', '.DS_Store') and path != output / 'manifest.json'}
    if actual != expected_files:
        raise ValueError('Distributed output contains missing or unexpected artifacts')
    if recorded['device'] not in ('cpu', 'cuda') or recorded['gpu_result'] != (recorded['device'] == 'cuda'):
        raise ValueError('Distributed report device label is inconsistent')
    if len(recorded['comparisons']) != recorded['repetitions']:
        raise ValueError('Distributed comparison round coverage is incomplete')
    expected_backend = 'gloo' if recorded['device'] == 'cpu' else 'nccl'
    total_steps = 0
    for rep, expected in enumerate(recorded['comparisons']):
        folders = [output / f'round_{rep}_world_{world}' for world in (1, 2)]
        for folder, world in zip(folders, (1, 2)):
            verify_hashes(folder, json.loads((folder / 'manifest.json').read_text()), run_files(world))
            summary = json.loads((folder / 'results.json').read_text())
            if any(summary[key] != recorded[key] for key in ('device', 'epochs', 'global_batch_size', 'training_precision')):
                raise ValueError('Run settings differ from the comparison settings')
            if summary['world_size'] != world or summary['backend'] != expected_backend:
                raise ValueError('Distributed world size or backend mismatch')
            if summary['checkpoint_sha256'] != hashlib.sha256((folder / 'neural_recommender.pt').read_bytes()).hexdigest():
                raise ValueError('Checkpoint differs from the training summary')
            if not np.isfinite(summary['fit_seconds_max_rank']) or summary['fit_seconds_max_rank'] <= 0:
                raise ValueError('Invalid model-fit time')
            frames = []
            for rank in range(world):
                info = json.loads((folder / f'rank_{rank}.json').read_text())
                expected_device = 'cpu' if recorded['device'] == 'cpu' else f'cuda:{rank}'
                if (info['rank'] != rank or info['local_rank'] != rank or info['world_size'] != world
                        or info['backend'] != expected_backend or info['device'] != expected_device):
                    raise ValueError('Rank identity or device binding is inconsistent')
                frames.append(pd.read_csv(folder / f'steps_rank_{rank}.csv'))
            total_steps += verify_steps(frames, summary)
            verify_history(folder, summary)
            verify_communication(folder, summary)
            verify_predictions(folder, summary)
        measured = compare(*folders, comparison=recorded['comparison'])
        measured['repetition'] = rep
        if measured != expected:
            raise ValueError('Distributed comparison differs from checkpoint outputs')
    return {'device': recorded['device'], 'gpu_result': recorded['gpu_result'],
            'repetitions': recorded['repetitions'], 'files': len(manifest), 'verified_global_steps': total_steps}


def verify_predictions(folder, summary):
    from src.neural import NeuralSystem
    from src.data_pipeline import build_dataset
    from src.evaluation import chronological_holdout, evaluate_all
    system = NeuralSystem.load(folder / 'neural_recommender.pt')
    if (system.config.epochs != summary['epochs'] or system.config.batch_size != summary['global_batch_size']
            or system.training_summary != summary['training']):
        raise ValueError('Training summary differs from the checkpoint payload')
    try:
        pd.testing.assert_frame_equal(
            pd.read_csv(folder / 'training_history.csv').sort_values(['stage', 'epoch']).reset_index(drop=True),
            pd.DataFrame(system.history).sort_values(['stage', 'epoch']).reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    except AssertionError as exc:
        raise ValueError('Training history differs from the checkpoint payload') from exc
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    eligible_users = bundle.users.loc[bundle.users.created_at <= split.train.timestamp.max()]
    try:
        for actual, expected in ((system.contents, split.contents), (system.users, eligible_users),
                                 (system.interactions, split.train)):
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                          check_dtype=False, check_exact=False)
    except AssertionError as exc:
        raise ValueError('Checkpoint data differs from the reproducible temporal split') from exc
    models = system.recommenders()
    saved = pd.read_csv(folder / 'recommendations.csv')
    users = [*system.users.user_id, 'new-user']
    expected_groups = {(model, user) for model in models for user in users}
    if set(zip(saved.model, saved.user_id)) != expected_groups:
        raise ValueError('Recommendation export has incomplete model/user coverage')
    if saved.duplicated(['model', 'user_id', 'rank']).any() or saved.duplicated(['model', 'user_id', 'content_id']).any():
        raise ValueError('Recommendation export has duplicate ranks or contents')
    if not np.isfinite(saved.score).all():
        raise ValueError('Recommendation export contains nonfinite scores')
    for (name, user), rows in saved.groupby(['model', 'user_id']):
        actual = models[name].recommend(user)
        rows = rows.sort_values('rank')
        if (rows['rank'].tolist() != list(range(1, len(actual) + 1))
                or rows.content_id.tolist() != actual.content_id.tolist()
                or not np.allclose(rows.score, actual.score, atol=1e-6, rtol=1e-5)):
            raise ValueError('Recommendation export differs from checkpoint inference')
    details = pd.read_csv(folder / 'evaluation_details.csv')
    targets = details[['user_id', 'target_content_id']].drop_duplicates()
    if not targets.user_id.is_unique or set(zip(details.model, details.user_id)) != {
            (name, user) for name in models for user in targets.user_id}:
        raise ValueError('Evaluation targets or model coverage are inconsistent')
    metrics, computed_details = evaluate_all(models, split.test)
    try:
        pd.testing.assert_frame_equal(metrics.sort_values('model').reset_index(drop=True),
            pd.read_csv(folder / 'model_metrics.csv').sort_values('model').reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
        pd.testing.assert_frame_equal(computed_details.sort_values(['model', 'user_id']).reset_index(drop=True),
            details.sort_values(['model', 'user_id']).reset_index(drop=True),
            check_dtype=False, check_exact=False, atol=1e-10, rtol=1e-10)
    except AssertionError as exc:
        raise ValueError('Evaluation statistics differ from checkpoint inference') from exc
    for attr, stage in [('tower', 'two_tower'), ('ranker', 'neural_ranker')]:
        bytes_per_value = {'float32': 4, 'float64': 8}[summary['training_precision']]
        expected_bytes = sum(value.numel() for value in getattr(system, attr).state_dict().values()) * bytes_per_value
        if summary['training'][stage]['parameter_bytes'] != expected_bytes:
            raise ValueError('Communication payload does not match checkpoint parameter count')
