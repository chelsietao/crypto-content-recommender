import json
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.config import PROJECT_ROOT
from src.distributed_training import global_batches, rank_batch, weighted_rank_loss


@pytest.mark.parametrize('count', [1, 3, 13])
@pytest.mark.parametrize('world_size', [1, 2, 4])
def test_rank_weighting_matches_global_gradient_with_uneven_or_empty_shards(count, world_size):
    torch.manual_seed(7)
    inputs = torch.randn(count, 3)
    labels = (torch.arange(count) % 2).float()
    reference = torch.nn.Linear(3, 1)
    candidate = torch.nn.Linear(3, 1)
    candidate.load_state_dict(reference.state_dict())
    positive_weight = torch.tensor(2.5)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        reference(inputs).squeeze(-1), labels, pos_weight=positive_weight)
    loss.backward()
    for rank in range(world_size):
        selected, real_count = rank_batch(torch.arange(count), rank, world_size)
        local = candidate(inputs[selected]).squeeze(-1)
        (weighted_rank_loss(local, labels[selected], positive_weight,
                            real_count, count, world_size) / world_size).backward()
    for expected, actual in zip(reference.parameters(), candidate.parameters()):
        torch.testing.assert_close(expected.grad, actual.grad, atol=1e-7, rtol=1e-6)


def test_global_shuffle_matches_existing_generator_and_has_no_padding():
    indices = np.arange(2345)
    generator = torch.Generator().manual_seed(42)
    for epoch in [1, 2, 3]:
        expected = torch.as_tensor(indices)[torch.randperm(len(indices), generator=generator)]
        actual = global_batches(indices, 128, 42, epoch)
        torch.testing.assert_close(torch.cat(actual), expected)
        for batch in actual:
            shards = [rank_batch(batch, rank, 2) for rank in range(2)]
            real = torch.cat([values[:count] for values, count in shards])
            assert sorted(real.tolist()) == sorted(batch.tolist())
    assert len(actual[-1]) == 41


@pytest.mark.skipif(os.environ.get('RUN_DDP_INTEGRATION') != '1', reason='Requires local process-group sockets')
def test_two_process_training_matches_single_and_handles_one_sample_tail(tmp_path):
    output = tmp_path / 'ddp'
    command = [sys.executable, '-m', 'src.validate_distributed', '--device', 'cpu', '--epochs', '2',
        '--precision', 'float64', '--global-batch-size', '586', '--output', str(output)]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=150)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads((output / 'comparison.json').read_text())
    assert not data['gpu_result']
    assert data['comparison'] == 'numerical'
    from src.validate_distributed import verify_exports
    assert verify_exports(output)['device'] == 'cpu'
    comparison = data['comparisons'][0]
    assert comparison['matched_recommendation_groups'] == 162
    assert max(comparison['maximum_parameter_error'].values()) < 1e-4
    import pandas as pd
    steps = pd.read_csv(output / 'round_0_world_2' / 'steps_rank_1.csv')
    assert (steps.local_examples == 0).any()
    metadata = json.loads((output / 'round_0_world_2' / 'results.json').read_text())
    assert all(metadata['training'][stage]['replica_max_parameter_error'] == 0
               for stage in ['two_tower', 'neural_ranker'])
