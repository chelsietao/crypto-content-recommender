import numpy as np
import pytest

torch = pytest.importorskip('torch')
from src.config import ARTIFACTS_DIR
from src.gpu_inference import BatchedNeuralInference, MODEL_NAMES
from src.neural import NeuralSystem


@pytest.fixture(scope='module')
def system():
    torch.set_num_threads(1)
    return NeuralSystem.load(ARTIFACTS_DIR / 'neural' / 'neural_recommender.pt')


@pytest.mark.parametrize('model', MODEL_NAMES)
def test_cpu_batch_matches_checkpoint_for_every_user(system, model):
    engine = BatchedNeuralInference(system)
    users = [*system.users.user_id, 'new-user']
    actual = engine.decode(*engine.forward(engine.prepare(users), model))
    reference = system.recommenders()[model]
    for user, rows in zip(users, actual):
        expected = reference.recommend(user)
        assert [row[0] for row in rows] == expected.content_id.tolist()
        np.testing.assert_allclose([row[1] for row in rows], expected.score, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('batch_size', [1, 8, 32, 128])
def test_vectorized_batch_matches_per_user_execution(system, batch_size):
    engine = BatchedNeuralInference(system)
    users = ['U0001', 'U0042', 'new-user']
    batch = engine.prepare([users[i % len(users)] for i in range(batch_size)])
    for model in MODEL_NAMES:
        indices, scores = engine.forward(batch, model)
        rows = [engine.forward(batch.row(i), model) for i in range(batch_size)]
        torch.testing.assert_close(indices, torch.cat([row[0] for row in rows]))
        torch.testing.assert_close(scores, torch.cat([row[1] for row in rows]), atol=1e-6, rtol=1e-5)
        assert not scores.requires_grad


def test_seen_masks_cold_start_and_zero_k(system):
    engine = BatchedNeuralInference(system)
    for exclude in [True, False]:
        for model in MODEL_NAMES:
            actual = engine.decode(*engine.forward(engine.prepare(['U0001', 'new-user'], exclude), model))
            for user, rows in zip(['U0001', 'new-user'], actual):
                expected = system.recommenders()[model].recommend(user, exclude_seen=exclude)
                assert [row[0] for row in rows] == expected.content_id.tolist()
    batch = engine.prepare(['U0001'])
    batch.seen.fill_(True)
    for model in MODEL_NAMES:
        assert engine.decode(*engine.forward(batch, model)) == [[]]
        assert engine.decode(*engine.forward(batch, model, k=0)) == [[]]


def test_batch_engine_rejects_invalid_inputs(system):
    with pytest.raises(ValueError):
        BatchedNeuralInference(system, device='unknown')
    with pytest.raises(ValueError):
        BatchedNeuralInference(system, dtype=torch.int32)
    engine = BatchedNeuralInference(system)
    with pytest.raises(ValueError):
        engine.prepare([])
    batch = engine.prepare(['U0001'])
    for k in [-1, True, 1.5]:
        with pytest.raises(ValueError):
            engine.forward(batch, MODEL_NAMES[0], k=k)
    with pytest.raises(ValueError):
        engine.forward(batch, 'unknown')


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='An Apple MPS GPU is required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_mps_executes_and_fp32_preserves_recommendations(system, dtype):
    engine = BatchedNeuralInference(system, 'mps', dtype)
    users = [*system.users.user_id, 'new-user']
    for model in MODEL_NAMES:
        indices, scores = engine.forward(engine.prepare(users), model)
        assert indices.device.type == scores.device.type == 'mps'
        assert torch.isfinite(scores).all().item()
        if dtype == torch.float32:
            actual = engine.decode(indices, scores)
            for user, rows in zip(users, actual):
                expected = system.recommenders()[model].recommend(user)
                assert [row[0] for row in rows] == expected.content_id.tolist()
                np.testing.assert_allclose([row[1] for row in rows], expected.score, atol=1e-5, rtol=1e-5)
    torch.mps.synchronize()


@pytest.mark.parametrize('model', MODEL_NAMES)
def test_sort_optimization_preserves_baseline_and_id_ties(system, model):
    engine = BatchedNeuralInference(system)
    batch = engine.prepare(['U0001', 'new-user'])
    before = engine.forward(batch, model, optimized=False)
    after = engine.forward(batch, model)
    torch.testing.assert_close(before[0], after[0])
    torch.testing.assert_close(before[1], after[1], atol=0, rtol=0)
    candidates = torch.tensor([[5, 2, 8]])
    scores = torch.tensor([[0.5, 0.5, 0.5]])
    indices, _ = engine.finish(candidates, scores, 3)
    assert indices.tolist() == [[2, 5, 8]]


def test_precision_audit_counts_reference_candidates_by_id(system):
    from src.profile_gpu import validate_precision
    engine = BatchedNeuralInference(system)
    references = {}
    for model, recommender in system.recommenders().items():
        result = recommender.recommend('U0001')
        references[(model, 'U0001')] = (result.content_id.tolist(), dict(zip(result.content_id, result.score)))
        rows = validate_precision(engine, ['U0001'], model, references)
        assert rows[0]['exact_id_order']
        assert rows[0]['items_outside_cpu_candidates'] == 0


@pytest.mark.parametrize('candidate_count', [1, 5])
@pytest.mark.parametrize('model', MODEL_NAMES)
def test_small_candidate_limits_and_large_k_match_cpu(system, candidate_count, model):
    from dataclasses import replace
    from unittest.mock import patch
    with patch.object(system, 'config', replace(system.config, candidate_count=candidate_count)):
        engine = BatchedNeuralInference(system)
        batch = engine.prepare(['U0001'])
        for k in [10, 100, np.int64(10)]:
            actual = engine.decode(*engine.forward(batch, model, k=k))[0]
            expected = system.recommenders()[model].recommend('U0001', k=k)
            assert len(actual) == candidate_count
            assert [row[0] for row in actual] == expected.content_id.tolist()
            np.testing.assert_allclose([row[1] for row in actual], expected.score, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('mutation', ['wrong_shape', 'negative_index', 'outside_index', 'fractional_index', 'nan_score', 'positive_inf'])
def test_decode_rejects_malformed_result_tensors(system, mutation):
    engine = BatchedNeuralInference(system)
    indices, scores = engine.forward(engine.prepare(['U0001']), MODEL_NAMES[0])
    indices, scores = indices.clone(), scores.clone()
    if mutation == 'wrong_shape':
        scores = scores[:, :-1]
    elif mutation == 'negative_index':
        indices[0, 0] = -1
    elif mutation == 'outside_index':
        indices[0, 0] = len(engine.content_ids)
    elif mutation == 'fractional_index':
        indices = indices.float()
    elif mutation == 'nan_score':
        scores[0, 0] = float('nan')
    else:
        scores[0, 0] = float('inf')
    with pytest.raises(ValueError):
        engine.decode(indices, scores)


def audit_fixture(system):
    engine = BatchedNeuralInference(system)
    users = ['U0001', 'U0042']
    model = MODEL_NAMES[0]
    references = {}
    for user in users:
        result = system.recommenders()[model].recommend(user)
        references[(model, user)] = (result.content_id.tolist(), dict(zip(result.content_id, result.score)))
    return engine, users, model, references


@pytest.mark.parametrize('mutation', ['missing_user', 'missing_item', 'duplicate_item', 'wrong_precision', 'wrong_device'])
def test_precision_audit_rejects_incomplete_and_mislabeled_results(system, mutation):
    from types import SimpleNamespace
    from src.profile_gpu import audit_result
    engine, users, model, references = audit_fixture(system)
    indices, scores = engine.forward(engine.prepare(users), model)
    indices, scores = indices.clone(), scores.clone()
    if mutation == 'missing_user':
        indices, scores = indices[:1], scores[:1]
    elif mutation == 'missing_item':
        indices, scores = indices[:, :-1], scores[:, :-1]
    elif mutation == 'duplicate_item':
        indices[0, 1] = indices[0, 0]
    elif mutation == 'wrong_precision':
        scores = scores.double()
    else:
        engine = SimpleNamespace(device=torch.device('cuda'), dtype=engine.dtype, decode=engine.decode)
    with pytest.raises(ValueError):
        audit_result(engine, users, model, references, (indices, scores))


@pytest.mark.parametrize('mode', ['per_user_loop', 'batched_baseline', 'batched'])
def test_every_timed_mode_has_complete_fp32_audit(system, mode):
    from src.profile_gpu import validate_variant, require_fp32_equivalence
    engine, users, model, references = audit_fixture(system)
    rows = validate_variant(engine, engine.prepare(users), users, model, mode, references)
    assert len(rows) == len(users)
    assert [row['request_index'] for row in rows] == [0, 1]
    assert all(row['mode'] == mode and row['batch_size'] == 2 for row in rows)
    assert all(row['common_score_items'] == 10 for row in rows)
    require_fp32_equivalence(rows)
