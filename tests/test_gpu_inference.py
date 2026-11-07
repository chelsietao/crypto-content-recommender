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
    for k in [-1, True, 51]:
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
