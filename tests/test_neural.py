from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip('torch')

from src.data_pipeline import build_dataset, load_bundle
from src.evaluation import chronological_holdout, evaluate_all
from src.neural import FeatureEncoder, NeuralConfig, NeuralSystem, prepare_examples


@pytest.fixture(scope='module')
def trained():
    bundle = load_bundle()
    split = chronological_holdout(bundle.interactions, bundle.contents)
    system = NeuralSystem(split.contents, bundle.users, split.train, NeuralConfig(epochs=6)).fit()
    return bundle, split, system


def test_neural_stages_train_parameters_and_select_by_validation(trained):
    _, _, system = trained
    for stage in ['two_tower', 'neural_ranker']:
        summary = system.training_summary[stage]
        history = pd.DataFrame(system.history)
        rows = history.loc[history.stage == stage]
        assert summary['parameters_changed']
        assert summary['selected_epoch'] == rows.loc[rows.validation_loss.idxmin(), 'epoch']
        assert np.isfinite(rows[['train_loss', 'validation_loss']]).all().all()
    assert all(not parameter.requires_grad for parameter in system.tower.parameters())


def test_example_features_and_labels_respect_all_temporal_boundaries(trained):
    _, split, system = trained
    examples = system.examples
    for window, boundary in enumerate(examples.boundaries):
        indices = examples.indices(window)
        assert len(indices)
        assert (examples.timestamps[indices] < examples.label_ends[indices]).all()
        assert (examples.label_ends[indices] <= boundary.value).all()
        if window:
            assert (examples.timestamps[indices] > examples.boundaries[window - 1].value).all()
    selected = system.ranker_example_indices
    assert set(examples.windows[selected]) == {2, 3}
    assert examples.boundaries[-1] <= split.cutoff
    for index in selected:
        assert examples.item_indices[index] in examples.eligible[index]
        assert (system.encoder.published_ns[examples.eligible[index]] <= examples.timestamps[index]).all()


def test_current_impression_outcome_cannot_change_its_input_features(trained):
    _, split, system = trained
    examples = system.examples
    index = int(np.flatnonzero(examples.labels > 0)[0])
    user = examples.user_ids[index]
    item = system.encoder.content_ids[examples.item_indices[index]]
    changed = split.train.copy()
    timestamp = pd.Timestamp(examples.timestamps[index], tz='UTC')
    current = ((changed.user_id == user) & (changed.content_id == item)
               & (changed.timestamp > timestamp)
               & (changed.timestamp <= timestamp + pd.Timedelta(minutes=15)))
    changed.loc[current, 'event_type'] = 'hide'
    updated, _ = prepare_examples(system.encoder, changed, system.config)
    target = [n for n, (u, i, t) in enumerate(zip(updated.user_ids, updated.item_indices, updated.timestamps))
              if u == user and i == examples.item_indices[index] and t == examples.timestamps[index]][0]
    assert updated.labels[target] == 0
    np.testing.assert_array_equal(updated.user_features[target], examples.user_features[index])


def test_future_catalog_rows_do_not_change_existing_item_features(trained):
    bundle, split, system = trained
    extended = FeatureEncoder(bundle.contents, bundle.users, system.config.text_dim)
    for index, content_id in enumerate(system.encoder.content_ids):
        np.testing.assert_array_equal(system.encoder.item_features[index], extended.item_features[extended.content_index[content_id]])


def test_neural_candidates_and_ranking_obey_common_contract(trained):
    _, split, system = trained
    models = system.recommenders()
    for user in ['U0001', 'new-user']:
        expected = set(models['v7_pytorch_two_tower'].candidate_ids(user))
        assert len(expected) <= system.config.candidate_count
        for model in models.values():
            result = model.recommend(user, k=10)
            assert result.content_id.is_unique
            assert np.isfinite(result.score).all()
            assert set(result.content_id) <= expected
            assert not (set(result.content_id) & model.seen_by_user.get(user, set()))
            assert model.recommend(user, k=0).empty
            with pytest.raises(ValueError):
                model.recommend(user, k=-1)
    metrics, _ = evaluate_all(models, split.test)
    assert (metrics['hit_rate@10'] <= metrics.candidate_recall).all()
    assert metrics.candidate_recall.between(0, 1).all()


def test_checkpoint_restores_scores_and_candidate_ids(trained, tmp_path):
    _, _, system = trained
    checkpoint = tmp_path / 'models.pt'
    system.save(checkpoint)
    restored = NeuralSystem.load(checkpoint)
    for user in ['U0001', 'U0042', 'new-user']:
        np.testing.assert_array_equal(system.candidate_indices(user), restored.candidate_indices(user))
        np.testing.assert_allclose(system.retrieval_scores(user), restored.retrieval_scores(user), atol=1e-7, rtol=0)
        np.testing.assert_allclose(system.rank_scores(user), restored.rank_scores(user), atol=1e-7, rtol=0)


def test_fixed_seed_repeats_cpu_training(trained):
    bundle, split, first = trained
    second = NeuralSystem(split.contents, bundle.users, split.train, first.config).fit()
    pd.testing.assert_frame_equal(pd.DataFrame(first.history), pd.DataFrame(second.history))
    for name, value in first.tower.state_dict().items():
        torch.testing.assert_close(value, second.tower.state_dict()[name], rtol=0, atol=0)
    for name, value in first.ranker.state_dict().items():
        torch.testing.assert_close(value, second.ranker.state_dict()[name], rtol=0, atol=0)


def test_neural_configuration_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        NeuralConfig(candidate_count=0)
    with pytest.raises(ValueError):
        NeuralConfig(temperature=float('nan'))


def test_neural_training_rejects_future_catalog_contents(trained):
    bundle, split, _ = trained
    with pytest.raises(ValueError, match='published after'):
        NeuralSystem(bundle.contents, bundle.users, split.train, NeuralConfig())
