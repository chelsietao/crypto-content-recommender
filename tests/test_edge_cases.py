import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import DatasetBundle, build_dataset, validate_bundle
from src.evaluation import chronological_holdout, evaluate_model
from src.recommenders import RankingRecommender, build_all_models, minmax


@pytest.fixture(scope='module')
def system():
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    return bundle, split, build_all_models(split.contents, bundle.users, split.train)


@pytest.mark.parametrize('k', [-1, 1.5, True])
def test_all_recommenders_reject_invalid_k(system, k):
    _, _, models = system
    for model in models.values():
        with pytest.raises(ValueError, match='non-negative integer'):
            model.recommend('U0001', k=k)


def test_zero_k_returns_empty_schema_for_every_model(system):
    _, _, models = system
    for model in models.values():
        result = model.recommend('U0001', k=0)
        assert result.empty
        assert {'content_id', 'score', 'model'} <= set(result.columns)


def test_seen_option_is_honored_by_candidate_and_scoring_interfaces(system):
    _, _, models = system
    for model in models.values():
        seen = model.seen_by_user['U0001']
        assert set(model.candidate_ids('U0001', exclude_seen=False)) & seen
        result = model.recommend('U0001', k=1000, exclude_seen=False)
        assert set(result.content_id) & seen
        assert not (set(model.recommend('U0001', k=1000).content_id) & seen)


def test_unknown_user_has_finite_unique_recommendations(system):
    _, _, models = system
    for model in models.values():
        result = model.recommend('new-user', k=10)
        assert not result.empty
        assert result.content_id.is_unique
        assert np.isfinite(result.score).all()


def test_empty_history_uses_a_ranker_fallback(system):
    bundle, split, _ = system
    models = build_all_models(split.contents, bundle.users, split.train.iloc[:0])
    ranker = models['v5_logistic_ranker']
    assert ranker.training_mode == 'hybrid_fallback'
    assert ranker.feature_importance().empty
    for model in models.values():
        result = model.recommend('U0001')
        assert not result.empty
        assert np.isfinite(result.score).all()


def test_single_class_label_window_uses_constant_label_fallback(system):
    bundle, split, models = system
    events = split.train.copy()
    late = events.timestamp > models['v5_logistic_ranker'].feature_cutoff
    events.loc[late, 'event_type'] = 'impression'
    model = RankingRecommender().fit(split.contents, bundle.users, events)
    assert model.training_mode == 'constant_label'
    assert model.constant_probability == 0
    assert np.isfinite(model.recommend('U0001').score).all()


def test_empty_catalog_is_rejected_before_model_fitting(system):
    bundle, split, _ = system
    with pytest.raises(ValueError, match='content item'):
        RankingRecommender().fit(split.contents.iloc[:0], bundle.users, split.train.iloc[:0])


def test_minmax_does_not_propagate_nan_or_infinity():
    result = minmax(np.array([np.nan, -np.inf, np.inf, 1.0, 2.0]))
    np.testing.assert_array_equal(result, [0, 0, 0, 0, 1])


def test_event_before_publication_is_rejected(system):
    bundle, _, _ = system
    events = bundle.interactions.copy()
    events.loc[0, 'timestamp'] = bundle.contents.published_at.min() - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match='precedes content'):
        validate_bundle(DatasetBundle(bundle.contents, bundle.users, events))


def test_invalid_quality_and_missing_ids_are_rejected(system):
    bundle, _, _ = system
    contents = bundle.contents.copy()
    contents.loc[0, 'quality_score'] = np.inf
    with pytest.raises(ValueError, match='quality_score'):
        validate_bundle(DatasetBundle(contents, bundle.users, bundle.interactions))
    contents = bundle.contents.copy()
    contents.loc[0, 'content_id'] = None
    with pytest.raises(ValueError, match='IDs'):
        validate_bundle(DatasetBundle(contents, bundle.users, bundle.interactions))


def test_evaluation_rejects_zero_k_and_empty_targets(system):
    _, split, models = system
    with pytest.raises(ValueError, match='positive'):
        evaluate_model(models['v1_popularity'], split.test, k=0)
    with pytest.raises(ValueError, match='test target'):
        evaluate_model(models['v1_popularity'], split.test.iloc[:0])
