import numpy as np

from src.data_pipeline import build_dataset
from src.evaluation import evaluate_all, chronological_holdout
from src.recommenders import build_all_models


def _small_system():
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    models = build_all_models(split.contents, bundle.users, split.train)
    return bundle, split, models


def test_all_six_models_return_unique_unseen_items():
    bundle, split, models = _small_system()
    assert len(models) == 6
    user_id = bundle.users.iloc[0]["user_id"]
    seen = set(split.train.loc[split.train["user_id"] == user_id, "content_id"])
    for model in models.values():
        recommendations = model.recommend(user_id, k=10)
        assert len(recommendations) == 10
        assert recommendations["content_id"].is_unique
        assert not (set(recommendations["content_id"]) & seen)
        assert np.isfinite(recommendations["score"]).all()


def test_offline_metrics_are_bounded():
    _, split, models = _small_system()
    metrics, details = evaluate_all(models, split.test, k=10)
    assert len(metrics) == 6
    assert not details.empty
    for column in ("hit_rate@10", "recall@10", "ndcg@10", "catalog_coverage", "intra_list_diversity"):
        assert metrics[column].between(0, 1).all()


def test_hide_overrides_impression_and_positive_actions():
    import pandas as pd
    from src.recommenders import aggregate_interactions
    events = pd.DataFrame({
        'event_id': ['e1', 'e2', 'e3'], 'user_id': ['u'] * 3,
        'content_id': ['c'] * 3, 'event_type': ['impression', 'click', 'hide'],
        'timestamp': pd.date_range('2026-01-01', periods=3, tz='UTC'),
    })
    assert aggregate_interactions(events).preference.iloc[0] == -2
    assert aggregate_interactions(events.iloc[[0, 2]]).preference.iloc[0] == -2


def test_collaborative_storage_is_sparse_and_neighbor_bounded():
    from scipy import sparse
    from src.recommenders import CollaborativeFilteringRecommender
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    model = CollaborativeFilteringRecommender(neighbors=5, block_size=11).fit(
        bundle.contents, bundle.users, bundle.interactions)
    assert sparse.issparse(model.user_item)
    assert sparse.issparse(model.item_similarity)
    assert np.diff(model.item_similarity.indptr).max() <= 5
    assert np.all(model.item_similarity.diagonal() == 0)
    assert np.isfinite(model.score_all(bundle.users.user_id.iloc[0])).all()


def test_ranker_scores_only_recalled_candidates_and_separates_time_windows():
    _, split, models = _small_system()
    ranker = models['v5_logistic_ranker']
    assert ranker.feature_cutoff < ranker.label_window_start <= split.cutoff
    user_id = split.test.user_id.iloc[0]
    candidates = set(ranker.candidate_ids(user_id))
    assert len(candidates) <= 4 * 20
    scores = ranker.score_all(user_id)
    finite_ids = {ranker.content_ids[index] for index in np.flatnonzero(np.isfinite(scores))}
    assert finite_ids == candidates
    assert set(ranker.recommend(user_id, k=10).content_id) <= candidates


def test_diversity_caps_hold_even_when_fewer_than_k_items_are_feasible():
    _, _, models = _small_system()
    model = models['v6_diversity_reranking']
    model.contents.loc[:, 'source'] = 'one-source'
    model.contents.loc[:, 'topic_tags'] = 'one-topic'
    result = model.recommend(model.users.user_id.iloc[0], k=10)
    assert len(result) == 2
    assert result.source.value_counts().max() <= model.max_per_source


def test_failure_analysis_retains_misses_from_every_model():
    from src.evaluation import build_failure_analysis
    bundle, split, models = _small_system()
    _, details = evaluate_all(models, split.test, k=1)
    failures = build_failure_analysis(details, bundle.users, split.contents)
    expected = details.loc[details.hit_at_k == 0].groupby('model').size().to_dict()
    assert failures.groupby('model').size().to_dict() == expected


def test_later_label_changes_leave_training_features_unchanged(monkeypatch):
    from src.recommenders import RankingRecommender
    from sklearn.pipeline import Pipeline
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    captured = []
    original_fit = Pipeline.fit

    def capture_fit(self, features, labels, **kwargs):
        captured.append((features.copy(), labels.copy()))
        return original_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(Pipeline, 'fit', capture_fit)
    first = RankingRecommender().fit(split.contents, bundle.users, split.train)
    changed = split.train.copy()
    # Change only label-window action weights, preserving timestamps and binary labels.
    late = changed.timestamp > first.feature_cutoff
    changed.loc[late & changed.event_type.isin(['like', 'save']), 'event_type'] = 'click'
    RankingRecommender().fit(split.contents, bundle.users, changed)
    np.testing.assert_allclose(captured[0][0], captured[1][0])
    np.testing.assert_array_equal(captured[0][1], captured[1][1])
