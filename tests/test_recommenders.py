import numpy as np

from src.data_pipeline import build_dataset
from src.evaluation import evaluate_all, leave_latest_positive_out
from src.recommenders import build_all_models


def _small_system():
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    split = leave_latest_positive_out(bundle.interactions)
    models = build_all_models(bundle.contents, bundle.users, split.train)
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
