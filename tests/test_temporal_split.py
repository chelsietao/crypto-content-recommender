from src.data_pipeline import build_dataset
from src.evaluation import leave_latest_positive_out


def test_latest_positive_pair_is_removed_from_training():
    bundle = build_dataset(n_contents=60, n_users=12, impressions_per_user=20, save=False)
    split = leave_latest_positive_out(bundle.interactions)
    assert split.test["user_id"].is_unique
    train_pairs = set(zip(split.train["user_id"], split.train["content_id"]))
    test_pairs = set(zip(split.test["user_id"], split.test["content_id"]))
    assert train_pairs.isdisjoint(test_pairs)
