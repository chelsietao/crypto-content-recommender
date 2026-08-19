import pandas as pd

from src.data_pipeline import build_dataset, validate_bundle


def test_generated_dataset_is_valid_and_reproducible():
    first = build_dataset(n_contents=60, n_users=12, impressions_per_user=20, save=False)
    second = build_dataset(n_contents=60, n_users=12, impressions_per_user=20, save=False)
    validate_bundle(first)
    assert len(first.contents) == 60
    assert len(first.users) == 12
    assert len(first.interactions) > 12 * 20
    pd.testing.assert_frame_equal(first.contents, second.contents)
    pd.testing.assert_frame_equal(first.users, second.users)
    pd.testing.assert_frame_equal(first.interactions, second.interactions)


def test_all_interactions_reference_existing_entities():
    bundle = build_dataset(n_contents=48, n_users=12, impressions_per_user=15, save=False)
    assert set(bundle.interactions["user_id"]).issubset(set(bundle.users["user_id"]))
    assert set(bundle.interactions["content_id"]).issubset(set(bundle.contents["content_id"]))
