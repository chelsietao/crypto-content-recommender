import pandas as pd

from src.data_pipeline import build_dataset
from src.evaluation import chronological_holdout


def test_global_cutoff_separates_all_training_and_test_events():
    bundle = build_dataset(n_contents=72, n_users=18, impressions_per_user=28, save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    assert not split.test.empty
    assert split.test['user_id'].is_unique
    assert split.train['timestamp'].max() <= split.cutoff
    assert (split.test['test_timestamp'] > split.cutoff).all()
    assert (split.contents['published_at'] <= split.cutoff).all()
    train_pairs = set(zip(split.train.user_id, split.train.content_id))
    test_pairs = set(zip(split.test.user_id, split.test.content_id))
    assert train_pairs.isdisjoint(test_pairs)
    assert set(split.test.content_id) <= set(split.contents.content_id)


def test_unpublished_and_already_seen_items_cannot_be_targets():
    times = pd.date_range('2026-01-01', periods=7, tz='UTC')
    events = pd.DataFrame([
        ('e1', 'u', 'seen', 'impression', times[0]),
        ('e2', 'v', 'old', 'impression', times[1]),
        ('e3', 'u', 'seen', 'click', times[2]),
        ('e4', 'u', 'future', 'click', times[3]),
        ('e5', 'u', 'hidden', 'click', times[4]),
        ('e6', 'u', 'hidden', 'hide', times[5]),
        ('e7', 'u', 'old', 'click', times[6]),
    ], columns=['event_id', 'user_id', 'content_id', 'event_type', 'timestamp'])
    catalog = pd.DataFrame({'content_id': ['seen', 'old', 'hidden', 'future'],
                            'published_at': [times[0], times[0], times[0], times[3]]})
    split = chronological_holdout(events, catalog, train_fraction=0.3)
    assert split.test.content_id.tolist() == ['old']
    assert split.test.test_timestamp.tolist() == [times[6]]
