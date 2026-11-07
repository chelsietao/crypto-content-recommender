"""Temporal offline evaluation for the recommendation strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EVENT_WEIGHTS, TOP_K
from src.recommenders import BaseRecommender, aggregate_interactions, split_tags, validate_k


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    contents: pd.DataFrame
    cutoff: pd.Timestamp


def chronological_holdout(
    interactions: pd.DataFrame,
    contents: pd.DataFrame,
    train_fraction: float = 0.8,
) -> TemporalSplit:
    """Freeze training and catalog at one global cutoff, evaluate later unseen positives."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    times = interactions["timestamp"].sort_values().drop_duplicates().reset_index(drop=True)
    if len(times) < 2:
        raise ValueError("At least two distinct event timestamps are required")
    index = min(max(int(len(times) * train_fraction) - 1, 0), len(times) - 2)
    cutoff = times.iloc[index]
    catalog = contents.loc[contents["published_at"] <= cutoff].reset_index(drop=True).copy()
    train = interactions.loc[interactions["timestamp"] <= cutoff].copy()
    train = train.loc[train["content_id"].isin(catalog["content_id"])].reset_index(drop=True)
    later = interactions.loc[
        (interactions["timestamp"] > cutoff)
        & interactions["content_id"].isin(catalog["content_id"])
    ].copy()
    # The evaluation target must not have appeared in the frozen training history.
    seen = pd.MultiIndex.from_frame(train[["user_id", "content_id"]])
    later = later.loc[~pd.MultiIndex.from_frame(later[["user_id", "content_id"]]).isin(seen)]
    labels = aggregate_interactions(later)
    positive_pairs = labels.loc[labels["preference"] > 0, ["user_id", "content_id"]]
    positives = later.loc[later["event_type"].map(EVENT_WEIGHTS) > 0].merge(
        positive_pairs, on=["user_id", "content_id"], how="inner"
    )
    test = (
        positives.sort_values(["timestamp", "event_id"], kind="stable")
        .groupby("user_id", as_index=False).head(1)
        [["user_id", "content_id", "timestamp"]]
        .rename(columns={"timestamp": "test_timestamp"}).reset_index(drop=True)
    )
    return TemporalSplit(train=train, test=test, contents=catalog, cutoff=cutoff)


def intra_list_diversity(recommendations: pd.DataFrame) -> float:
    tag_sets = [split_tags(value) for value in recommendations["topic_tags"]]
    if len(tag_sets) < 2:
        return 0.0
    distances: list[float] = []
    for left in range(len(tag_sets)):
        for right in range(left + 1, len(tag_sets)):
            union = tag_sets[left] | tag_sets[right]
            similarity = len(tag_sets[left] & tag_sets[right]) / max(len(union), 1)
            distances.append(1.0 - similarity)
    return float(np.mean(distances))


def evaluate_model(
    model: BaseRecommender,
    test: pd.DataFrame,
    k: int = TOP_K,
) -> tuple[dict[str, float], pd.DataFrame]:
    validate_k(k)
    if k == 0:
        raise ValueError("Evaluation k must be positive")
    if test.empty:
        raise ValueError("Evaluation requires at least one eligible test target")
    if not test["user_id"].is_unique:
        raise ValueError("Evaluation requires one target per user")
    if not test["content_id"].isin(model.content_ids).all():
        raise ValueError("Evaluation target is absent from the model catalog")
    if any(row.content_id in model.seen_by_user.get(row.user_id, set()) for row in test.itertuples()):
        raise ValueError("Evaluation targets must be unseen in training")
    rows: list[dict[str, object]] = []
    all_recommended: set[str] = set()
    diversities: list[float] = []
    topic_coverages: list[int] = []
    source_coverages: list[int] = []

    for record in test.itertuples():
        recommendations = model.recommend(record.user_id, k=k)
        ids = recommendations["content_id"].tolist()
        all_recommended.update(ids)
        diversities.append(intra_list_diversity(recommendations))
        topic_coverages.append(
            len(set().union(*(split_tags(value) for value in recommendations["topic_tags"])))
            if not recommendations.empty
            else 0
        )
        source_coverages.append(recommendations["source"].nunique())
        if record.content_id in ids:
            rank = ids.index(record.content_id) + 1
            hit = 1.0
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            rank = np.nan
            hit = 0.0
            ndcg = 0.0
        rows.append(
            {
                "model": model.name,
                "user_id": record.user_id,
                "target_content_id": record.content_id,
                "rank": rank,
                "hit_at_k": hit,
                "recall_at_k": hit,
                "ndcg_at_k": ndcg,
                "target_in_candidates": int(record.content_id in model.candidate_ids(record.user_id)),
            }
        )

    details = pd.DataFrame(rows)
    metrics = {
        "model": model.name,
        f"hit_rate@{k}": float(details["hit_at_k"].mean()),
        f"recall@{k}": float(details["recall_at_k"].mean()),
        f"ndcg@{k}": float(details["ndcg_at_k"].mean()),
        "candidate_recall": float(details["target_in_candidates"].mean()),
        "catalog_coverage": len(all_recommended) / max(len(model.contents), 1),
        "intra_list_diversity": float(np.mean(diversities)),
        "mean_topic_coverage": float(np.mean(topic_coverages)),
        "mean_source_coverage": float(np.mean(source_coverages)),
        "evaluated_users": float(len(details)),
    }
    return metrics, details


def evaluate_all(
    models: dict[str, BaseRecommender],
    test: pd.DataFrame,
    k: int = TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float]] = []
    detail_frames: list[pd.DataFrame] = []
    for model in models.values():
        metrics, details = evaluate_model(model, test=test, k=k)
        metric_rows.append(metrics)
        detail_frames.append(details)
    return pd.DataFrame(metric_rows), pd.concat(detail_frames, ignore_index=True)


def build_failure_analysis(
    details: pd.DataFrame,
    users: pd.DataFrame,
    contents: pd.DataFrame,
) -> pd.DataFrame:
    failed = details[details["hit_at_k"] == 0].copy()
    failed = failed.merge(users[["user_id", "persona", "experience_level"]], on="user_id", how="left")
    failed = failed.merge(
        contents[["content_id", "title", "topic_tags", "coin_tags", "difficulty"]],
        left_on="target_content_id",
        right_on="content_id",
        how="left",
    )
    return failed[
        ["model", "user_id", "persona", "target_content_id", "title", "topic_tags",
         "coin_tags", "target_in_candidates"]
    ].reset_index(drop=True)
