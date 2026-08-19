"""Temporal offline evaluation for the recommendation strategies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import EVENT_WEIGHTS, TOP_K
from src.recommenders import BaseRecommender, split_tags


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame


def leave_latest_positive_out(interactions: pd.DataFrame) -> TemporalSplit:
    weighted = interactions.copy()
    weighted["weight"] = weighted["event_type"].map(EVENT_WEIGHTS)
    positives = weighted[weighted["weight"] > 0].copy()
    latest = (
        positives.sort_values("timestamp")
        .groupby("user_id", as_index=False)
        .tail(1)[["user_id", "content_id", "timestamp"]]
        .rename(columns={"timestamp": "test_timestamp"})
    )
    holdout_pairs = set(zip(latest["user_id"], latest["content_id"]))
    keep = [
        (user_id, content_id) not in holdout_pairs
        for user_id, content_id in zip(weighted["user_id"], weighted["content_id"])
    ]
    train = weighted.loc[keep].drop(columns="weight").reset_index(drop=True)
    return TemporalSplit(train=train, test=latest.reset_index(drop=True))


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
            rank = None
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
            }
        )

    details = pd.DataFrame(rows)
    metrics = {
        "model": model.name,
        f"hit_rate@{k}": float(details["hit_at_k"].mean()),
        f"recall@{k}": float(details["recall_at_k"].mean()),
        f"ndcg@{k}": float(details["ndcg_at_k"].mean()),
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
    failed["failure_hypothesis"] = np.select(
        [
            failed["model"].eq("v1_popularity"),
            failed["model"].eq("v3_collaborative_filtering"),
            failed["experience_level"].ne(failed["difficulty"]),
        ],
        [
            "Global popularity did not capture the user's niche interest.",
            "Sparse co-interaction history limited collaborative signal.",
            "Recommended pool may not match the user's preferred difficulty.",
        ],
        default="The held-out interest was not represented strongly enough in candidate scores.",
    )
    return failed[
        [
            "model",
            "user_id",
            "persona",
            "target_content_id",
            "title",
            "topic_tags",
            "coin_tags",
            "failure_hypothesis",
        ]
    ].head(80)
