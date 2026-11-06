"""Run the full data, modeling, evaluation, and reporting workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics import product_analytics, save_eda_charts, write_eda_report
from src.config import ARTIFACTS_DIR, REPORTS_DIR, TOP_K
from src.data_pipeline import build_dataset
from src.evaluation import build_failure_analysis, evaluate_all, chronological_holdout
from src.recommenders import RankingRecommender, build_all_models


def write_experiment_report(metrics: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Offline Recommendation Evaluation", "",
        "Training interactions and the content catalog are frozen at one global cutoff, "
        "and each evaluated user contributes the first eligible positive item after that cutoff.",
        "Targets were available at the cutoff and absent from that user's training history, "
        "and hidden pairs are excluded from positive labels.", "",
        "The ranker uses a separate historical feature snapshot within the training period, "
        "and labels come from the subsequent window.", "",
        "## Results", "", metrics.round(4).to_markdown(index=False), "",
        "Hit Rate and Recall match because evaluation uses one target per user, "
        "and candidate recall records whether that target entered the model's candidate pool.",
        "Catalog coverage uses the frozen content catalog as its denominator, "
        "and diversity uses the average pairwise topic-tag distance within each result list.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_failure_report(failures: pd.DataFrame, path: Path) -> None:
    counts = failures.groupby(["model", "target_in_candidates"]).size().reset_index(name="missed_targets")
    text = "\n".join([
        "# Recommendation Miss Analysis", "",
        "Every missed target is retained in the CSV, with its candidate-pool membership recorded.",
        "A value of 0 identifies a recall-stage miss, and a value of 1 identifies a target that "
        "entered the pool but did not appear in the final result list.", "",
        "## Counts", "", counts.to_markdown(index=False), "",
        "## Sample records", "", failures.groupby("model", group_keys=False).head(5).to_markdown(index=False),
    ])
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset()
    analytics_tables = product_analytics(bundle)
    write_eda_report(bundle, analytics_tables)
    save_eda_charts(bundle)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    models = build_all_models(split.contents, bundle.users, split.train)
    metrics, details = evaluate_all(models, split.test)
    failures = build_failure_analysis(details, bundle.users, bundle.contents)

    metrics.to_csv(ARTIFACTS_DIR / "model_metrics.csv", index=False)
    details.to_csv(ARTIFACTS_DIR / "evaluation_details.csv", index=False)
    failures.to_csv(ARTIFACTS_DIR / "failure_analysis.csv", index=False)
    split.train.to_csv(ARTIFACTS_DIR / "train_interactions.csv", index=False)
    split.test.to_csv(ARTIFACTS_DIR / "test_holdout.csv", index=False)

    ranking = models[RankingRecommender.name]
    if isinstance(ranking, RankingRecommender):
        ranking.feature_importance().to_csv(ARTIFACTS_DIR / "ranker_coefficients.csv", index=False)

    sample_user = bundle.users.iloc[0]["user_id"]
    for model_name, model in models.items():
        model.recommend(sample_user, k=TOP_K).to_csv(
            ARTIFACTS_DIR / f"recommendations_{model_name}.csv", index=False
        )

    write_experiment_report(metrics, REPORTS_DIR / "EXPERIMENT_RESULTS.md")
    write_failure_report(failures, REPORTS_DIR / "FAILURE_ANALYSIS.md")
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "contents": len(bundle.contents),
                "users": len(bundle.users),
                "events": len(bundle.interactions),
                "train_events": len(split.train),
                "training_cutoff": split.cutoff.isoformat(),
                "eligible_contents": len(split.contents),
                "ranker_feature_cutoff": ranking.feature_cutoff.isoformat(),
                "ranker_training_examples": ranking.training_feature_rows,
                "ranker_training_mode": ranking.training_mode,
                "test_users": len(split.test),
                "models": list(models),
                "top_k": TOP_K,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(metrics.round(4).to_string(index=False))
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
