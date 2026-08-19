"""Run the full data, modeling, evaluation, and reporting workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.analytics import product_analytics, save_eda_charts, write_eda_report
from src.config import ARTIFACTS_DIR, REPORTS_DIR, TOP_K
from src.data_pipeline import build_dataset
from src.evaluation import build_failure_analysis, evaluate_all, leave_latest_positive_out
from src.recommenders import RankingRecommender, build_all_models


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def write_experiment_report(metrics: pd.DataFrame, path: Path) -> None:
    best_relevance = metrics.sort_values(f"ndcg@{TOP_K}", ascending=False).iloc[0]
    best_diversity = metrics.sort_values("intra_list_diversity", ascending=False).iloc[0]
    lines = [
        "# Offline Experiment Results",
        "",
        "## Evaluation design",
        "",
        "For each synthetic user, the latest positive user-content interaction is held out.",
        "All events for that held-out pair are removed from training. Models must rank the",
        f"held-out item in the top {TOP_K} while excluding items observed during training.",
        "This chronological leave-one-out design is more realistic than a random split, but",
        "results remain synthetic and do not demonstrate production impact.",
        "",
        "## Results",
        "",
        metrics.round(4).to_markdown(index=False),
        "",
        "## Findings",
        "",
        f"- Best NDCG@{TOP_K}: **{best_relevance['model']}** ({_format_metric(best_relevance[f'ndcg@{TOP_K}'])}).",
        f"- Best intra-list diversity: **{best_diversity['model']}** ({_format_metric(best_diversity['intra_list_diversity'])}).",
        "- Popularity is a necessary baseline, not a personalized solution.",
        "- Content-based retrieval supports new-item and low-history scenarios.",
        "- Collaborative filtering depends on sufficient co-interaction density.",
        "- Hybrid and ranking stages combine complementary signals.",
        "- Diversity reranking explicitly trades a small amount of relevance for healthier exposure.",
        "",
        "## Interpretation limits",
        "",
        "- Users and interactions are deterministic synthetic data.",
        "- Offline ranking metrics cannot be interpreted as real CTR or retention lift.",
        "- A real deployment requires online experimentation, safety review, and data-quality monitoring.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ab_test_proposal(path: Path) -> None:
    text = """# A/B Test Proposal

## Objective

Determine whether the personalized, diversity-aware feed improves meaningful content discovery
relative to a time-decayed popularity feed.

## Variants

- **Control:** time-decayed popularity recommendation (V1).
- **Treatment:** logistic ranking followed by diversity reranking (V6).

## Hypothesis

Treatment increases meaningful engagement and content coverage without materially increasing
negative feedback.

## Unit and allocation

- Randomization unit: authenticated user ID.
- Allocation: 50% control / 50% treatment.
- Sticky assignment for the full experiment.
- Exclude employees, bots, and users with invalid instrumentation.

## Metrics

- Primary: meaningful engagement rate (like or save per eligible impression).
- Secondary: CTR, dwell time, session depth, day-7 return rate, creator coverage.
- Guardrails: hide/report rate, latency p95, source concentration, policy-violating exposure.

## Analysis

- Pre-register sample-size assumptions before launch.
- Compare intention-to-treat groups with 95% confidence intervals.
- Segment by new vs. established users, language, and activity level.
- Do not stop early solely because a metric becomes temporarily significant.
- Ship only if the primary metric improves and no guardrail crosses its agreed threshold.

## Instrumentation requirements

Every recommendation impression must record model version, rank, request ID, user assignment,
content ID, timestamp in UTC, latency, and subsequent interaction events.

## Caveat

This document is a product experiment proposal. No online experiment was performed in this
portfolio project.
"""
    path.write_text(text, encoding="utf-8")


def write_failure_report(failures: pd.DataFrame, path: Path) -> None:
    counts = failures.groupby(["model", "failure_hypothesis"]).size().reset_index(name="examples")
    text = "\n".join(
        [
            "# Recommendation Failure Analysis",
            "",
            "This report samples missed held-out items and assigns a testable failure hypothesis.",
            "The labels are diagnostic heuristics, not verified root causes.",
            "",
            "## Failure categories",
            "",
            counts.to_markdown(index=False),
            "",
            "## Sample cases",
            "",
            failures.head(30).to_markdown(index=False),
            "",
            "## Follow-up experiments",
            "",
            "1. Add position-aware negative sampling from eligible impressions.",
            "2. Compare short-term and long-term interest profiles.",
            "3. Calibrate ranker probabilities by persona and activity cohort.",
            "4. Add semantic embeddings and multilingual content representations.",
            "5. Test source-quality and safety constraints as explicit guardrails.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset()
    analytics_tables = product_analytics(bundle)
    write_eda_report(bundle, analytics_tables)
    save_eda_charts(bundle)
    split = leave_latest_positive_out(bundle.interactions)
    models = build_all_models(bundle.contents, bundle.users, split.train)
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
    write_ab_test_proposal(REPORTS_DIR / "AB_TEST_PROPOSAL.md")
    write_failure_report(failures, REPORTS_DIR / "FAILURE_ANALYSIS.md")
    (ARTIFACTS_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "contents": len(bundle.contents),
                "users": len(bundle.users),
                "events": len(bundle.interactions),
                "train_events": len(split.train),
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
