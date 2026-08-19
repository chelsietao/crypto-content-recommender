"""DuckDB product analytics and reproducible EDA outputs."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.data_pipeline import DatasetBundle

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS_DIR / ".matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402


def product_analytics(bundle: DatasetBundle) -> dict[str, pd.DataFrame]:
    connection = duckdb.connect(database=":memory:")
    connection.register("contents", bundle.contents)
    connection.register("users", bundle.users)
    connection.register("interactions", bundle.interactions)
    queries = {
        "event_funnel": """
            SELECT event_type, COUNT(*) AS events, COUNT(DISTINCT user_id) AS users
            FROM interactions GROUP BY event_type ORDER BY events DESC
        """,
        "persona_engagement": """
            SELECT u.persona,
                   COUNT(*) FILTER (WHERE i.event_type = 'impression') AS impressions,
                   COUNT(*) FILTER (WHERE i.event_type = 'click') AS clicks,
                   COUNT(*) FILTER (WHERE i.event_type IN ('like', 'save')) AS meaningful_actions,
                   ROUND(COUNT(*) FILTER (WHERE i.event_type = 'click') * 1.0 /
                         NULLIF(COUNT(*) FILTER (WHERE i.event_type = 'impression'), 0), 4) AS ctr_proxy
            FROM interactions i JOIN users u USING (user_id)
            GROUP BY u.persona ORDER BY ctr_proxy DESC
        """,
        "source_performance": """
            SELECT c.source,
                   COUNT(*) FILTER (WHERE i.event_type = 'impression') AS impressions,
                   COUNT(*) FILTER (WHERE i.event_type = 'click') AS clicks,
                   COUNT(*) FILTER (WHERE i.event_type = 'hide') AS hides,
                   ROUND(AVG(c.quality_score), 4) AS mean_quality
            FROM interactions i JOIN contents c USING (content_id)
            GROUP BY c.source ORDER BY clicks DESC
        """,
        "daily_activity": """
            SELECT CAST(DATE_TRUNC('day', timestamp) AS DATE) AS day,
                   COUNT(*) AS events, COUNT(DISTINCT user_id) AS active_users
            FROM interactions GROUP BY day ORDER BY day
        """,
    }
    return {name: connection.execute(query).df() for name, query in queries.items()}


def write_eda_report(
    bundle: DatasetBundle,
    tables: dict[str, pd.DataFrame],
    path: Path = REPORTS_DIR / "EDA_REPORT.md",
) -> None:
    event_counts = tables["event_funnel"]
    impressions = int(event_counts.loc[event_counts["event_type"] == "impression", "events"].sum())
    clicks = int(event_counts.loc[event_counts["event_type"] == "click", "events"].sum())
    ctr_proxy = clicks / max(impressions, 1)
    text = "\n".join(
        [
            "# Exploratory Data Analysis",
            "",
            "## Dataset overview",
            "",
            f"- Content items: **{len(bundle.contents):,}**",
            f"- Synthetic users: **{len(bundle.users):,}**",
            f"- Interaction events: **{len(bundle.interactions):,}**",
            f"- Impression-to-click proxy: **{ctr_proxy:.2%}**",
            "",
            "All behavior is synthetic and exists to exercise the system and evaluation pipeline.",
            "",
            "## Event funnel",
            "",
            tables["event_funnel"].to_markdown(index=False),
            "",
            "## Engagement by persona",
            "",
            tables["persona_engagement"].to_markdown(index=False),
            "",
            "## Source performance",
            "",
            tables["source_performance"].to_markdown(index=False),
            "",
            "## Data-quality observations",
            "",
            "- IDs are unique and all interaction foreign keys resolve.",
            "- All timestamps are timezone-aware UTC values.",
            "- Topic and coin tags use normalized pipe-separated values.",
            "- Exposure contains both preference-biased and exploratory impressions.",
            "- Synthetic propensities create controlled signal; they are not estimates of real users.",
            "",
            "## SQL skills demonstrated",
            "",
            "The analysis uses DuckDB joins, grouped aggregation, filtered counts, distinct counts,",
            "safe division with `NULLIF`, date truncation, and ordered product metrics.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def save_eda_charts(bundle: DatasetBundle, directory: Path = ARTIFACTS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    bundle.contents["content_type"].value_counts().plot.bar(ax=axes[0], color="#F0B90B")
    axes[0].set_title("Content type distribution")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Items")
    bundle.interactions["event_type"].value_counts().plot.bar(ax=axes[1], color="#5B8FF9")
    axes[1].set_title("Interaction event distribution")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Events")
    figure.tight_layout()
    figure.savefig(directory / "eda_distributions.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
