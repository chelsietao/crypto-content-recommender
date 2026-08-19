"""Streamlit portfolio demo: user feed and data-science dashboard."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, TOP_K
from src.data_pipeline import build_dataset
from src.evaluation import evaluate_all, leave_latest_positive_out
from src.recommenders import RankingRecommender, build_all_models, split_tags


st.set_page_config(page_title="Crypto Content Recommender", page_icon="🧭", layout="wide")


@st.cache_resource(show_spinner="Training six recommendation strategies…")
def load_system():
    bundle = build_dataset(save=True)
    split = leave_latest_positive_out(bundle.interactions)
    models = build_all_models(bundle.contents, bundle.users, split.train)
    metrics, details = evaluate_all(models, split.test)
    return bundle, split, models, metrics, details


def explain(row: pd.Series, user: pd.Series) -> str:
    topics = split_tags(row["topic_tags"])
    preferred_topics = split_tags(user["preferred_topics"])
    coins = split_tags(row["coin_tags"])
    preferred_coins = split_tags(user["preferred_coins"])
    reasons: list[str] = []
    if topics & preferred_topics:
        reasons.append("topic match: " + ", ".join(sorted(topics & preferred_topics)))
    if coins & preferred_coins:
        reasons.append("asset match: " + ", ".join(sorted(coins & preferred_coins)))
    if row["difficulty"] == user["experience_level"]:
        reasons.append(f"{row['difficulty']} difficulty")
    return "Recommended because of " + ("; ".join(reasons) if reasons else "freshness and discovery value") + "."


bundle, split, models, metrics, details = load_system()

st.title("🧭 Personalized & Diverse Crypto Content Recommendation")
st.caption(
    "A synthetic, offline research prototype inspired by content-community recommendation problems. "
    "Not affiliated with Binance and not financial advice."
)

feed_tab, dashboard_tab, methodology_tab = st.tabs(
    ["User Feed", "Data Scientist Dashboard", "Methodology & Limits"]
)

with feed_tab:
    left, right = st.columns([1, 3])
    with left:
        persona = st.selectbox("Choose a persona", bundle.users["persona"].drop_duplicates())
        eligible_users = bundle.users[bundle.users["persona"] == persona]
        user_id = st.selectbox("Synthetic user", eligible_users["user_id"])
        model_name = st.selectbox(
            "Recommendation strategy",
            list(models),
            index=list(models).index("v6_diversity_reranking"),
            format_func=lambda value: value.replace("_", " ").title(),
        )
        count = st.slider("Items", min_value=5, max_value=15, value=10)
        st.info("Like/Save/Hide buttons are UI demonstrations and remain in this browser session only.")

    user = bundle.users[bundle.users["user_id"] == user_id].iloc[0]
    recommendations = models[model_name].recommend(user_id, k=count)
    with right:
        st.subheader(f"Feed for {persona}")
        st.write(
            f"**Topics:** {user['preferred_topics']}  ·  **Assets:** {user['preferred_coins']}  ·  "
            f"**Level:** {user['experience_level']}"
        )
        for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
            with st.container(border=True):
                st.markdown(f"### {rank}. {row['title']}")
                st.write(row["summary"])
                st.caption(
                    f"{row['source']} · {row['content_type']} · {row['difficulty']} · "
                    f"topics: {row['topic_tags']} · score: {row['score']:.3f}"
                )
                st.success(explain(row, user), icon="💡")
                a, b, c, d = st.columns([1, 1, 1, 4])
                a.button("Like", key=f"like-{model_name}-{user_id}-{row['content_id']}")
                b.button("Save", key=f"save-{model_name}-{user_id}-{row['content_id']}")
                c.button("Hide", key=f"hide-{model_name}-{user_id}-{row['content_id']}")
                d.link_button("Original link (synthetic)", row["url"])

with dashboard_tab:
    st.subheader("Offline model comparison")
    st.caption(
        "Latest positive item per user is held out. Metrics are synthetic and must not be read as online lift."
    )
    display_metrics = metrics.copy()
    display_metrics["model"] = display_metrics["model"].str.replace("_", " ").str.title()
    st.dataframe(display_metrics.round(4), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Relevance: NDCG@{TOP_K}")
        st.bar_chart(metrics.set_index("model")[[f"ndcg@{TOP_K}"]])
    with c2:
        st.markdown("#### Intra-list diversity")
        st.bar_chart(metrics.set_index("model")[["intra_list_diversity"]])

    st.markdown("#### Performance by persona")
    persona_details = details.merge(bundle.users[["user_id", "persona"]], on="user_id", how="left")
    by_persona = (
        persona_details.groupby(["model", "persona"], as_index=False)[[f"ndcg_at_k", "hit_at_k"]]
        .mean()
        .rename(columns={"ndcg_at_k": f"ndcg@{TOP_K}", "hit_at_k": f"hit_rate@{TOP_K}"})
    )
    st.dataframe(by_persona.round(4), use_container_width=True, hide_index=True)

    ranking = models[RankingRecommender.name]
    if isinstance(ranking, RankingRecommender):
        st.markdown("#### Logistic ranker coefficients")
        coefficients = ranking.feature_importance().set_index("feature")
        st.bar_chart(coefficients)

    st.markdown("#### Dataset overview")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Content items", len(bundle.contents))
    d2.metric("Synthetic users", len(bundle.users))
    d3.metric("Interaction events", len(bundle.interactions))
    d4.metric("Held-out users", len(split.test))

with methodology_tab:
    st.subheader("Six versions")
    st.markdown(
        """
1. **Popularity:** event-weighted popularity with temporal decay.
2. **Content-based:** TF-IDF user profiles over titles, summaries, topics, and assets.
3. **Collaborative filtering:** item-item similarity from the user-item interaction matrix.
4. **Hybrid:** normalized popularity, content, collaborative, and exploration signals.
5. **Ranking:** logistic regression over relevance, user, quality, and freshness features.
6. **Diversity reranking:** relevance-aware reranking with source and topic exposure constraints.
"""
    )
    st.subheader("Responsible interpretation")
    st.warning(
        "All users and interactions are synthetic. This prototype validates system design and "
        "evaluation code; it does not demonstrate impact on real users, predict prices, or provide investment advice."
    )
    st.subheader("Production next steps")
    st.markdown(
        "Instrument eligible impressions, establish bot and policy filters, pre-register an A/B test, "
        "monitor latency and negative feedback, and review fairness across languages, creators, and user cohorts."
    )
