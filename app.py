"""Streamlit portfolio demo: user feed and data-science dashboard."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import TOP_K, REPORTS_DIR
from src.portfolio_exports import load_portfolio
from src.data_pipeline import build_dataset
from src.evaluation import evaluate_all, chronological_holdout
from src.recommenders import RankingRecommender, build_all_models, split_tags


st.set_page_config(page_title="Crypto Content Recommender", page_icon="🧭", layout="wide")


def system_signature() -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(__file__).parent / "src").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@st.cache_resource(show_spinner="Training six recommendation strategies…", max_entries=1)
def load_system(code_revision: str):
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    models = build_all_models(split.contents, bundle.users, split.train)
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
    return "Content matches: " + ("; ".join(reasons) if reasons else "content selected by the chosen strategy") + "."


bundle, split, models, metrics, details = load_system(system_signature())

st.title("Crypto Content Recommendation")
st.caption("Six baselines · Two-tower retrieval · Neural and Transformer ranking · Synthetic data")
summary = st.columns(4)
summary[0].metric("Baseline models", len(models))
summary[1].metric("Available content", len(split.contents))
summary[2].metric("Training events", f"{len(split.train):,}")
summary[3].metric("Evaluated users", len(split.test))
st.caption(f"Training and catalog cutoff: {split.cutoff.strftime('%Y-%m-%d %H:%M UTC')}")

feed_tab, dashboard_tab, neural_tab, methodology_tab = st.tabs(
    ["User Feed", "Data Scientist Dashboard", "Saved Model Results", "How it works"]
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
        st.info("Like, Save, and Hide record feedback in this browser session.")
        st.link_button("Source code", "https://github.com/chelsietao/crypto-content-recommender")
        st.link_button("System review", "https://github.com/chelsietao/crypto-content-recommender/blob/main/reports/SYSTEM_REVIEW_AND_FIXES.md")

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
                a, b, c = st.columns(3)
                for column, action in ((a, "Like"), (b, "Save"), (c, "Hide")):
                    if column.button(action, key=f"{action}-{model_name}-{user_id}-{row['content_id']}"):
                        feedback = st.session_state.setdefault("feedback", {})
                        feedback[(user_id, row["content_id"])] = action
                selected_action = st.session_state.get("feedback", {}).get((user_id, row["content_id"]))
                if selected_action:
                    st.caption(f"Session feedback: {selected_action}")
                st.caption(f"Content ID: {row['content_id']}")

with dashboard_tab:
    st.subheader("Offline model comparison")
    st.caption(
        "Training and content availability share one cutoff. Each user contributes one later unseen positive target."
    )
    display_metrics = metrics.copy()
    display_metrics["model"] = display_metrics["model"].str.replace("_", " ").str.title()
    st.dataframe(display_metrics.round(4), width="stretch", hide_index=True)

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
    st.dataframe(by_persona.round(4), width="stretch", hide_index=True)

    ranking = models[RankingRecommender.name]
    if isinstance(ranking, RankingRecommender):
        st.markdown("#### Logistic ranker coefficients")
        coefficients = ranking.feature_importance()
        if not coefficients.empty:
            st.bar_chart(coefficients.set_index("feature"))
        else:
            st.caption(f"Ranking mode: {ranking.training_mode}")

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
    st.subheader("Evaluation procedure")
    st.write(
        "The dataset contains synthetic users, content, and interactions. "
        "Training history and the content catalog share one cutoff. "
        "The ranker uses earlier history for features and a later window for labels. "
        "Hide events retain a negative preference, and hybrid recall supplies a bounded candidate pool."
    )


with neural_tab:
    st.subheader("Nine-model evaluation and saved recommendations")
    st.caption("Common temporal targets · Model-specific candidate pools · Synthetic data")
    try:
        portfolio = load_portfolio(REPORTS_DIR / "repository_evaluation")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Saved evaluation could not be verified: {exc}")
    else:
        saved_metrics = portfolio["model_metrics.csv"]
        saved_recommendations = portfolio["recommendations.csv"]
        saved_summary = portfolio["run_summary.json"]
        first, second, third = st.columns(3)
        first.metric("Evaluated models", len(saved_metrics))
        second.metric("Shared test users", saved_summary["evaluated_users"])
        third.metric("Saved neural models", saved_recommendations.model.nunique())
        st.dataframe(saved_metrics.round(4), width="stretch", hide_index=True)
        st.markdown("#### Candidate protocols")
        st.dataframe(portfolio["candidate_protocols.csv"].round(2), width="stretch", hide_index=True)
        st.markdown("#### Paired quality differences from V6")
        st.caption("2,000 bootstrap samples of the same users, percentile 95% intervals")
        st.dataframe(portfolio["quality_intervals.csv"].round(4), width="stretch", hide_index=True)
        chosen_user = st.selectbox("Saved model user", saved_recommendations.user_id.drop_duplicates(), key="saved_user")
        chosen_model = st.selectbox("Saved model", saved_recommendations.model.drop_duplicates(), key="saved_model")
        selected = saved_recommendations.loc[(saved_recommendations.user_id == chosen_user)
                                             & (saved_recommendations.model == chosen_model)]
        st.dataframe(selected[["rank", "content_id", "title", "topic_tags", "score"]].round(4),
                     width="stretch", hide_index=True)
        st.download_button("Download nine-model metrics", saved_metrics.to_csv(index=False),
                           file_name="nine_model_metrics.csv", mime="text/csv")
        st.markdown("#### Training curves")
        curve_path = (REPORTS_DIR / "transformer_reranking" / "training_curves.png") if chosen_model.startswith("v9_") else (Path(__file__).parent / "artifacts/neural/training_curves.png")
        st.image(str(curve_path), caption="Saved training and validation losses with selected checkpoints")
        st.link_button("Transformer quality and cost report", "https://github.com/chelsietao/crypto-content-recommender/blob/main/docs/README_TRANSFORMER_RERANKING.md")
