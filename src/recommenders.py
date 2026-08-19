"""Six recommendation strategies with a shared fit/recommend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EVENT_WEIGHTS, RANDOM_SEED


def split_tags(value: object) -> set[str]:
    if pd.isna(value) or value == "":
        return set()
    return {part for part in str(value).split("|") if part}


def aggregate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    weighted = interactions.copy()
    weighted["event_weight"] = weighted["event_type"].map(EVENT_WEIGHTS).astype(float)
    return (
        weighted.groupby(["user_id", "content_id"], as_index=False)
        .agg(
            preference=("event_weight", "max"),
            event_count=("event_id", "count"),
            last_timestamp=("timestamp", "max"),
            was_impressed=("event_type", lambda values: int("impression" in set(values))),
        )
        .reset_index(drop=True)
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    low = values[finite].min()
    high = values[finite].max()
    if np.isclose(low, high):
        return np.ones_like(values) if high > 0 else np.zeros_like(values)
    return (values - low) / (high - low)


class BaseRecommender(ABC):
    name = "base"

    def __init__(self) -> None:
        self.contents = pd.DataFrame()
        self.users = pd.DataFrame()
        self.interactions = pd.DataFrame()
        self.seen_by_user: dict[str, set[str]] = {}

    def fit(
        self,
        contents: pd.DataFrame,
        users: pd.DataFrame,
        interactions: pd.DataFrame,
    ) -> "BaseRecommender":
        self.contents = contents.reset_index(drop=True).copy()
        self.users = users.reset_index(drop=True).copy()
        self.interactions = interactions.copy()
        self.content_ids = self.contents["content_id"].tolist()
        self.content_index = {content_id: idx for idx, content_id in enumerate(self.content_ids)}
        self.seen_by_user = (
            interactions.groupby("user_id")["content_id"].agg(lambda values: set(values)).to_dict()
        )
        self._fit()
        return self

    def _fit(self) -> None:
        return None

    @abstractmethod
    def score_all(self, user_id: str) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        scores = np.asarray(self.score_all(user_id), dtype=float).copy()
        if exclude_seen:
            for content_id in self.seen_by_user.get(user_id, set()):
                if content_id in self.content_index:
                    scores[self.content_index[content_id]] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return pd.DataFrame(columns=[*self.contents.columns, "score", "model"])
        top_indices = np.argsort(-scores)[:k]
        result = self.contents.iloc[top_indices].copy()
        result["score"] = scores[top_indices]
        result["model"] = self.name
        return result.reset_index(drop=True)


class PopularityRecommender(BaseRecommender):
    name = "v1_popularity"

    def __init__(self, half_life_days: float = 14.0) -> None:
        super().__init__()
        self.half_life_days = half_life_days

    def _fit(self) -> None:
        weighted = self.interactions.copy()
        weighted["weight"] = weighted["event_type"].map(EVENT_WEIGHTS).astype(float)
        reference = max(
            self.contents["published_at"].max(),
            self.interactions["timestamp"].max(),
        )
        age_days = (reference - weighted["timestamp"]).dt.total_seconds() / 86400.0
        weighted["decay"] = np.exp(-np.log(2) * age_days / self.half_life_days)
        weighted["weighted_score"] = weighted["weight"] * weighted["decay"]
        totals = weighted.groupby("content_id")["weighted_score"].sum()
        freshness_days = (
            reference - self.contents["published_at"]
        ).dt.total_seconds() / 86400.0
        freshness = np.exp(-np.log(2) * freshness_days / (self.half_life_days * 2))
        self.scores = np.array(
            [float(totals.get(content_id, 0.0)) for content_id in self.content_ids]
        ) + 0.05 * freshness.to_numpy()

    def score_all(self, user_id: str) -> np.ndarray:
        del user_id
        return self.scores.copy()


class ContentBasedRecommender(BaseRecommender):
    name = "v2_content_based"

    def _fit(self) -> None:
        text = (
            self.contents["title"].fillna("")
            + " "
            + self.contents["summary"].fillna("")
            + " "
            + self.contents["topic_tags"].fillna("").str.replace("|", " ", regex=False)
            + " "
            + self.contents["coin_tags"].fillna("").str.replace("|", " ", regex=False)
        )
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
        self.item_matrix = self.vectorizer.fit_transform(text)
        aggregated = aggregate_interactions(self.interactions)
        self.profiles: dict[str, sparse.csr_matrix] = {}
        for user_id, group in aggregated.groupby("user_id"):
            indices: list[int] = []
            weights: list[float] = []
            for row in group.itertuples():
                if row.content_id in self.content_index and row.preference != 0:
                    indices.append(self.content_index[row.content_id])
                    weights.append(float(row.preference))
            if indices:
                positive_weights = np.maximum(np.asarray(weights), 0.0)
                if positive_weights.sum() > 0:
                    profile = sparse.csr_matrix(
                        self.item_matrix[indices].multiply(positive_weights[:, None]).sum(axis=0)
                    )
                    self.profiles[user_id] = profile

    def _explicit_profile(self, user_id: str) -> sparse.csr_matrix | None:
        matched = self.users[self.users["user_id"] == user_id]
        if matched.empty:
            return None
        user = matched.iloc[0]
        query = (
            str(user["preferred_topics"]).replace("|", " ")
            + " "
            + str(user["preferred_coins"]).replace("|", " ")
            + " "
            + str(user["experience_level"])
        )
        return self.vectorizer.transform([query])

    def score_all(self, user_id: str) -> np.ndarray:
        behavior_profile = self.profiles.get(user_id)
        explicit_profile = self._explicit_profile(user_id)
        if behavior_profile is None and explicit_profile is None:
            return np.zeros(len(self.contents))
        behavior_scores = (
            cosine_similarity(behavior_profile, self.item_matrix).ravel()
            if behavior_profile is not None and behavior_profile.nnz > 0
            else np.zeros(len(self.contents))
        )
        explicit_scores = (
            cosine_similarity(explicit_profile, self.item_matrix).ravel()
            if explicit_profile is not None and explicit_profile.nnz > 0
            else np.zeros(len(self.contents))
        )
        # Explicit onboarding interests remain useful when behavioral history is
        # short; behavior receives more weight as the primary personalization signal.
        return 0.60 * behavior_scores + 0.40 * explicit_scores


class CollaborativeFilteringRecommender(BaseRecommender):
    name = "v3_collaborative_filtering"

    def _fit(self) -> None:
        aggregated = aggregate_interactions(self.interactions)
        user_ids = self.users["user_id"].tolist()
        self.user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
        matrix = np.zeros((len(user_ids), len(self.contents)), dtype=float)
        for row in aggregated.itertuples():
            if row.user_id in self.user_index and row.content_id in self.content_index:
                matrix[self.user_index[row.user_id], self.content_index[row.content_id]] = row.preference
        self.user_item = matrix
        # NumPy's Accelerate-backed matmul can emit spurious overflow warnings
        # for small normalized matrices on some macOS/Python combinations.
        # einsum is deterministic here and makes the cosine calculation explicit.
        item_vectors = matrix.T
        norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
        normalized = np.divide(
            item_vectors,
            norms,
            out=np.zeros_like(item_vectors),
            where=norms > 0,
        )
        self.item_similarity = np.einsum("ik,jk->ij", normalized, normalized)
        np.fill_diagonal(self.item_similarity, 0.0)

    def score_all(self, user_id: str) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(len(self.contents))
        vector = self.user_item[self.user_index[user_id]]
        positive = np.maximum(vector, 0.0)
        if positive.sum() == 0:
            return np.zeros(len(self.contents))
        active = (positive > 0).astype(float)
        denominator = np.einsum("ij,j->i", np.abs(self.item_similarity), active) + 1e-9
        numerator = np.einsum("ij,j->i", self.item_similarity, positive)
        return numerator / denominator


class HybridRecommender(BaseRecommender):
    name = "v4_hybrid"

    def __init__(
        self,
        popularity_weight: float = 0.20,
        content_weight: float = 0.40,
        collaborative_weight: float = 0.35,
        exploration_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.weights = (
            popularity_weight,
            content_weight,
            collaborative_weight,
            exploration_weight,
        )

    def _fit(self) -> None:
        self.popularity = PopularityRecommender().fit(self.contents, self.users, self.interactions)
        self.content = ContentBasedRecommender().fit(self.contents, self.users, self.interactions)
        self.collaborative = CollaborativeFilteringRecommender().fit(
            self.contents, self.users, self.interactions
        )
        newest = self.contents["published_at"].rank(pct=True).to_numpy()
        self.exploration = newest * (1.0 - minmax(self.popularity.scores))

    def score_components(self, user_id: str) -> dict[str, np.ndarray]:
        return {
            "popularity": minmax(self.popularity.score_all(user_id)),
            "content": minmax(self.content.score_all(user_id)),
            "collaborative": minmax(self.collaborative.score_all(user_id)),
            "exploration": minmax(self.exploration),
        }

    def score_all(self, user_id: str) -> np.ndarray:
        components = self.score_components(user_id)
        return sum(
            weight * components[name]
            for weight, name in zip(
                self.weights,
                ("popularity", "content", "collaborative", "exploration"),
            )
        )


class RankingRecommender(BaseRecommender):
    name = "v5_logistic_ranker"
    feature_names = (
        "content_similarity",
        "popularity",
        "collaborative",
        "topic_overlap",
        "coin_overlap",
        "difficulty_match",
        "quality_score",
        "freshness",
        "user_activity",
    )

    def _fit(self) -> None:
        self.hybrid = HybridRecommender().fit(self.contents, self.users, self.interactions)
        aggregated = aggregate_interactions(self.interactions)
        latest = self.contents["published_at"].max()
        self.freshness = np.exp(
            -np.log(2)
            * ((latest - self.contents["published_at"]).dt.total_seconds() / 86400.0)
            / 30.0
        ).to_numpy()
        activity = aggregated.groupby("user_id")["event_count"].sum()
        self.activity = activity.to_dict()
        rows: list[np.ndarray] = []
        labels: list[int] = []
        for record in aggregated.itertuples():
            index = self.content_index[record.content_id]
            rows.append(self._features(record.user_id)[index])
            labels.append(int(record.preference > 0))
        x = np.vstack(rows)
        y = np.asarray(labels)
        self.model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_SEED,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        self.model.fit(x, y)

    def _overlap_features(self, user_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matched = self.users[self.users["user_id"] == user_id]
        if matched.empty:
            return (
                np.zeros(len(self.contents)),
                np.zeros(len(self.contents)),
                np.zeros(len(self.contents)),
            )
        user = matched.iloc[0]
        user_topics = split_tags(user["preferred_topics"])
        user_coins = split_tags(user["preferred_coins"])
        topic_overlap = np.array(
            [len(user_topics & split_tags(value)) / max(len(user_topics), 1) for value in self.contents["topic_tags"]]
        )
        coin_overlap = np.array(
            [len(user_coins & split_tags(value)) / max(len(user_coins), 1) for value in self.contents["coin_tags"]]
        )
        difficulty = (self.contents["difficulty"] == user["experience_level"]).astype(float).to_numpy()
        return topic_overlap, coin_overlap, difficulty

    def _features(self, user_id: str) -> np.ndarray:
        components = self.hybrid.score_components(user_id)
        topic_overlap, coin_overlap, difficulty = self._overlap_features(user_id)
        activity = np.full(len(self.contents), np.log1p(self.activity.get(user_id, 0)))
        return np.column_stack(
            [
                components["content"],
                components["popularity"],
                components["collaborative"],
                topic_overlap,
                coin_overlap,
                difficulty,
                self.contents["quality_score"].to_numpy(),
                self.freshness,
                activity,
            ]
        )

    def score_all(self, user_id: str) -> np.ndarray:
        features = self.model.named_steps["scale"].transform(self._features(user_id))
        logistic = self.model.named_steps["logistic"]
        logits = np.einsum("ij,j->i", features, logistic.coef_[0]) + logistic.intercept_[0]
        logits = np.clip(logits, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        # The learned score refines rather than replaces candidate relevance.
        # This mirrors a two-stage system and protects against a weak ranker.
        hybrid_scores = minmax(self.hybrid.score_all(user_id))
        return 0.65 * probabilities + 0.35 * hybrid_scores

    def feature_importance(self) -> pd.DataFrame:
        coefficients = self.model.named_steps["logistic"].coef_[0]
        return pd.DataFrame(
            {"feature": self.feature_names, "coefficient": coefficients}
        ).sort_values("coefficient", ascending=False)


class DiversityRecommender(BaseRecommender):
    name = "v6_diversity_reranking"

    def __init__(
        self,
        candidate_multiplier: int = 6,
        relevance_weight: float = 0.76,
        max_per_source: int = 2,
        max_per_primary_topic: int = 3,
    ) -> None:
        super().__init__()
        self.candidate_multiplier = candidate_multiplier
        self.relevance_weight = relevance_weight
        self.max_per_source = max_per_source
        self.max_per_primary_topic = max_per_primary_topic

    def _fit(self) -> None:
        self.ranker = RankingRecommender().fit(self.contents, self.users, self.interactions)

    def score_all(self, user_id: str) -> np.ndarray:
        return self.ranker.score_all(user_id)

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        scores = self.score_all(user_id).copy()
        if exclude_seen:
            for content_id in self.seen_by_user.get(user_id, set()):
                if content_id in self.content_index:
                    scores[self.content_index[content_id]] = -np.inf
        candidate_count = min(k * self.candidate_multiplier, int(np.isfinite(scores).sum()))
        candidates = list(np.argsort(-scores)[:candidate_count])
        selected: list[int] = []
        source_counts: Counter[str] = Counter()
        topic_counts: Counter[str] = Counter()

        while candidates and len(selected) < k:
            best_index: int | None = None
            best_value = -np.inf
            for index in candidates:
                row = self.contents.iloc[index]
                primary_topic = str(row["topic_tags"]).split("|")[0]
                if source_counts[str(row["source"])] >= self.max_per_source:
                    continue
                if topic_counts[primary_topic] >= self.max_per_primary_topic:
                    continue
                if selected:
                    candidate_tags = split_tags(row["topic_tags"])
                    similarities = []
                    for selected_index in selected:
                        selected_tags = split_tags(self.contents.iloc[selected_index]["topic_tags"])
                        union = candidate_tags | selected_tags
                        similarities.append(len(candidate_tags & selected_tags) / max(len(union), 1))
                    novelty = 1.0 - max(similarities)
                else:
                    novelty = 1.0
                value = self.relevance_weight * scores[index] + (1.0 - self.relevance_weight) * novelty
                if value > best_value:
                    best_value = value
                    best_index = index
            if best_index is None:
                best_index = candidates[0]
            candidates.remove(best_index)
            selected.append(best_index)
            selected_row = self.contents.iloc[best_index]
            source_counts[str(selected_row["source"])] += 1
            topic_counts[str(selected_row["topic_tags"]).split("|")[0]] += 1

        result = self.contents.iloc[selected].copy()
        result["score"] = scores[selected]
        result["model"] = self.name
        return result.reset_index(drop=True)


def build_all_models(
    contents: pd.DataFrame,
    users: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, BaseRecommender]:
    model_classes = (
        PopularityRecommender,
        ContentBasedRecommender,
        CollaborativeFilteringRecommender,
        HybridRecommender,
        RankingRecommender,
        DiversityRecommender,
    )
    return {
        model_class.name: model_class().fit(contents, users, interactions)
        for model_class in model_classes
    }
