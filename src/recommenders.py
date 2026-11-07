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
    if not interactions["event_type"].isin(EVENT_WEIGHTS).all():
        raise ValueError("Unknown interaction event type")
    weighted = interactions.copy()
    weighted["event_weight"] = weighted["event_type"].map(EVENT_WEIGHTS).astype(float)
    return (
        weighted.groupby(["user_id", "content_id"], as_index=False)
        .agg(
            preference=("event_weight", lambda values: float(values.min()) if (values < 0).any() else float(values.max())),
            event_count=("event_id", "count"),
            last_timestamp=("timestamp", "max"),
            was_impressed=("event_type", lambda values: int("impression" in set(values))),
        )
        .reset_index(drop=True)
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.zeros_like(values)
    finite = np.isfinite(values)
    if not finite.any():
        return result
    low, high = values[finite].min(), values[finite].max()
    if np.isclose(low, high):
        result[finite] = float(high > 0)
    else:
        result[finite] = (values[finite] - low) / (high - low)
    return result


def validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
        raise ValueError("k must be a non-negative integer")


def ordered_indices(scores: np.ndarray, content_ids: list[str]) -> np.ndarray:
    """Order finite scores by 12-decimal score, then content ID for numeric ties."""
    finite = np.flatnonzero(np.isfinite(scores))
    keys = np.asarray(content_ids)[finite]
    return finite[np.lexsort((keys, -np.round(scores[finite], 12)))]


def top_results(contents: pd.DataFrame, scores: np.ndarray, k: int, model: str) -> pd.DataFrame:
    validate_k(k)
    indices = ordered_indices(scores, contents["content_id"].tolist())[:k]
    result = contents.iloc[indices].copy()
    result["score"] = scores[indices]
    result["model"] = model
    return result.reset_index(drop=True)


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
        if contents.empty:
            raise ValueError("At least one content item is required")
        if not contents["content_id"].is_unique or not users["user_id"].is_unique:
            raise ValueError("Content and user IDs must be unique")
        if not set(interactions["content_id"]).issubset(set(contents["content_id"])):
            raise ValueError("Interactions reference unknown content IDs")
        if not set(interactions["user_id"]).issubset(set(users["user_id"])):
            raise ValueError("Interactions reference unknown user IDs")
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

    def candidate_ids(self, user_id: str, exclude_seen: bool = True) -> list[str]:
        seen = self.seen_by_user.get(user_id, set()) if exclude_seen else set()
        return [content_id for content_id in self.content_ids if content_id not in seen]

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        scores = np.asarray(self.score_all(user_id), dtype=float).copy()
        if exclude_seen:
            for content_id in self.seen_by_user.get(user_id, set()):
                if content_id in self.content_index:
                    scores[self.content_index[content_id]] = -np.inf
        return top_results(self.contents, scores, k, self.name)


class PopularityRecommender(BaseRecommender):
    name = "v1_popularity"

    def __init__(self, half_life_days: float = 14.0) -> None:
        super().__init__()
        self.half_life_days = half_life_days

    def _fit(self) -> None:
        weighted = self.interactions.copy()
        weighted["weight"] = weighted["event_type"].map(EVENT_WEIGHTS).astype(float)
        reference = self.contents["published_at"].max()
        if not self.interactions.empty:
            reference = max(reference, self.interactions["timestamp"].max())
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

    def __init__(self, neighbors: int = 40, block_size: int = 128) -> None:
        super().__init__()
        if neighbors <= 0 or block_size <= 0:
            raise ValueError("neighbors and block_size must be positive")
        self.neighbors = neighbors
        self.block_size = block_size

    def _fit(self) -> None:
        aggregated = aggregate_interactions(self.interactions)
        self.user_index = {value: index for index, value in enumerate(self.users["user_id"])}
        valid = aggregated.loc[
            aggregated["user_id"].isin(self.user_index)
            & aggregated["content_id"].isin(self.content_index)
        ]
        self.user_item = sparse.csr_matrix(
            (valid["preference"].to_numpy(),
             (valid["user_id"].map(self.user_index), valid["content_id"].map(self.content_index))),
            shape=(len(self.user_index), len(self.contents)), dtype=float,
        )
        item_vectors = self.user_item.T.tocsr()
        blocks = []
        for start in range(0, len(self.contents), self.block_size):
            similarities = cosine_similarity(
                item_vectors[start:start + self.block_size], item_vectors, dense_output=False
            ).tocsr()
            for row in range(similarities.shape[0]):
                left, right = similarities.indptr[row:row + 2]
                values = similarities.data[left:right]
                indices = similarities.indices[left:right]
                values[indices == start + row] = 0.0
                positive = np.flatnonzero(values > 0)
                keys = np.asarray(self.content_ids)[indices[positive]]
                order = np.lexsort((keys, -np.round(values[positive], 12)))
                keep = positive[order[:self.neighbors]]
                drop = np.ones(len(values), dtype=bool)
                drop[keep] = False
                values[drop] = 0.0
            similarities.eliminate_zeros()
            blocks.append(similarities)
        self.item_similarity = sparse.vstack(blocks, format="csr")

    def score_all(self, user_id: str) -> np.ndarray:
        if user_id not in self.user_index:
            return np.zeros(len(self.contents))
        vector = self.user_item.getrow(self.user_index[user_id]).T
        active = vector.copy()
        active.data = np.ones_like(active.data)
        numerator = (self.item_similarity @ vector).toarray().ravel()
        denominator = (abs(self.item_similarity) @ active).toarray().ravel()
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


class HybridRecommender(BaseRecommender):
    name = "v4_hybrid"

    def __init__(
        self,
        popularity_weight: float = 0.20,
        content_weight: float = 0.40,
        collaborative_weight: float = 0.35,
        exploration_weight: float = 0.05,
        candidates_per_channel: int = 20,
    ) -> None:
        super().__init__()
        if candidates_per_channel <= 0:
            raise ValueError("candidates_per_channel must be positive")
        self.candidates_per_channel = candidates_per_channel
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
        self._component_cache: dict[str, dict[str, np.ndarray]] = {}

    def score_components(self, user_id: str) -> dict[str, np.ndarray]:
        if user_id in self._component_cache:
            return self._component_cache[user_id]
        components = {
            "popularity": minmax(self.popularity.score_all(user_id)),
            "content": minmax(self.content.score_all(user_id)),
            "collaborative": minmax(self.collaborative.score_all(user_id)),
            "exploration": minmax(self.exploration),
        }
        self._component_cache[user_id] = components
        return components

    def score_all(self, user_id: str) -> np.ndarray:
        components = self.score_components(user_id)
        return sum(
            weight * components[name]
            for weight, name in zip(
                self.weights,
                ("popularity", "content", "collaborative", "exploration"),
            )
        )
    def candidate_ids(self, user_id: str, exclude_seen: bool = True) -> list[str]:
        seen = self.seen_by_user.get(user_id, set()) if exclude_seen else set()
        selected: set[int] = set()
        for scores in self.score_components(user_id).values():
            eligible = [index for index in ordered_indices(scores, self.content_ids)
                        if self.content_ids[index] not in seen]
            selected.update(eligible[:self.candidates_per_channel])
        return [self.content_ids[index] for index in sorted(selected)]

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        validate_k(k)
        candidates = set(self.candidate_ids(user_id, exclude_seen=exclude_seen))
        scores = self.score_all(user_id).copy()
        scores[[value not in candidates for value in self.content_ids]] = -np.inf
        return top_results(self.contents, scores, k, self.name)


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
        # A separate historical snapshot supplies features for a later label window.
        times = self.interactions["timestamp"].sort_values().drop_duplicates().reset_index(drop=True)
        self.model = None
        self.constant_probability: float | None = None
        self.training_mode = "hybrid_fallback"
        self.training_feature_rows = 0
        if len(times) < 2:
            reference = times.iloc[-1] if len(times) else self.contents["published_at"].max()
            self.feature_cutoff = reference
            self.label_window_start = pd.NaT
            hybrid = HybridRecommender().fit(self.contents, self.users, self.interactions)
            self._set_snapshot(hybrid, self.interactions, reference)
            return
        boundary = times.iloc[min(max(int(len(times) * 0.7) - 1, 0), len(times) - 2)]
        history = self.interactions.loc[self.interactions["timestamp"] <= boundary]
        label_events = self.interactions.loc[self.interactions["timestamp"] > boundary]
        historical_contents = self.contents.loc[self.contents["published_at"] <= boundary]
        feature_hybrid = HybridRecommender().fit(historical_contents, self.users, history)
        labels = aggregate_interactions(label_events)
        rows: list[np.ndarray] = []
        targets: list[int] = []
        eligible_ids = set(historical_contents["content_id"])
        # Features and candidate membership are computed once per user for the frozen snapshot.
        self._set_snapshot(feature_hybrid, history, boundary)
        self.training_feature_rows = 0
        for user_id, group in labels.groupby("user_id"):
            candidates = set(feature_hybrid.candidate_ids(user_id))
            group = group.loc[group["content_id"].isin(candidates & eligible_ids)]
            if group.empty:
                continue
            features = self._features(user_id)
            for record in group.itertuples():
                rows.append(features[self.content_index[record.content_id]])
                targets.append(int(record.preference > 0))
        self.training_feature_rows = len(rows)
        self.feature_cutoff = boundary
        self.label_window_start = label_events["timestamp"].min()
        if rows and len(set(targets)) == 2:
            self.model = Pipeline([
                ("scale", StandardScaler()),
                ("logistic", LogisticRegression(class_weight="balanced", max_iter=1000,
                    random_state=RANDOM_SEED, solver="liblinear")),
            ])
            self.model.fit(np.vstack(rows), np.asarray(targets))
            self.training_mode = "logistic_regression"
        elif rows:
            self.constant_probability = float(np.mean(targets))
            self.training_mode = "constant_label"
        # Serving uses all history available at the outer training cutoff.
        serving_hybrid = HybridRecommender().fit(self.contents, self.users, self.interactions)
        self._set_snapshot(serving_hybrid, self.interactions, self.interactions["timestamp"].max())

    def _set_snapshot(self, hybrid: HybridRecommender, history: pd.DataFrame, reference: pd.Timestamp) -> None:
        self.hybrid = hybrid
        self._feature_cache: dict[str, np.ndarray] = {}
        self.freshness = np.exp(-np.log(2) * np.maximum(
            (reference - self.contents["published_at"]).dt.total_seconds().to_numpy() / 86400, 0
        ) / 30.0)
        self.activity = history.groupby("user_id").size().to_dict()

    def candidate_ids(self, user_id: str, exclude_seen: bool = True) -> list[str]:
        return self.hybrid.candidate_ids(user_id, exclude_seen=exclude_seen)

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
        if user_id in self._feature_cache:
            return self._feature_cache[user_id]
        raw = self.hybrid.score_components(user_id)
        components = {name: np.array([values[self.hybrid.content_index[value]]
            if value in self.hybrid.content_index else 0.0 for value in self.content_ids])
            for name, values in raw.items()}
        topic_overlap, coin_overlap, difficulty = self._overlap_features(user_id)
        activity = np.full(len(self.contents), np.log1p(self.activity.get(user_id, 0)))
        features = np.column_stack(
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

        features = np.round(features, 12)
        self._feature_cache[user_id] = features
        return features

    def score_all(self, user_id: str, exclude_seen: bool = True) -> np.ndarray:
        candidates = self.candidate_ids(user_id, exclude_seen=exclude_seen)
        indices = np.array([self.content_index[value] for value in candidates], dtype=int)
        scores = np.full(len(self.contents), -np.inf)
        if not len(indices):
            return scores
        hybrid_scores = minmax(self.hybrid.score_all(user_id))[indices]
        if self.model is not None:
            features = self.model.named_steps["scale"].transform(self._features(user_id)[indices])
            logistic = self.model.named_steps["logistic"]
            logits = np.einsum("ij,j->i", features, logistic.coef_[0]) + logistic.intercept_[0]
            probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
            scores[indices] = 0.65 * probabilities + 0.35 * hybrid_scores
        elif self.constant_probability is not None:
            scores[indices] = 0.65 * self.constant_probability + 0.35 * hybrid_scores
        else:
            scores[indices] = hybrid_scores
        return scores

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        validate_k(k)
        return top_results(self.contents, self.score_all(user_id, exclude_seen), k, self.name)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            return pd.DataFrame(columns=["feature", "coefficient"])
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
        if candidate_multiplier <= 0 or max_per_source <= 0 or max_per_primary_topic <= 0:
            raise ValueError("Candidate multiplier and diversity caps must be positive")
        if not 0 <= relevance_weight <= 1:
            raise ValueError("relevance_weight must be between 0 and 1")
        self.candidate_multiplier = candidate_multiplier
        self.relevance_weight = relevance_weight
        self.max_per_source = max_per_source
        self.max_per_primary_topic = max_per_primary_topic

    def _fit(self) -> None:
        self.ranker = RankingRecommender().fit(self.contents, self.users, self.interactions)

    def score_all(self, user_id: str) -> np.ndarray:
        return self.ranker.score_all(user_id)

    def candidate_ids(self, user_id: str, exclude_seen: bool = True) -> list[str]:
        return self.ranker.candidate_ids(user_id, exclude_seen=exclude_seen)

    def recommend(self, user_id: str, k: int = 10, exclude_seen: bool = True) -> pd.DataFrame:
        validate_k(k)
        scores = self.ranker.score_all(user_id, exclude_seen=exclude_seen).copy()
        if exclude_seen:
            for content_id in self.seen_by_user.get(user_id, set()):
                if content_id in self.content_index:
                    scores[self.content_index[content_id]] = -np.inf
        candidate_count = min(k * self.candidate_multiplier, int(np.isfinite(scores).sum()))
        candidates = list(ordered_indices(scores, self.content_ids)[:candidate_count])
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
                if value > best_value + 1e-12:
                    best_value = value
                    best_index = index
            if best_index is None:
                break
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
