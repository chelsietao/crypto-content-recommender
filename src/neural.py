"""Point-in-time two-tower retrieval, a frozen-tower neural ranker, and checkpoints."""
from __future__ import annotations

import copy
import io
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn
from torch.nn import functional as F

from src.config import COINS, CONTENT_TYPES, DIFFICULTIES, EVENT_WEIGHTS, TOPICS
from src.data_pipeline import DatasetBundle, validate_bundle
from src.recommenders import BaseRecommender, ordered_indices, split_tags, top_results, validate_k


@dataclass(frozen=True)
class NeuralConfig:
    seed: int = 42
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 0.003
    weight_decay: float = 0.001
    hidden_dim: int = 64
    embedding_dim: int = 32
    text_dim: int = 64
    candidate_count: int = 50
    temperature: float = 0.2
    outcome_minutes: int = 15

    def __post_init__(self):
        for name in ("epochs", "batch_size", "hidden_dim", "embedding_dim", "text_dim", "candidate_count", "outcome_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


def seed_cpu(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class FeatureEncoder:
    """Fixed feature definitions and stateless text hashing, with no corpus fitting."""
    def __init__(self, contents: pd.DataFrame, users: pd.DataFrame, text_dim: int = 64):
        self.contents = contents.reset_index(drop=True).copy()
        self.users = users.reset_index(drop=True).copy()
        self.content_ids = self.contents.content_id.tolist()
        self.content_index = {value: index for index, value in enumerate(self.content_ids)}
        self.user_rows = {row.user_id: row for row in self.users.itertuples()}
        self.categories = [*TOPICS, *COINS, *DIFFICULTIES]
        self.category_index = {name: index for index, name in enumerate(self.categories)}
        self.preference_dim = len(self.categories)
        self.user_dim = 3 * self.preference_dim + 1
        self.item_metadata = np.stack([
            self.tags(row.topic_tags, row.coin_tags, row.difficulty)
            for row in self.contents.itertuples()
        ])
        text = (self.contents.title + " " + self.contents.summary).fillna("")
        hashed = HashingVectorizer(n_features=text_dim, alternate_sign=False, norm="l2").transform(text)
        types = np.array([[float(value == category) for category in CONTENT_TYPES]
                          for value in self.contents.content_type], dtype=np.float32)
        self.item_features = np.column_stack([
            hashed.toarray(), self.item_metadata, types, self.contents.quality_score.to_numpy()
        ]).astype(np.float32)
        self.published_ns = self.contents.published_at.astype("int64").to_numpy()

    def tags(self, topics, coins, difficulty) -> np.ndarray:
        values = np.zeros(self.preference_dim, dtype=np.float32)
        for name in split_tags(topics) | split_tags(coins) | {difficulty}:
            if name in self.category_index:
                values[self.category_index[name]] = 1
        return values

    def user_features(self, user_id: str, state: "UserHistory") -> np.ndarray:
        row = self.user_rows.get(user_id)
        explicit = (self.tags(row.preferred_topics, row.preferred_coins, row.experience_level)
                    if row is not None else np.zeros(self.preference_dim, dtype=np.float32))
        return np.concatenate([
            explicit, state.positive / max(state.positive_weight, 1),
            state.negative / max(state.negative_weight, 1), [np.log1p(state.activity) / 10]
        ]).astype(np.float32)


class UserHistory:
    def __init__(self, width: int):
        self.positive = np.zeros(width, dtype=np.float64)
        self.negative = np.zeros(width, dtype=np.float64)
        self.positive_weight = 0.0
        self.negative_weight = 0.0
        self.activity = 0
        self.preferences: dict[int, float] = {}
        self.seen: set[int] = set()

    def update(self, item: int, weight: float, metadata: np.ndarray):
        previous = self.preferences.get(item, 0.0)
        current = min(previous, weight) if previous < 0 or weight < 0 else max(previous, weight)
        pos_delta = max(current, 0) - max(previous, 0)
        neg_delta = max(-current, 0) - max(-previous, 0)
        self.positive += pos_delta * metadata
        self.negative += neg_delta * metadata
        self.positive_weight += pos_delta
        self.negative_weight += neg_delta
        self.preferences[item] = current
        self.seen.add(item)
        self.activity += 1


@dataclass
class ImpressionExamples:
    user_features: np.ndarray
    item_indices: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    windows: np.ndarray
    eligible: list[np.ndarray]
    boundaries: list[pd.Timestamp]
    user_ids: list[str]
    label_ends: np.ndarray

    def indices(self, window: int) -> np.ndarray:
        return np.flatnonzero(self.windows == window)


def prepare_examples(encoder: FeatureEncoder, interactions: pd.DataFrame, config: NeuralConfig):
    """Snapshot strictly earlier events; labels use a completed, bounded outcome window."""
    times = interactions.timestamp.sort_values().drop_duplicates().reset_index(drop=True)
    if len(times) < 8:
        raise ValueError("Neural training requires at least eight distinct timestamps")
    boundaries = [times.iloc[int((len(times) - 1) * fraction)] for fraction in (0.5, 0.65, 0.85, 1.0)]
    if len(set(boundaries)) != 4:
        raise ValueError("Neural training needs four nonempty chronological windows")
    boundary_ns = np.array([value.value for value in boundaries])
    horizon = pd.Timedelta(minutes=config.outcome_minutes)
    groups = {key: frame for key, frame in interactions.groupby(["user_id", "content_id"])}
    histories = {user_id: UserHistory(encoder.preference_dim) for user_id in encoder.user_rows}
    records = []
    # All events sharing a timestamp are processed after the requests at that timestamp.
    for timestamp, frame in interactions.sort_values(["timestamp", "event_id"], kind="stable").groupby("timestamp", sort=True):
        window = int(np.searchsorted(boundary_ns, timestamp.value, side="left"))
        for row in frame.loc[frame.event_type == "impression"].drop_duplicates(["user_id", "content_id"]).itertuples():
            state = histories[row.user_id]
            item = encoder.content_index[row.content_id]
            end = timestamp + horizon
            if item in state.seen or end > boundaries[window]:
                continue
            outcomes = groups[(row.user_id, row.content_id)]
            outcomes = outcomes.loc[(outcomes.timestamp >= timestamp) & (outcomes.timestamp <= end)]
            weights = outcomes.event_type.map(EVENT_WEIGHTS)
            label = float((weights > 0).any() and not (weights < 0).any())
            eligible = np.flatnonzero(encoder.published_ns <= timestamp.value)
            eligible = np.array([index for index in eligible if index not in state.seen], dtype=np.int64)
            records.append((encoder.user_features(row.user_id, state), item, label, timestamp.value,
                            window, eligible, row.user_id, end.value))
        for row in frame.itertuples():
            item = encoder.content_index[row.content_id]
            histories[row.user_id].update(item, EVENT_WEIGHTS[row.event_type], encoder.item_metadata[item])
    if not records:
        raise ValueError("No complete impression outcomes available for neural training")
    features, items, labels, stamps, windows, eligible, user_ids, label_ends = zip(*records)
    examples = ImpressionExamples(np.stack(features), np.array(items), np.array(labels, dtype=np.float32),
        np.array(stamps), np.array(windows), list(eligible), boundaries, list(user_ids), np.array(label_ends))
    return examples, histories


class TwoTower(nn.Module):
    def __init__(self, user_dim: int, item_dim: int, config: NeuralConfig):
        super().__init__()
        def tower(width):
            return nn.Sequential(nn.Linear(width, config.hidden_dim), nn.ReLU(),
                                 nn.Linear(config.hidden_dim, config.embedding_dim))
        self.user_tower = tower(user_dim)
        self.item_tower = tower(item_dim)
        self.bias = nn.Parameter(torch.zeros(()))
        self.temperature = config.temperature

    def users(self, values):
        return F.normalize(self.user_tower(values), dim=-1)

    def items(self, values):
        return F.normalize(self.item_tower(values), dim=-1)

    def forward(self, users, items):
        return (self.users(users) * self.items(items)).sum(-1) / self.temperature + self.bias


class NeuralRanker(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(embedding_dim * 4 + 6, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, features):
        return self.layers(features).squeeze(-1)


def ranking_features(users: torch.Tensor, items: torch.Tensor, extras: torch.Tensor):
    return torch.cat([users, items, users * items, torch.abs(users - items), extras], dim=-1)


def train_network(model, inputs: tuple[torch.Tensor, ...], labels: torch.Tensor,
                  train_indices: np.ndarray, validation_indices: np.ndarray,
                  config: NeuralConfig, stage: str):
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"{stage}: training and validation windows must contain examples")
    target = labels[train_indices]
    positives = float(target.sum())
    if not 0 < positives < len(target):
        raise ValueError(f"{stage}: training needs both positive and negative impression labels")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(target) - positives) / positives))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    initial = copy.deepcopy(model.state_dict())
    def loss_on(indices):
        with torch.no_grad():
            return float(criterion(model(*(value[indices] for value in inputs)), labels[indices]))
    for epoch in range(config.epochs + 1):
        if epoch:
            model.train()
            order = torch.as_tensor(train_indices)[torch.randperm(len(train_indices), generator=generator)]
            for selected in order.split(config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(*(value[selected] for value in inputs)), labels[selected])
                if not torch.isfinite(loss):
                    raise RuntimeError(f"{stage}: nonfinite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        model.eval()
        train_loss, validation_loss = loss_on(train_indices), loss_on(validation_indices)
        if not np.isfinite([train_loss, validation_loss]).all():
            raise RuntimeError(f"{stage}: nonfinite evaluation loss")
        history.append({"stage": stage, "epoch": epoch, "train_loss": train_loss,
                        "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    changed = any(not torch.equal(initial[key], best_state[key]) for key in initial)
    return history, {"selected_epoch": best_epoch, "validation_loss": best_loss,
                     "parameters_changed": changed, "training_examples": len(train_indices),
                     "validation_examples": len(validation_indices)}


class NeuralSystem:
    """One trained retrieval model and ranker, shared by evaluation wrappers."""
    def __init__(self, contents, users, interactions, config: NeuralConfig):
        validate_bundle(DatasetBundle(contents, users, interactions))
        if interactions.empty:
            raise ValueError("Neural training requires interaction history")
        reference = interactions.timestamp.max()
        if (contents.published_at > reference).any():
            raise ValueError("Neural catalog contains items published after the training cutoff")
        self.contents = contents.reset_index(drop=True).copy()
        self.users = users.loc[users.created_at <= reference].reset_index(drop=True).copy()
        self.interactions = interactions.copy()
        self.config = config
        self.encoder = FeatureEncoder(self.contents, self.users, config.text_dim)
        self.reference = interactions.timestamp.max()
        self.history = []
        self.training_summary = {}

    def extras(self, user_values: np.ndarray, item_indices: np.ndarray, timestamp_ns: int):
        meta = self.encoder.item_metadata[item_indices]
        explicit = user_values[:self.encoder.preference_dim]
        topic_end, coin_end = len(TOPICS), len(TOPICS) + len(COINS)
        topic_overlap = (meta[:, :topic_end] * explicit[:topic_end]).sum(1) / max(explicit[:topic_end].sum(), 1)
        coin_overlap = (meta[:, topic_end:coin_end] * explicit[topic_end:coin_end]).sum(1) / max(explicit[topic_end:coin_end].sum(), 1)
        difficulty = (meta[:, coin_end:] * explicit[coin_end:]).sum(1)
        age = np.maximum((timestamp_ns - self.encoder.published_ns[item_indices]) / (86400 * 1e9), 0)
        return np.column_stack([topic_overlap, coin_overlap, difficulty,
            self.contents.quality_score.to_numpy()[item_indices], np.exp(-np.log(2) * age / 30),
            np.full(len(item_indices), user_values[-1])]).astype(np.float32)

    def fit(self, trainer=None):
        fit_network = train_network if trainer is None else trainer
        seed_cpu(self.config.seed)
        examples, self.histories = prepare_examples(self.encoder, self.interactions, self.config)
        self.examples = examples
        item_tensor = torch.from_numpy(self.encoder.item_features)
        user_tensor = torch.from_numpy(examples.user_features)
        labels = torch.from_numpy(examples.labels)
        self.tower = TwoTower(self.encoder.user_dim, item_tensor.shape[1], self.config)
        curve, summary = fit_network(self.tower, (user_tensor, item_tensor[examples.item_indices]), labels,
            examples.indices(0), examples.indices(1), self.config, "two_tower")
        self.history.extend(curve)
        self.training_summary["two_tower"] = summary
        # Tower weights are frozen before either ranker label window starts.
        for parameter in self.tower.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            all_users = self.tower.users(user_tensor)
            all_items = self.tower.items(item_tensor)
        self.item_embeddings = all_items
        selected = []
        rank_features = []
        for index in np.concatenate([examples.indices(2), examples.indices(3)]):
            eligible = examples.eligible[index]
            scores = (all_items[eligible] * all_users[index]).sum(1).numpy()
            ids = [self.encoder.content_ids[value] for value in eligible]
            candidates = eligible[ordered_indices(scores, ids)[:self.config.candidate_count]]
            item = examples.item_indices[index]
            if item not in candidates:
                continue
            extras = torch.from_numpy(self.extras(examples.user_features[index], np.array([item]), examples.timestamps[index]))
            rank_features.append(ranking_features(all_users[index:index+1], all_items[item:item+1], extras)[0])
            selected.append(index)
        if not selected:
            raise ValueError("No exposed recall candidates available for ranker training")
        selected = np.array(selected)
        rank_inputs = torch.stack(rank_features)
        self.ranker = NeuralRanker(self.config.embedding_dim, self.config.hidden_dim)
        curve, summary = fit_network(self.ranker, (rank_inputs,), labels[selected],
            np.flatnonzero(examples.windows[selected] == 2), np.flatnonzero(examples.windows[selected] == 3),
            self.config, "neural_ranker")
        self.history.extend(curve)
        self.training_summary["neural_ranker"] = summary
        self.training_summary["window_boundaries"] = [time.isoformat() for time in examples.boundaries]
        self.training_summary["window_examples"] = [len(examples.indices(window)) for window in range(4)]
        self.ranker_example_indices = selected
        self.training_summary["ranker_candidate_examples"] = len(selected)
        self.serving_users = {value: self.encoder.user_features(value, state) for value, state in self.histories.items()}
        self.seen = {value: set(state.seen) for value, state in self.histories.items()}
        return self

    def user_vector(self, user_id):
        if user_id in self.serving_users:
            return self.serving_users[user_id]
        return self.encoder.user_features(user_id, UserHistory(self.encoder.preference_dim))

    def retrieval_scores(self, user_id):
        with torch.no_grad():
            user = self.tower.users(torch.from_numpy(self.user_vector(user_id))[None])
            return (self.item_embeddings * user).sum(1).numpy().copy()

    def select_candidates(self, user_id, scores, exclude_seen=True):
        scores = scores.copy()
        if exclude_seen:
            scores[list(self.seen.get(user_id, set()))] = -np.inf
        return ordered_indices(scores, self.encoder.content_ids)[:self.config.candidate_count]

    def candidate_indices(self, user_id, exclude_seen=True):
        return self.select_candidates(user_id, self.retrieval_scores(user_id), exclude_seen)

    def rank_scores(self, user_id, exclude_seen=True):
        candidates = self.candidate_indices(user_id, exclude_seen)
        scores = np.full(len(self.contents), -np.inf)
        if not len(candidates):
            return scores
        values = self.user_vector(user_id)
        with torch.no_grad():
            embedding = self.tower.users(torch.from_numpy(values)[None]).expand(len(candidates), -1)
            extras = torch.from_numpy(self.extras(values, candidates, self.reference.value))
            inputs = ranking_features(embedding, self.item_embeddings[candidates], extras)
            scores[candidates] = torch.sigmoid(self.ranker(inputs)).numpy()
        return scores

    def recommenders(self):
        return {cls.name: cls(self) for cls in (TwoTowerRecommender, NeuralRankingRecommender)}

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format_version": 1, "config": asdict(self.config),
            "tower_state": self.tower.state_dict(), "ranker_state": self.ranker.state_dict(),
            "contents_json": self.contents.to_json(orient="table", date_format="iso"),
            "users_json": self.users.to_json(orient="table", date_format="iso"),
            "interactions_json": self.interactions.to_json(orient="table", date_format="iso"),
            "serving_users": {key: torch.from_numpy(value) for key, value in self.serving_users.items()},
            "seen": {key: sorted(value) for key, value in self.seen.items()},
            "history": self.history, "training_summary": self.training_summary}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("Unsupported neural checkpoint format")
        config = NeuralConfig(**payload["config"])
        read = lambda key: pd.read_json(io.StringIO(payload[key]), orient="table")
        system = cls(read("contents_json"), read("users_json"), read("interactions_json"), config)
        system.tower = TwoTower(system.encoder.user_dim, system.encoder.item_features.shape[1], config)
        system.tower.load_state_dict(payload["tower_state"])
        system.tower.eval()
        system.tower.requires_grad_(False)
        system.ranker = NeuralRanker(config.embedding_dim, config.hidden_dim)
        system.ranker.load_state_dict(payload["ranker_state"])
        system.ranker.eval()
        with torch.no_grad():
            system.item_embeddings = system.tower.items(torch.from_numpy(system.encoder.item_features))
        system.serving_users = {key: value.numpy() for key, value in payload["serving_users"].items()}
        system.seen = {key: set(value) for key, value in payload["seen"].items()}
        system.history = payload["history"]
        system.training_summary = payload["training_summary"]
        return system


class TwoTowerRecommender(BaseRecommender):
    name = "v7_pytorch_two_tower"
    def __init__(self, system: NeuralSystem):
        super().__init__()
        self.system = system
        self.fit(system.contents, system.users, system.interactions)

    def score_all(self, user_id):
        return self.system.retrieval_scores(user_id)

    def candidate_ids(self, user_id, exclude_seen=True):
        return [self.content_ids[index] for index in self.system.candidate_indices(user_id, exclude_seen)]

    def recommend(self, user_id, k=10, exclude_seen=True):
        validate_k(k)
        scores = self.system.retrieval_scores(user_id)
        candidates = self.system.select_candidates(user_id, scores, exclude_seen)
        masked = np.full_like(scores, -np.inf)
        masked[candidates] = scores[candidates]
        scores = masked
        return top_results(self.contents, scores, k, self.name)


class NeuralRankingRecommender(TwoTowerRecommender):
    name = "v8_pytorch_neural_ranker"
    def score_all(self, user_id):
        return self.system.rank_scores(user_id)

    def recommend(self, user_id, k=10, exclude_seen=True):
        validate_k(k)
        return top_results(self.contents, self.system.rank_scores(user_id, exclude_seen), k, self.name)
