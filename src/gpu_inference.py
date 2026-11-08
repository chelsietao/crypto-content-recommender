"""Device-resident batched inference using the trained neural checkpoint."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from src.config import TOPICS, COINS
from src.recommenders import validate_k
from src.neural import NeuralSystem, ranking_features, TwoTowerRecommender, NeuralRankingRecommender

MODEL_NAMES = (TwoTowerRecommender.name, NeuralRankingRecommender.name)


@dataclass
class PreparedBatch:
    users: torch.Tensor
    seen: torch.Tensor

    def row(self, index):
        return PreparedBatch(self.users[index:index + 1], self.seen[index:index + 1])


class BatchedNeuralInference:
    """Keep item embeddings on the selected device and vectorize requests."""
    def __init__(self, system: NeuralSystem, device: str = 'cpu', dtype=torch.float32):
        if device not in ('cpu', 'mps', 'cuda'):
            raise ValueError('Supported devices are cpu, mps and cuda')
        if dtype not in (torch.float32, torch.float16):
            raise ValueError('Supported dtypes are float32 and float16')
        self.device, self.dtype, self.system = torch.device(device), dtype, system
        self.tower = copy.deepcopy(system.tower).to(device=self.device, dtype=dtype).eval()
        self.ranker = copy.deepcopy(system.ranker).to(device=self.device, dtype=dtype).eval()
        self.tower.requires_grad_(False)
        self.ranker.requires_grad_(False)
        # ID ordering provides the secondary key for stable score sorting.
        self.order = np.argsort(np.array(system.encoder.content_ids), kind='stable')
        self.content_ids = [system.encoder.content_ids[i] for i in self.order]
        tensor = lambda values: torch.as_tensor(values, device=self.device, dtype=dtype)
        with torch.inference_mode():
            self.items = self.tower.items(tensor(system.encoder.item_features[self.order]))
        self.metadata = tensor(system.encoder.item_metadata[self.order])
        self.quality = tensor(system.contents.quality_score.to_numpy()[self.order])
        age = np.maximum((system.reference.value - system.encoder.published_ns[self.order]) / (86400 * 1e9), 0)
        self.freshness = tensor(np.exp(-np.log(2) * age / 30))
        self.candidate_count = min(system.config.candidate_count, len(self.order))

    def prepare(self, user_ids: list[str], exclude_seen=True) -> PreparedBatch:
        if not user_ids:
            raise ValueError('A batch must contain at least one user')
        features = np.stack([self.system.user_vector(user) for user in user_ids])
        mask = np.zeros((len(user_ids), len(self.order)), dtype=bool)
        if exclude_seen:
            for row, user in enumerate(user_ids):
                mask[row] = np.isin(self.order, list(self.system.seen.get(user, set())))
        return PreparedBatch(torch.as_tensor(features, device=self.device, dtype=self.dtype),
                             torch.as_tensor(mask, device=self.device))

    def encode(self, batch: PreparedBatch):
        return self.tower.users(batch.users)

    def retrieve(self, batch: PreparedBatch, users):
        scores = (users @ self.items.T).masked_fill(batch.seen, -torch.inf)
        candidates = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :self.candidate_count]
        return candidates, scores.gather(1, candidates)

    def rank(self, batch: PreparedBatch, users, candidates, retrieval_scores):
        items = self.items[candidates]
        explicit = batch.users[:, :self.system.encoder.preference_dim]
        topic_end, coin_end = len(TOPICS), len(TOPICS) + len(COINS)
        topics = (explicit[:, :topic_end] @ self.metadata[:, :topic_end].T) / explicit[:, :topic_end].sum(1, keepdim=True).clamp_min(1)
        coins = (explicit[:, topic_end:coin_end] @ self.metadata[:, topic_end:coin_end].T) / explicit[:, topic_end:coin_end].sum(1, keepdim=True).clamp_min(1)
        difficulty = explicit[:, coin_end:] @ self.metadata[:, coin_end:].T
        shape = topics.shape
        extras = torch.stack([topics, coins, difficulty, self.quality.expand(shape),
                              self.freshness.expand(shape), batch.users[:, -1:].expand(shape)], dim=-1)
        extras = extras.gather(1, candidates[:, :, None].expand(-1, -1, 6))
        user_vectors = users[:, None, :].expand_as(items)
        scores = torch.sigmoid(self.ranker(ranking_features(user_vectors, items, extras)))
        return scores.masked_fill(~torch.isfinite(retrieval_scores), -torch.inf)

    def finish(self, candidates, scores, k, presorted=False, optimized=True):
        if presorted and optimized:
            return candidates[:, :k], scores[:, :k]
        if optimized:
            # Scatter into catalog ID order so only one stable score sort is needed.
            dense = scores.new_full((len(scores), len(self.content_ids)), -torch.inf)
            dense.scatter_(1, candidates, scores)
            selected = torch.argsort(dense, dim=1, descending=True, stable=True)[:, :k]
            return selected, dense.gather(1, selected)
        # Reference path retains the two sorts used before profiling.
        id_order = torch.argsort(candidates, dim=1, stable=True)
        candidates = candidates.gather(1, id_order)
        scores = scores.gather(1, id_order)
        selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
        return candidates.gather(1, selected), scores.gather(1, selected)

    @torch.inference_mode()
    def forward(self, batch: PreparedBatch, model: str, k: int = 10, optimized=True):
        if model not in MODEL_NAMES:
            raise ValueError('Unknown neural model')
        validate_k(k)
        k = min(k, self.candidate_count)
        users = self.encode(batch)
        candidates, scores = self.retrieve(batch, users)
        if model == MODEL_NAMES[1]:
            scores = self.rank(batch, users, candidates, scores)
        return self.finish(candidates, scores, k, presorted=model == MODEL_NAMES[0], optimized=optimized)

    def decode(self, indices, scores):
        indices, scores = indices.cpu().numpy(), scores.float().cpu().numpy()
        if indices.ndim != 2 or scores.ndim != 2 or indices.shape != scores.shape:
            raise ValueError('Decoded indices and scores must have the same two-dimensional shape')
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError('Decoded content indices must be integers')
        if (indices < 0).any() or (indices >= len(self.content_ids)).any():
            raise ValueError('Decoded content index is outside the catalog')
        if np.isnan(scores).any() or np.isposinf(scores).any():
            raise ValueError('Decoded scores must be finite or negative-infinity padding')
        return [[(self.content_ids[int(item)], float(score)) for item, score in zip(ids, values)
                 if np.isfinite(score)] for ids, values in zip(indices, scores)]
