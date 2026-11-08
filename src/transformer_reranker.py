"""CPU Transformer cross-encoder over point-in-time user features and content text."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.neural import NeuralConfig, NeuralSystem, TwoTowerRecommender, prepare_examples, seed_cpu, train_network
from src.recommenders import ordered_indices, top_results, validate_k


@dataclass(frozen=True)
class TransformerConfig:
    seed: int = 42
    epochs: int = 16
    batch_size: int = 32
    learning_rate: float = 0.001
    vocabulary_size: int = 2048
    max_tokens: int = 32
    width: int = 32
    heads: int = 4
    layers: int = 2
    inference_batch_size: int = 50

    def __post_init__(self):
        for name in ('epochs', 'batch_size', 'vocabulary_size', 'max_tokens', 'width', 'heads', 'layers', 'inference_batch_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.vocabulary_size < 2 or self.width % self.heads:
            raise ValueError('Vocabulary must include padding and width must be divisible by heads')
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('learning_rate must be positive and finite')


def tokenize(text, config):
    # Hashing is stateless, with zero reserved for padding and an explicit length bound.
    words = re.findall(r'\w+|[^\w\s]', str(text).lower(), flags=re.UNICODE)[:config.max_tokens]
    values = [1 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], 'little')
              % (config.vocabulary_size - 1) for word in words]
    return values + [0] * (config.max_tokens - len(values))


class ContentCrossEncoder(nn.Module):
    def __init__(self, user_dim, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocabulary_size, config.width, padding_idx=0)
        self.user = nn.Linear(user_dim, config.width)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.randn(1, config.max_tokens + 2, config.width) * 0.02)
        # Construct each layer separately so their initial weights differ.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(config.width, config.heads,
            dim_feedforward=2 * config.width, dropout=0, activation='gelu', batch_first=True,
            norm_first=True) for _ in range(config.layers)])
        self.output = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1))

    def forward(self, users, tokens):
        prefix = torch.cat([self.cls.expand(len(users), -1, -1), self.user(users)[:, None]], dim=1)
        values = torch.cat([prefix, self.embedding(tokens)], dim=1) + self.position
        mask = torch.cat([torch.zeros((len(users), 2), dtype=torch.bool, device=tokens.device), tokens.eq(0)], dim=1)
        for layer in self.layers:
            values = layer(values, src_key_padding_mask=mask)
        return self.output(values[:, 0]).squeeze(-1)


class TransformerReranker(TwoTowerRecommender):
    name = 'v9_transformer_content_reranker'

    def __init__(self, system, config=TransformerConfig()):
        super().__init__(system)
        self.config = config
        self.tokens = torch.tensor([tokenize(row.title + ' ' + row.summary, config)
                                   for row in system.contents.itertuples()], dtype=torch.long)
        self.network = ContentCrossEncoder(system.encoder.user_dim, config)
        self.history, self.training_summary = [], {}
        self.training_audit = None
        self.fitted = False

    def fit_transformer(self):
        seed_cpu(self.config.seed)
        self.network = ContentCrossEncoder(self.system.encoder.user_dim, self.config)
        examples, _ = prepare_examples(self.system.encoder, self.system.interactions, self.system.config)
        selected = []
        with torch.inference_mode():
            embeddings = self.system.tower.users(torch.from_numpy(examples.user_features))
            for index in np.concatenate([examples.indices(2), examples.indices(3)]):
                eligible = examples.eligible[index]
                scores = (self.system.item_embeddings[eligible] * embeddings[index]).sum(1).numpy()
                candidates = eligible[ordered_indices(scores, [self.content_ids[i] for i in eligible])
                                      [:self.system.config.candidate_count]]
                if examples.item_indices[index] in candidates:
                    selected.append(index)
        selected = np.asarray(selected, dtype=np.int64)
        config = NeuralConfig(seed=self.config.seed, epochs=self.config.epochs,
                              batch_size=self.config.batch_size, learning_rate=self.config.learning_rate)
        users = torch.from_numpy(examples.user_features[selected])
        tokens = self.tokens[examples.item_indices[selected]]
        labels = torch.from_numpy(examples.labels[selected])
        self.history, self.training_summary = train_network(self.network, (users, tokens), labels,
            np.flatnonzero(examples.windows[selected] == 2), np.flatnonzero(examples.windows[selected] == 3),
            config, 'transformer_reranker')
        self.training_audit = {'indices': selected.tolist(), 'windows': examples.windows[selected].tolist(),
            'timestamps_ns': examples.timestamps[selected].tolist(), 'label_ends_ns': examples.label_ends[selected].tolist(),
            'boundaries': [value.isoformat() for value in examples.boundaries],
            'example_indices_sha256': hashlib.sha256(selected.tobytes()).hexdigest()}
        self.fitted = True
        return self

    def score_all(self, user_id):
        return self.rank_scores(user_id)

    def rank_scores(self, user_id, exclude_seen=True):
        if not self.fitted:
            raise RuntimeError('Transformer must be fitted before inference')
        candidates = self.system.candidate_indices(user_id, exclude_seen)
        scores = np.full(len(self.contents), -np.inf)
        user = torch.from_numpy(self.system.user_vector(user_id))[None]
        with torch.inference_mode():
            for batch in torch.as_tensor(candidates).split(self.config.inference_batch_size):
                scores[batch.numpy()] = torch.sigmoid(self.network(user.expand(len(batch), -1), self.tokens[batch])).numpy()
        return scores

    def recommend(self, user_id, k=10, exclude_seen=True):
        validate_k(k)
        return top_results(self.contents, self.rank_scores(user_id, exclude_seen), k, self.name)

    def save(self, path, base_checkpoint):
        if not self.fitted:
            raise RuntimeError('Cannot save an unfitted Transformer')
        torch.save({'format_version': 1, 'config': asdict(self.config), 'state': self.network.state_dict(),
                    'base_sha256': hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest(),
                    'history': self.history, 'training_summary': self.training_summary,
                    'training_audit': self.training_audit}, path)

    @classmethod
    def load(cls, path, base_checkpoint):
        digest = hashlib.sha256(Path(base_checkpoint).read_bytes()).hexdigest()
        payload = cls._read_checkpoint(path, digest)
        return cls._restore(payload, NeuralSystem.load(base_checkpoint))

    @classmethod
    def from_checkpoint(cls, path, system, base_sha256):
        """Attach a checkpoint to the base system already loaded by the host."""
        return cls._restore(cls._read_checkpoint(path, base_sha256), system)

    @staticmethod
    def _read_checkpoint(path, base_sha256):
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['format_version'] != 1:
            raise ValueError('Unsupported Transformer checkpoint format')
        if payload['base_sha256'] != base_sha256:
            raise ValueError('Frozen retrieval checkpoint differs')
        return payload

    @classmethod
    def _restore(cls, payload, system):
        result = cls(system, TransformerConfig(**payload['config']))
        result.network.load_state_dict(payload['state'])
        result.network.eval()
        result.history, result.training_summary = payload['history'], payload['training_summary']
        result.training_audit, result.fitted = payload['training_audit'], True
        return result
