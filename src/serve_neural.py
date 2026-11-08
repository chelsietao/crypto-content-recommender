"""Standalone CPU HTTP inference for the exported two-tower and neural ranker."""
from __future__ import annotations

import argparse
import math
import os
import threading
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

import torch
from anyio.to_thread import current_default_thread_limiter
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.config import ARTIFACTS_DIR, REPORTS_DIR
from src.checkpoint_io import verified_checkpoint_hash
from src.neural import NeuralSystem
from src.transformer_reranker import TransformerReranker

MODEL_NAMES = ("v7_pytorch_two_tower", "v8_pytorch_neural_ranker", "v9_transformer_content_reranker")
ModelName = Literal["v7_pytorch_two_tower", "v8_pytorch_neural_ranker", "v9_transformer_content_reranker"]
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "neural" / "neural_recommender.pt"
DEFAULT_TRANSFORMER = REPORTS_DIR / 'transformer_reranking' / 'transformer.pt'


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    model: ModelName = MODEL_NAMES[0]
    k: Annotated[int, Field(ge=1, le=50)] = 10
    exclude_seen: bool = True


class RecommendationItem(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    content_id: str
    title: str
    topic_tags: str
    score: Annotated[float, Field(allow_inf_nan=False)]


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: str
    model: ModelName
    checkpoint_sha256: str
    reranker_checkpoint_sha256: Optional[str] = None
    training_cutoff: str
    items: list[RecommendationItem]


class InferenceGate:
    """Bound model admission and lock wait without occupying every worker token."""
    def __init__(self, capacity: int, queue_timeout: float):
        self.admission = threading.BoundedSemaphore(capacity)
        self.execution = threading.Lock()
        self.queue_timeout = queue_timeout

    @contextmanager
    def slot(self):
        if not self.admission.acquire(blocking=False):
            raise HTTPException(status_code=429, detail="Inference capacity is full",
                                headers={"Retry-After": "1"})
        acquired = False
        try:
            acquired = self.execution.acquire(timeout=self.queue_timeout)
            if not acquired:
                raise HTTPException(status_code=503, detail="Inference queue wait timed out",
                                    headers={"Retry-After": "1"})
            yield
        finally:
            if acquired:
                self.execution.release()
            self.admission.release()


def create_app(checkpoint: Path = DEFAULT_CHECKPOINT, torch_threads: int = 1,
               max_inflight: int = 32, queue_timeout: float = 1.0,
               transformer_checkpoint: Path | None = DEFAULT_TRANSFORMER) -> FastAPI:
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or not 1 <= max_inflight <= 32:
        raise ValueError("max_inflight must be an integer between 1 and 32")
    if (isinstance(queue_timeout, bool) or not isinstance(queue_timeout, (int, float))
            or not math.isfinite(queue_timeout) or queue_timeout <= 0):
        raise ValueError("queue_timeout must be positive and finite")
    checkpoint = Path(checkpoint).resolve()

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(torch_threads)
        current_default_thread_limiter().total_tokens = 40
        # Verify the adjacent manifest before loading weights into the process.
        actual = verified_checkpoint_hash(checkpoint)
        system = NeuralSystem.load(checkpoint)
        models = system.recommenders()
        reranker_hash = None
        if transformer_checkpoint is not None:
            reranker_hash = verified_checkpoint_hash(transformer_checkpoint)
            transformer = TransformerReranker.from_checkpoint(transformer_checkpoint, system, actual)
            models[transformer.name] = transformer
        for model in models.values():
            model.recommend("new-user", k=10)
        app.state.runtime = (system, models, InferenceGate(max_inflight, queue_timeout))
        app.state.metadata = {
            "status": "ready", "pid": os.getpid(), "device": "cpu",
            "torch_threads": torch.get_num_threads(), "workers": 1,
            "inference_slots": 1, "checkpoint_sha256": actual,
            "transformer_checkpoint_sha256": reranker_hash,
            "max_inflight": max_inflight, "queue_timeout_seconds": queue_timeout,
            "threadpool_tokens": 40,
            "training_cutoff": system.reference.isoformat(),
            "catalog_size": len(system.contents), "known_users": len(system.users),
            "candidate_count": system.config.candidate_count,
            "models": list(models),
        }
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="Crypto Content Neural Inference", version="1.2.0", lifespan=lifespan)
    app.state.runtime = None

    def runtime():
        if app.state.runtime is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        return app.state.runtime

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready():
        runtime()
        return app.state.metadata

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(request: RecommendationRequest):
        system, models, gate = runtime()
        if request.model not in models:
            raise HTTPException(status_code=422, detail="Requested model is not enabled")
        # Requests share immutable weights and history, with one CPU inference slot.
        with gate.slot():
            result = models[request.model].recommend(request.user_id, request.k, request.exclude_seen)
        return {
            "user_id": request.user_id, "model": request.model,
            "checkpoint_sha256": app.state.metadata["checkpoint_sha256"],
            "reranker_checkpoint_sha256": app.state.metadata['transformer_checkpoint_sha256']
                if request.model == TransformerReranker.name else None,
            "training_cutoff": system.reference.isoformat(),
            "items": result[["content_id", "title", "topic_tags", "score"]].to_dict("records"),
        }

    return app


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--transformer-checkpoint', type=Path, default=DEFAULT_TRANSFORMER)
    parser.add_argument('--disable-transformer', action='store_true')
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--queue-timeout", type=float, default=1.0)
    args = parser.parse_args()
    app = create_app(args.checkpoint, args.torch_threads, args.max_inflight, args.queue_timeout,
                     None if args.disable_transformer else args.transformer_checkpoint)
    uvicorn.run(app, host=args.host, port=args.port, workers=1, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
