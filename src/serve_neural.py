"""Standalone CPU HTTP inference for the exported two-tower and neural ranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from src.config import ARTIFACTS_DIR
from src.neural import NeuralSystem

MODEL_NAMES = ("v7_pytorch_two_tower", "v8_pytorch_neural_ranker")
ModelName = Literal["v7_pytorch_two_tower", "v8_pytorch_neural_ranker"]
DEFAULT_CHECKPOINT = ARTIFACTS_DIR / "neural" / "neural_recommender.pt"


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    user_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    model: ModelName = MODEL_NAMES[0]
    k: Annotated[int, Field(ge=1, le=50)] = 10
    exclude_seen: bool = True


class RecommendationItem(BaseModel):
    content_id: str
    title: str
    topic_tags: str
    score: Annotated[float, Field(allow_inf_nan=False)]


class RecommendationResponse(BaseModel):
    user_id: str
    model: ModelName
    checkpoint_sha256: str
    training_cutoff: str
    items: list[RecommendationItem]


def create_app(checkpoint: Path = DEFAULT_CHECKPOINT, torch_threads: int = 1) -> FastAPI:
    if isinstance(torch_threads, bool) or not isinstance(torch_threads, int) or torch_threads < 1:
        raise ValueError("torch_threads must be a positive integer")
    checkpoint = Path(checkpoint).resolve()

    @asynccontextmanager
    async def lifespan(app):
        torch.set_num_threads(torch_threads)
        # Verify the adjacent manifest before loading weights into the process.
        manifest = json.loads(checkpoint.with_name("manifest.json").read_text())
        expected = manifest.get(checkpoint.name)
        actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if not isinstance(expected, str) or actual != expected:
            raise ValueError("Checkpoint hash does not match manifest")
        system = NeuralSystem.load(checkpoint)
        models = system.recommenders()
        for model in models.values():
            model.recommend("new-user", k=10)
        app.state.runtime = (system, models, threading.Lock())
        app.state.metadata = {
            "status": "ready", "pid": os.getpid(), "device": "cpu",
            "torch_threads": torch.get_num_threads(), "workers": 1,
            "inference_slots": 1, "checkpoint_sha256": actual,
            "training_cutoff": system.reference.isoformat(),
            "catalog_size": len(system.contents), "known_users": len(system.users),
            "candidate_count": system.config.candidate_count,
            "models": list(models),
        }
        try:
            yield
        finally:
            app.state.runtime = None

    app = FastAPI(title="Crypto Content Neural Inference", version="1.0.0", lifespan=lifespan)
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
        system, models, lock = runtime()
        # Requests share immutable weights and history, with one CPU inference slot.
        with lock:
            result = models[request.model].recommend(request.user_id, request.k, request.exclude_seen)
        return {
            "user_id": request.user_id, "model": request.model,
            "checkpoint_sha256": app.state.metadata["checkpoint_sha256"],
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
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    uvicorn.run(create_app(args.checkpoint, args.torch_threads), host=args.host, port=args.port,
                workers=1, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
