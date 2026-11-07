"""Train, evaluate, export, and reload-verify the two neural recommendation stages."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ARTIFACTS_DIR, PROJECT_ROOT, REPORTS_DIR
from src.data_pipeline import build_dataset
from src.evaluation import chronological_holdout, evaluate_all
from src.neural import NeuralConfig, NeuralSystem
from src.recommenders import build_all_models


def run(output: Path, config: NeuralConfig):
    output.mkdir(parents=True, exist_ok=True)
    bundle = build_dataset(save=False)
    split = chronological_holdout(bundle.interactions, bundle.contents)
    system = NeuralSystem(split.contents, bundle.users, split.train, config).fit()
    models = build_all_models(split.contents, bundle.users, split.train)
    neural_models = system.recommenders()
    models.update(neural_models)
    metrics, details = evaluate_all(models, split.test)
    metrics.to_csv(output / "model_metrics.csv", index=False)
    details.to_csv(output / "evaluation_details.csv", index=False)
    history = pd.DataFrame(system.history)
    history.to_csv(output / "training_history.csv", index=False)
    frames = []
    for user_id in bundle.users.user_id:
        for model in neural_models.values():
            result = model.recommend(user_id, k=10)
            result.insert(0, "user_id", user_id)
            result.insert(1, "rank", np.arange(1, len(result) + 1))
            frames.append(result)
    pd.concat(frames, ignore_index=True).to_csv(output / "recommendations.csv", index=False)
    examples = system.examples
    audit = pd.DataFrame({
        "user_id": examples.user_ids,
        "content_id": [system.encoder.content_ids[index] for index in examples.item_indices],
        "impression_time": pd.to_datetime(examples.timestamps, utc=True),
        "outcome_window_end": pd.to_datetime(examples.label_ends, utc=True),
        "stage_window": examples.windows,
        "label": examples.labels.astype(int),
    })
    audit["ranker_candidate"] = False
    audit.loc[system.ranker_example_indices, "ranker_candidate"] = True
    audit.to_csv(output / "training_examples.csv", index=False)
    checkpoint = output / "neural_recommender.pt"
    system.save(checkpoint)
    restored = NeuralSystem.load(checkpoint)
    max_error = 0.0
    for user_id in [*bundle.users.user_id, "new-user"]:
        np.testing.assert_allclose(system.retrieval_scores(user_id), restored.retrieval_scores(user_id), atol=1e-7, rtol=0)
        np.testing.assert_allclose(system.rank_scores(user_id), restored.rank_scores(user_id), atol=1e-7, rtol=0)
        np.testing.assert_array_equal(system.candidate_indices(user_id), restored.candidate_indices(user_id))
        max_error = max(max_error, float(np.max(np.abs(system.retrieval_scores(user_id) - restored.retrieval_scores(user_id)))))
    # Matplotlib uses the existing project's noninteractive configuration.
    from src.analytics import plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, stage in zip(axes, ("two_tower", "neural_ranker")):
        data = history.loc[history.stage == stage]
        axis.plot(data.epoch, data.train_loss, label="Training")
        axis.plot(data.epoch, data.validation_loss, label="Validation")
        axis.axvline(system.training_summary[stage]["selected_epoch"], color="#555555", linestyle=":", label="Selected epoch")
        axis.set(title=stage.replace("_", " ").title(), xlabel="Epoch", ylabel="Weighted BCE loss")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=160)
    plt.close(figure)
    sources = [PROJECT_ROOT / "src" / name for name in (
        "neural.py", "run_neural.py", "data_pipeline.py", "evaluation.py", "recommenders.py", "config.py")]
    source_hash = hashlib.sha256(b"".join(path.read_bytes() for path in sources)).hexdigest()
    window_counts = []
    for index in range(4):
        target = examples.labels[examples.indices(index)]
        window_counts.append({"window": index, "examples": len(target), "positive": int(target.sum()),
                              "negative": int(len(target) - target.sum())})
    summary = {"config": asdict(config), "device": "cpu", "python": platform.python_version(),
        "torch": torch.__version__, "platform": platform.platform(), "source_sha256": source_hash,
        "training_cutoff": split.cutoff.isoformat(), "training_events": len(split.train),
        "eligible_contents": len(split.contents), "evaluated_users": len(split.test),
        "training": system.training_summary, "window_label_counts": window_counts,
        "checkpoint_reload_users": len(bundle.users) + 1,
        "checkpoint_reload_max_score_error": max_error,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "feature_dimensions": {"user": system.encoder.user_dim, "item": system.encoder.item_features.shape[1],
                               "embedding": config.embedding_dim}}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.iterdir()) if path.is_file() and path.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(hashes, indent=2) + "\n")
    report = ["# PyTorch Recommendation Experiment", "",
        "The run trains two towers on impression outcomes, then freezes their weights before training a neural ranker.",
        "Labels are observed within 15 minutes of each impression, and user features use events strictly before the request.", "",
        "## Training", "", pd.DataFrame({key: value for key, value in system.training_summary.items()
            if key in ("two_tower", "neural_ranker")}).T.to_markdown(), "",
        "## Offline evaluation", "", metrics.round(4).to_markdown(index=False), "",
        f"The checkpoint reload was checked for {len(bundle.users) + 1} users, with maximum retrieval-score error {max_error}.",
        "The summary records CPU execution, configuration, chronological boundaries, label counts, and file hashes."]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "NEURAL_EXPERIMENT_RESULTS.md").write_text("\n".join(report) + "\n")
    print(metrics.round(4).to_string(index=False))
    print(json.dumps(system.training_summary, indent=2))
    return system, metrics, summary


def verify_export(output: Path):
    hashes = json.loads((output / "manifest.json").read_text())
    for name, expected in hashes.items():
        if Path(name).name != name:
            raise ValueError("Invalid manifest path")
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Artifact hash mismatch: {name}")
    system = NeuralSystem.load(output / "neural_recommender.pt")
    saved = pd.read_csv(output / "recommendations.csv")
    models = system.recommenders()
    for (model, user), expected in saved.groupby(["model", "user_id"]):
        actual = models[model].recommend(user, k=10)
        expected = expected.sort_values("rank")
        if actual.content_id.tolist() != expected.content_id.tolist():
            raise AssertionError(f"Recommendation IDs changed for {model}/{user}")
        np.testing.assert_allclose(actual.score, expected.score, rtol=1e-5, atol=1e-6)
    print(f"Verified {len(hashes)} artifact hashes and {saved.user_id.nunique()} users' checkpoint recommendations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ARTIFACTS_DIR / "neural")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_export(args.output)
    else:
        run(args.output, NeuralConfig(epochs=args.epochs, seed=args.seed))


if __name__ == "__main__":
    main()
