"""torchrun entry point for CPU/Gloo and CUDA/NCCL neural training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import platform
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.data_pipeline import build_dataset
from src.distributed_training import DistributedTrainer
from src.evaluation import chronological_holdout, evaluate_all
from src.neural import NeuralConfig, NeuralSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--precision', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--global-batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path, default=REPORTS_DIR / 'ddp_run')
    args = parser.parse_args()
    config = NeuralConfig(epochs=args.epochs, batch_size=args.global_batch_size)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if args.device == 'cuda':
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError('Each CUDA rank requires its own visible GPU')
        if not dist.is_nccl_available():
            raise RuntimeError('CUDA distributed training requires NCCL')
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device, backend = torch.device('cuda', local_rank), 'nccl'
    else:
        device, backend = torch.device('cpu'), 'gloo'
    torch.set_num_threads(1)
    dist.init_process_group(backend, init_method='env://', timeout=timedelta(seconds=120))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        trainer = DistributedTrainer(device, getattr(torch, args.precision))
        started = time.perf_counter()
        bundle = build_dataset(save=False)
        split = chronological_holdout(bundle.interactions, bundle.contents)
        system = NeuralSystem(split.contents, bundle.users, split.train, config).fit(trainer=trainer)
        trainer.synchronize()
        fit_seconds = trainer.maximum_time(time.perf_counter() - started)
        for stage, payload in trainer.payloads:
            trainer.communication_benchmark(payload, stage)
        rank_info = {'rank': rank, 'local_rank': local_rank, 'world_size': world_size,
            'device': str(device), 'backend': backend, 'pid': os.getpid(),
            'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor()}
        if device.type == 'cuda':
            rank_info['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
        (output / f'rank_{rank}.json').write_text(json.dumps(rank_info, indent=2) + '\n')
        pd.DataFrame(trainer.steps).to_csv(output / f'steps_rank_{rank}.csv', index=False)
        if rank == 0:
            checkpoint = output / 'neural_recommender.pt'
            system.save(checkpoint)
            restored = NeuralSystem.load(checkpoint)
            restored_models = restored.recommenders()
            models = system.recommenders()
            metrics, details = evaluate_all(models, split.test)
            metrics.to_csv(output / 'model_metrics.csv', index=False)
            details.to_csv(output / 'evaluation_details.csv', index=False)
            pd.DataFrame(system.history).to_csv(output / 'training_history.csv', index=False)
            pd.DataFrame(trainer.communication).to_csv(output / 'communication.csv', index=False)
            frames = []
            for user in [*bundle.users.user_id, 'new-user']:
                for model, wrapper in models.items():
                    actual = wrapper.recommend(user)
                    reloaded = restored_models[model].recommend(user)
                    assert actual.content_id.tolist() == reloaded.content_id.tolist()
                    np.testing.assert_allclose(actual.score, reloaded.score, atol=1e-7, rtol=0)
                    actual.insert(0, 'user_id', user)
                    actual.insert(1, 'rank', range(1, len(actual) + 1))
                    frames.append(actual)
            pd.concat(frames, ignore_index=True).to_csv(output / 'recommendations.csv', index=False)
            sources = ('distributed_training.py', 'train_distributed.py', 'neural.py', 'data_pipeline.py', 'evaluation.py')
            summary = {'device': args.device, 'training_precision': args.precision, 'backend': backend, 'world_size': world_size,
                'epochs': config.epochs, 'global_batch_size': config.batch_size,
                'torch': torch.__version__, 'python': platform.python_version(), 'platform': platform.platform(),
                'fit_seconds_max_rank': fit_seconds, 'training': system.training_summary,
                'timing_scope': 'data preparation and model fit, excluding isolated communication benchmark and exports',
                'communication_scope': 'independent all_reduce at each stage parameter payload, synchronized, five warmups and ten samples',
                'checkpoint_reload_users': len(bundle.users) + 1,
                'ranker_example_index_sha256': hashlib.sha256(system.ranker_example_indices.tobytes()).hexdigest(),
                'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'source_sha256': {name: hashlib.sha256((PROJECT_ROOT / 'src' / name).read_bytes()).hexdigest() for name in sources}}
            (output / 'results.json').write_text(json.dumps(summary, indent=2) + '\n')
            print(json.dumps({'rank': rank, 'device': args.device, 'world_size': world_size,
                'fit_seconds': fit_seconds, 'training': system.training_summary}), flush=True)
        dist.barrier()
        if rank == 0:
            files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.iterdir())
                     if path.is_file() and path.name not in ('manifest.json', 'launcher.log')}
            (output / 'manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
