"""Render measured throughput from the GPU profiling CSV."""
import hashlib
import json
from pathlib import Path

import pandas as pd
from src.analytics import plt
from src.config import REPORTS_DIR
from src.gpu_inference import MODEL_NAMES


def plot(output: Path = REPORTS_DIR / 'gpu_profile'):
    output = Path(output)
    frame = pd.read_csv(output / 'summary.csv')
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    paths = [('cpu', 'float32', 'batched', 'CPU FP32 batch'),
        ('mps', 'float32', 'per_user_loop', 'MPS FP32 per-user loop'),
        ('mps', 'float32', 'batched_baseline', 'MPS FP32 batch before sort optimization'),
        ('mps', 'float32', 'batched', 'MPS FP32 optimized batch'),
        ('mps', 'float16', 'batched', 'MPS FP16 optimized batch')]
    for axis, model in zip(axes, MODEL_NAMES):
        for device, precision, mode, label in paths:
            rows = frame.loc[(frame.model == model) & (frame.device == device)
                & (frame.precision == precision) & (frame['mode'] == mode)].sort_values('batch_size')
            axis.plot(rows.batch_size, rows.requests_per_second, marker='o', label=label)
        axis.set_xscale('log', base=2)
        axis.set_yscale('log')
        axis.set(title='Two-tower retrieval' if model == MODEL_NAMES[0] else 'Neural ranking',
                 xlabel='Batch size', ylabel='Requests per second')
        axis.set_xticks(sorted(frame.batch_size.unique()))
        axis.set_xticklabels(sorted(frame.batch_size.unique()))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    path = output / 'throughput.png'
    figure.savefig(path, dpi=160)
    plt.close(figure)
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return path


if __name__ == '__main__':
    plot()
