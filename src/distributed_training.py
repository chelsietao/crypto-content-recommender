"""DDP trainer with a fixed global batch and sample-weighted rank reductions."""
from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def global_batches(indices, batch_size, seed, epoch):
    if epoch < 1:
        raise ValueError('Training epochs are numbered from one')
    generator = torch.Generator().manual_seed(seed)
    # Preserve the sequential generator sequence of the existing CPU trainer.
    for _ in range(epoch):
        order = torch.randperm(len(indices), generator=generator)
    return torch.as_tensor(indices)[order].split(batch_size)


def rank_batch(global_batch, rank, world_size):
    selected = global_batch[rank::world_size]
    return selected if len(selected) else global_batch[:1], len(selected)


def weighted_rank_loss(logits, labels, positive_weight, real_count, global_count, world_size):
    if not real_count:
        return logits.sum() * 0
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels,
        pos_weight=positive_weight, reduction='sum')
    return loss * (world_size / global_count)


class DistributedTrainer:
    def __init__(self, device, dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.steps, self.communication, self.payloads = [], [], []

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def maximum_time(self, elapsed):
        value = torch.tensor(elapsed, dtype=torch.float64, device=self.device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return value.item()

    def __call__(self, model, inputs, labels, train_indices, validation_indices, config, stage):
        if not len(train_indices) or not len(validation_indices):
            raise ValueError('Distributed training requires nonempty train and validation windows')
        positive_count = float(labels[train_indices].sum())
        if not 0 < positive_count < len(train_indices):
            raise ValueError('Distributed training requires positive and negative labels')
        fingerprint = hashlib.sha256()
        for value in (*inputs, labels, torch.as_tensor(train_indices), torch.as_tensor(validation_indices)):
            fingerprint.update(value.contiguous().numpy().tobytes())
        fingerprints = [None] * self.world_size
        dist.all_gather_object(fingerprints, fingerprint.hexdigest())
        if len(set(fingerprints)) != 1:
            raise ValueError('Ranks received inconsistent training inputs')
        original_dtype = next(model.parameters()).dtype
        model.to(device=self.device, dtype=self.dtype)
        distributed = DistributedDataParallel(model,
            device_ids=[self.device.index] if self.device.type == 'cuda' else None,
            broadcast_buffers=False)
        values = tuple(value.to(device=self.device, dtype=self.dtype) for value in inputs)
        targets = labels.to(device=self.device, dtype=self.dtype)
        positive_weight = torch.tensor((len(train_indices) - positive_count) / positive_count, device=self.device, dtype=self.dtype)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        initial = copy.deepcopy(model.state_dict())
        history = []
        best_loss, best_epoch, best_state = float('inf'), 0, None

        def loss_on(indices):
            selected = torch.as_tensor(indices[self.rank::self.world_size], device=self.device)
            with torch.no_grad():
                if len(selected):
                    loss = nn.functional.binary_cross_entropy_with_logits(
                        model(*(value[selected] for value in values)), targets[selected],
                        pos_weight=positive_weight, reduction='sum')
                else:
                    loss = torch.zeros((), device=self.device)
            totals = torch.stack([loss.double(), torch.tensor(len(selected), dtype=torch.float64, device=self.device)])
            dist.all_reduce(totals)
            return (totals[0] / totals[1]).item()

        dist.barrier()
        self.synchronize()
        started = time.perf_counter()
        for epoch in range(config.epochs + 1):
            if epoch:
                distributed.train()
                for batch_index, global_batch in enumerate(global_batches(train_indices, config.batch_size, config.seed, epoch)):
                    local_indices, real_count = rank_batch(global_batch, self.rank, self.world_size)
                    selected = local_indices.to(self.device)
                    optimizer.zero_grad(set_to_none=True)
                    self.synchronize()
                    tick = time.perf_counter()
                    logits = distributed(*(value[selected] for value in values))
                    loss = weighted_rank_loss(logits, targets[selected], positive_weight,
                                              real_count, len(global_batch), self.world_size)
                    finite = torch.tensor(int(torch.isfinite(loss).item()), device=self.device)
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                    if not finite.item():
                        raise RuntimeError('Nonfinite distributed loss')
                    self.synchronize()
                    forward_seconds = time.perf_counter() - tick
                    tick = time.perf_counter()
                    loss.backward()
                    self.synchronize()
                    backward_seconds = time.perf_counter() - tick
                    tick = time.perf_counter()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    self.synchronize()
                    optimizer_seconds = time.perf_counter() - tick
                    self.steps.append({'rank': self.rank, 'stage': stage, 'epoch': epoch,
                        'batch_index': batch_index, 'global_examples': len(global_batch), 'local_examples': real_count,
                        'forward_seconds': forward_seconds, 'backward_with_ddp_seconds': backward_seconds,
                        'optimizer_seconds': optimizer_seconds})
            model.eval()
            train_loss, validation_loss = loss_on(train_indices), loss_on(validation_indices)
            if not np.isfinite([train_loss, validation_loss]).all():
                raise RuntimeError('Nonfinite distributed validation loss')
            history.append({'stage': stage, 'epoch': epoch, 'train_loss': train_loss, 'validation_loss': validation_loss})
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_state = copy.deepcopy(model.state_dict())
        self.synchronize()
        elapsed = self.maximum_time(time.perf_counter() - started)
        model.load_state_dict(best_state)
        model.eval()
        changed = any(not torch.equal(initial[key], best_state[key]) for key in initial)
        # Verify the selected weights agree on every rank before CPU feature preparation.
        flattened = torch.cat([value.detach().flatten() for value in model.state_dict().values()])
        reference = flattened.clone()
        dist.broadcast(reference, src=0)
        difference = torch.max(torch.abs(flattened - reference))
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        if difference.item() != 0:
            raise AssertionError('Distributed ranks selected different weights')
        payload_bytes = flattened.numel() * flattened.element_size()
        self.payloads.append((stage, payload_bytes))
        del distributed
        model.to(device='cpu', dtype=original_dtype)
        return history, {'selected_epoch': best_epoch, 'validation_loss': best_loss,
            'parameters_changed': changed, 'training_examples': len(train_indices),
            'validation_examples': len(validation_indices), 'world_size': self.world_size,
            'global_batch_size': config.batch_size, 'training_precision': str(self.dtype), 'training_seconds_max_rank': elapsed,
            'training_input_sha256': fingerprint.hexdigest(),
            'label_partition_sha256': hashlib.sha256(b''.join(value.contiguous().numpy().tobytes()
                for value in (labels, torch.as_tensor(train_indices), torch.as_tensor(validation_indices)))).hexdigest(),
            'replica_max_parameter_error': difference.item(), 'parameter_bytes': payload_bytes}

    def communication_benchmark(self, payload_bytes, stage):
        value = torch.ones(max(payload_bytes // torch.empty((), dtype=self.dtype).element_size(), 1), device=self.device, dtype=self.dtype)
        for iteration in range(15):
            value.fill_(1)
            dist.barrier()
            self.synchronize()
            tick = time.perf_counter()
            dist.all_reduce(value)
            self.synchronize()
            elapsed = time.perf_counter() - tick
            if not torch.all(value == self.world_size).item():
                raise AssertionError('Communication benchmark returned an incorrect sum')
            maximum = self.maximum_time(elapsed)
            if iteration >= 5:
                self.communication.append({'stage': stage, 'iteration': iteration - 5,
                    'payload_bytes_per_rank': payload_bytes, 'world_size': self.world_size,
                    'all_reduce_seconds_max_rank': maximum})
