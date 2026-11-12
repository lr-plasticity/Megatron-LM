# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dense snapshots of the fp32 master weights during training, for post-hoc weight averaging.

Two kinds of snapshot, each in its own directory under ``--weight-snapshot-dir``:

    block/iter_XXXXXXX   the mean of the weights after every step in the block of
                         ``--weight-snapshot-block-interval`` steps that ends at iteration X.
                         Every step enters exactly one block, so EMAs and uniform averages
                         built from the blocks see every iterate, not a subsample.
    raw/iter_XXXXXXX     the weights after iteration X, every
                         ``--weight-snapshot-raw-interval`` steps.

Each data-parallel rank writes ``rank_<r>.pt`` holding its own shard of every parameter of the
distributed optimizer: ``{'iteration', 'kind', 'count', 'first_iteration', 'rank',
'world_size', 'params': {name: {'shape', 'start', 'end', 'data'}}}``, with ``data`` the flat
fp32 slice ``[start, end)`` of the parameter. A snapshot is complete when every rank's file
exists (files are written under a temporary name and renamed).

A block cut short by a restart (training resumed from a checkpoint inside the block) is saved
with its actual ``count`` and ``first_iteration``; restarts from checkpoints at multiples of the
block interval never cut a block.
"""

import os

import torch

from megatron.core import parallel_state
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def _strip_wrappers(name):
    while name.startswith('module.'):
        name = name[len('module.'):]
    return name


class WeightSnapshots:
    def __init__(self, model, optimizer, save_dir, block_interval, raw_interval):
        assert parallel_state.get_tensor_model_parallel_world_size() == 1, 'TP=1 only'
        assert parallel_state.get_pipeline_model_parallel_world_size() == 1, 'PP=1 only'
        optimizers = getattr(optimizer, 'chained_optimizers', [optimizer])
        assert all(isinstance(opt, DistributedOptimizer) for opt in optimizers), \
            '--weight-snapshot-dir needs --use-distributed-optimizer'

        names = {}
        for chunk in model:
            for name, param in chunk.named_parameters():
                names[param] = _strip_wrappers(name)

        # (name, shape, start, end, optimizer, model param) for every shard this rank owns.
        self.shards = []
        for opt in optimizers:
            for model_param in opt.model_param_group_index_map:
                param_range = opt._get_model_param_range_map(model_param)['param']
                self.shards.append((names[model_param], tuple(model_param.shape),
                                    param_range.start, param_range.end, opt, model_param))
        self.save_dir = save_dir
        self.block_interval = block_interval
        self.raw_interval = raw_interval
        self.rank = parallel_state.get_data_parallel_rank()
        self.world_size = parallel_state.get_data_parallel_world_size()
        self.sums = None
        self.count = 0
        self.first_iteration = None

    def _main_params(self):
        for _, _, _, _, opt, model_param in self.shards:
            yield opt._get_main_param_and_optimizer_states(model_param)['param']

    def _save(self, kind, iteration, tensors, count, first_iteration):
        """Save tensors / count (on the CPU, so no GPU memory is allocated)."""
        directory = os.path.join(self.save_dir, kind, f'iter_{iteration:07d}')
        os.makedirs(directory, exist_ok=True)
        params = {
            name: {'shape': shape, 'start': start, 'end': end,
                   'data': data.detach().to('cpu', torch.float32, copy=True).div_(count)}
            for (name, shape, start, end, _, _), data in zip(self.shards, tensors)
        }
        path = os.path.join(directory, f'rank_{self.rank}.pt')
        torch.save({'iteration': iteration, 'kind': kind, 'count': count,
                    'first_iteration': first_iteration, 'rank': self.rank,
                    'world_size': self.world_size, 'params': params}, path + '.tmp')
        os.replace(path + '.tmp', path)

    @torch.no_grad()
    def step(self, iteration):
        """Record the weights after optimizer step ``iteration``."""
        if self.block_interval:
            main_params = list(self._main_params())
            if self.sums is None:
                self.sums = [torch.zeros_like(p, dtype=torch.float32) for p in main_params]
            if self.count == 0:
                self.first_iteration = iteration
            for total, p in zip(self.sums, main_params):
                total.add_(p)
            self.count += 1
            if iteration % self.block_interval == 0:
                self._save('block', iteration, self.sums, self.count, self.first_iteration)
                for total in self.sums:
                    total.zero_()
                self.count = 0
        if self.raw_interval and iteration % self.raw_interval == 0:
            self._save('raw', iteration, list(self._main_params()), 1, iteration)
