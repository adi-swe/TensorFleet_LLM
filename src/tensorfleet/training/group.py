"""Shared executor contract for ordinary single-worker and DDP execution."""

from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tensorfleet.training.device import backend_name


class WorkerGroup:
    def __init__(self, config, rank, rendezvous):
        self.config = config
        self.rank = rank
        self.rendezvous = rendezvous
        self.backend = backend_name(config)

    def __enter__(self):
        if self.backend is not None:
            dist.init_process_group(
                self.backend,
                init_method=self.rendezvous,
                rank=self.rank,
                world_size=self.config.world_size,
                timeout=timedelta(seconds=self.config.collective_timeout),
            )
        return self

    def __exit__(self, *_):
        if self.backend is not None:
            dist.destroy_process_group()

    def wrap(self, model):
        if self.backend is None:
            return model
        return DistributedDataParallel(
            model, device_ids=[self.rank] if self.backend == "nccl" else None
        )

    def barrier(self):
        if self.backend is not None:
            dist.barrier()

    def gather_states(self, state):
        if self.backend is None:
            return [state]
        states = [None] * self.config.world_size
        dist.all_gather_object(states, state)
        return states

    def sum_metrics(self, values):
        if self.backend is not None:
            dist.all_reduce(values)

    def durations(self, duration):
        if self.backend is None:
            return [duration.item()]
        durations = [torch.zeros_like(duration) for _ in range(self.config.world_size)]
        dist.all_gather(durations, duration)
        return [value.item() for value in durations]
