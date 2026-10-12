"""Shuffled, rank-sharded token data with an explicit recoverable cursor."""

import random

import torch


class StatefulTokenStream:
    def __init__(self, config, rank):
        self.config = config
        self.generator = torch.Generator().manual_seed(config.seed + 10000 + rank)
        source = torch.Generator().manual_seed(config.seed + 20000)
        # Each worker owns a disjoint shard; a shuffle is repeated each epoch.
        starts = torch.randint(config.vocab_size, (config.dataset_size,), generator=source)
        self.starts = starts[rank :: config.world_size]
        self.order = torch.randperm(len(self.starts), generator=self.generator)
        self.cursor = 0
        self.epoch = 0

    def next_batch(self):
        indices = []
        for _ in range(self.config.batch_size):
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self.order = torch.randperm(len(self.starts), generator=self.generator)
            indices.append(self.order[self.cursor].item())
            self.cursor += 1
        starts = self.starts[indices, None]
        # Random augmentation and dropout exercise both Python and torch RNG recovery.
        offset = random.randrange(self.config.vocab_size)
        noise = torch.randint(self.config.vocab_size, starts.shape)
        tokens = starts + offset + noise + torch.arange(self.config.sequence_length + 1)
        tokens %= self.config.vocab_size
        return tokens[:, :-1], tokens[:, 1:]

    def state_dict(self):
        return {
            "generator": self.generator.get_state(),
            "order": self.order.clone(),
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        order = state["order"]
        if (
            order.shape != self.order.shape
            or not 0 <= state["cursor"] <= len(order)
            or state["epoch"] < 0
            or not torch.equal(order.sort().values, torch.arange(len(order)))
        ):
            raise ValueError("invalid dataset cursor/permutation")
        self.generator.set_state(state["generator"])
        self.order = order.clone()
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]


def capture_state(stream, device):
    return {
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "mps_rng": torch.mps.get_rng_state().cpu() if device.type == "mps" else None,
        "data": stream.state_dict() if stream is not None else None,
    }


def restore_state(state, stream, device):
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps_rng"])
    if stream is not None:
        stream.load_state_dict(state["data"])
