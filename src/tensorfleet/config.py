"""Validated, serializable training configuration."""

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 10
    world_size: int = 2
    batch_size: int = 4
    sequence_length: int = 16
    vocab_size: int = 64
    d_model: int = 32
    nhead: int = 4
    layers: int = 1
    learning_rate: float = 0.001
    seed: int = 42
    threads: int = 1
    checkpoint_every: int = 2
    collective_timeout: float = 30.0
    attempt_timeout: float = 120.0
    max_restarts: int = 1
    fail_step: int | None = None
    fail_rank: int = 1
    device: str = "cpu"
    dropout: float = 0.0
    stochastic_data: bool = False
    dataset_size: int = 64
    straggler_rank: int = 1
    straggler_seconds: float = 0.0
    straggler_start: int = 0
    straggler_stop: int | None = None

    def __post_init__(self):
        for name in (
            "steps",
            "world_size",
            "batch_size",
            "sequence_length",
            "vocab_size",
            "d_model",
            "nhead",
            "layers",
            "threads",
            "checkpoint_every",
            "dataset_size",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_restarts", "fail_rank", "straggler_rank", "straggler_start"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name in ("fail_step", "straggler_stop"):
            if getattr(self, name) is not None and type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer or None")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda or mps")
        if self.device == "mps" and self.world_size != 1:
            raise ValueError("MPS supports exactly one worker")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.dataset_size < self.world_size:
            raise ValueError("dataset_size must provide at least one item per rank")
        if not math.isfinite(self.straggler_seconds) or self.straggler_seconds < 0:
            raise ValueError("straggler_seconds must be finite and nonnegative")
        if not 0 <= self.straggler_rank < self.world_size:
            raise ValueError("invalid straggler rank")
        if self.straggler_start < 0 or (
            self.straggler_stop is not None and self.straggler_stop <= self.straggler_start
        ):
            raise ValueError("invalid straggler step range")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (self.learning_rate, self.collective_timeout, self.attempt_timeout)
        ):
            raise ValueError("learning rate and timeouts must be positive")
        if self.max_restarts < 0 or not 0 <= self.fail_rank < self.world_size:
            raise ValueError("invalid restart budget or failure rank")
        if self.fail_step is not None and not 0 <= self.fail_step < self.steps:
            raise ValueError("fail_step must be a zero-based step within the run")

    def compatibility(self):
        """Fields whose change would invalidate deterministic checkpoint continuation."""
        return {
            k: v
            for k, v in asdict(self).items()
            if k
            not in {
                "steps",
                "checkpoint_every",
                "collective_timeout",
                "attempt_timeout",
                "max_restarts",
                "fail_step",
                "fail_rank",
                "straggler_rank",
                "straggler_seconds",
                "straggler_start",
                "straggler_stop",
            }
        }
