"""Checkpoint cadence, write/restart costs and committed simulation progress."""

import math
from dataclasses import dataclass, field


@dataclass
class CheckpointModel:
    interval: int = 2
    write_seconds: float = 0.0
    restart_seconds: float = 1.0
    committed_step: int = field(default=0, init=False)

    def __post_init__(self):
        if type(self.interval) is not int or self.interval <= 0:
            raise ValueError("checkpoint interval must be a positive integer")
        if any(not math.isfinite(v) or v < 0 for v in (self.write_seconds, self.restart_seconds)):
            raise ValueError("checkpoint and restart costs must be finite and nonnegative")

    def due(self, next_step, total_steps):
        return next_step % self.interval == 0 or next_step == total_steps

    def commit(self, next_step):
        if type(next_step) is not int or next_step <= self.committed_step:
            raise ValueError("checkpoint progress must increase after a completed write")
        self.committed_step = next_step

    def lost_steps(self, next_step):
        if type(next_step) is not int or next_step < self.committed_step:
            raise ValueError("rollback cannot precede committed progress")
        return next_step - self.committed_step
