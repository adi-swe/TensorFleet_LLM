"""Profiled work duration and controlled slowdown/straggler assumptions.

Default samples are end-to-end training steps, including DDP communication. They
must not be interpreted as isolated compute demand or extrapolated across hardware.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeModel:
    durations: tuple[float, ...]
    contention_factor: float = 1.0
    worker_delay_seconds: float = 0.0
    delay_start: int = 0
    delay_stop: int | None = None
    includes_communication: bool = True

    def __post_init__(self):
        object.__setattr__(self, "durations", tuple(self.durations))
        if not self.durations or any(not math.isfinite(d) or d <= 0 for d in self.durations):
            raise ValueError("profile durations must be finite and positive")
        if not math.isfinite(self.contention_factor) or self.contention_factor < 1:
            raise ValueError("contention factor must be finite and at least one")
        if not math.isfinite(self.worker_delay_seconds) or self.worker_delay_seconds < 0:
            raise ValueError("worker delay must be finite and nonnegative")
        if (
            type(self.delay_start) is not int
            or self.delay_start < 0
            or (
                self.delay_stop is not None
                and (type(self.delay_stop) is not int or self.delay_stop <= self.delay_start)
            )
        ):
            raise ValueError("invalid delay interval")
        if type(self.includes_communication) is not bool:
            raise ValueError("includes_communication must be boolean")

    def duration(self, step):
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        delay = (
            self.worker_delay_seconds
            if (step >= self.delay_start and (self.delay_stop is None or step < self.delay_stop))
            else 0.0
        )
        return self.durations[step % len(self.durations)] * self.contention_factor + delay
