"""Explicit communication samples, with protection against double counting.

No topology/bandwidth extrapolation is inferred. Empty samples mean communication
is already included in the work profile. Separate samples may only be composed
with a profile explicitly identified as isolated compute.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CommunicationModel:
    durations: tuple[float, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "durations", tuple(self.durations))
        if any(not math.isfinite(d) or d < 0 for d in self.durations):
            raise ValueError("communication durations must be finite and nonnegative")

    def validate_profile(self, includes_communication):
        if self.durations and includes_communication:
            raise ValueError("communication is already included in the work profile")

    def duration(self, step):
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return self.durations[step % len(self.durations)] if self.durations else 0.0
