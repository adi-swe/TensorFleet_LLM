"""Stable discrete-event queue with monotonic simulated time."""

import heapq
import math
from dataclasses import dataclass
from itertools import count


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    step: int
    job: int | None = None
    rank: int | None = None


class EventQueue:
    def __init__(self):
        self._heap = []
        self._sequence = count()
        self.now = 0.0

    def schedule(self, time, kind, step, *, job=None, rank=None):
        if not math.isfinite(time) or time < self.now:
            raise ValueError("event time must be finite and cannot precede simulated time")
        if not isinstance(kind, str) or not kind or type(step) is not int or step < 0:
            raise ValueError("event requires a kind and nonnegative integer step")
        event = Event(time, kind, step, job, rank)
        heapq.heappush(self._heap, (time, next(self._sequence), event))

    def pop(self):
        _, _, event = heapq.heappop(self._heap)
        self.now = event.time
        return event

    def __len__(self):
        return len(self._heap)
