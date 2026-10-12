"""Discrete events replay profiled group durations; no training is executed."""

import heapq
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SimulationResult:
    kind: str
    total_seconds: float
    completed_steps: int
    replayed_steps: int
    events: list[dict]

    def to_dict(self):
        return asdict(self)


def load_profile(directory):
    # Later successful attempts replace rolled-back measurements at the same step.
    by_step = {}
    paths = sorted(
        Path(directory).glob("metrics-attempt-*.jsonl"),
        key=lambda path: int(path.stem.rsplit("-", 1)[1]),
    )
    for path in paths:
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["kind"] != "measured":
                raise ValueError("profile must contain measured training records")
            by_step[record["step"]] = record["step_seconds"]
    if not by_step:
        raise ValueError("no measured step durations found")
    return [by_step[step] for step in sorted(by_step)]


def simulate(
    durations,
    steps,
    checkpoint_every=2,
    fail_step=None,
    restart_seconds=1.0,
    checkpoint_seconds=0.0,
    overhead_seconds=0.0,
    startup_seconds=0.0,
    teardown_seconds=0.0,
    contention_factor=1.0,
    worker_delay_seconds=0.0,
    delay_start=0,
    delay_stop=None,
):
    durations = list(durations)
    if not durations or any(not math.isfinite(d) or d <= 0 for d in durations):
        raise ValueError("profile durations must be finite and positive")
    if (
        type(steps) is not int
        or steps <= 0
        or type(checkpoint_every) is not int
        or checkpoint_every <= 0
        or not math.isfinite(restart_seconds)
        or restart_seconds < 0
    ):
        raise ValueError("invalid simulation configuration")
    if fail_step is not None and (type(fail_step) is not int or not 0 <= fail_step < steps):
        raise ValueError("failure step outside simulation")
    for value in (
        checkpoint_seconds,
        overhead_seconds,
        startup_seconds,
        teardown_seconds,
        worker_delay_seconds,
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError("costs and delays must be finite and nonnegative")
    if not math.isfinite(contention_factor) or contention_factor < 1:
        raise ValueError("contention factor must be finite and at least one")
    if delay_start < 0 or (delay_stop is not None and delay_stop <= delay_start):
        raise ValueError("invalid delay interval")
    queue = [(startup_seconds, "start", 0)]
    events = []
    committed = 0
    replayed = 0
    injected = False
    while queue:
        now, event, step = heapq.heappop(queue)
        events.append({"time": now, "event": event, "step": step})
        if event in ("start", "restart"):
            if step == fail_step and not injected:
                injected = True
                replayed += step - committed
                events.append({"time": now, "event": "failure", "step": step})
                heapq.heappush(queue, (now + restart_seconds, "restart", committed))
            else:
                heapq.heappush(
                    queue,
                    (
                        now
                        + durations[step % len(durations)] * contention_factor
                        + overhead_seconds
                        + (
                            worker_delay_seconds
                            if step >= delay_start and (delay_stop is None or step < delay_stop)
                            else 0.0
                        ),
                        "complete",
                        step,
                    ),
                )
        elif event == "complete":
            if (step + 1) % checkpoint_every == 0 or step + 1 == steps:
                heapq.heappush(queue, (now + checkpoint_seconds, "checkpoint", step + 1))
            elif step + 1 < steps:
                heapq.heappush(queue, (now, "start", step + 1))
        if event == "checkpoint":
            committed = step
            if step < steps:
                heapq.heappush(queue, (now, "start", step))
    return SimulationResult("simulated", now + teardown_seconds, steps, replayed, events)
