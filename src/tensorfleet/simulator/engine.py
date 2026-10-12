"""Compose event and cost models; no real training is launched here."""

import math
from dataclasses import asdict, dataclass

from tensorfleet.simulator.checkpoint_model import CheckpointModel
from tensorfleet.simulator.communication_model import CommunicationModel
from tensorfleet.simulator.compute_model import ComputeModel
from tensorfleet.simulator.event_queue import EventQueue


@dataclass(frozen=True)
class SimulationResult:
    kind: str
    total_seconds: float
    completed_steps: int
    replayed_steps: int
    events: list[dict]

    def to_dict(self):
        return asdict(self)


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
    *,
    communication_durations=None,
    durations_include_communication=True,
):
    work = ComputeModel(
        tuple(durations),
        contention_factor,
        worker_delay_seconds,
        delay_start,
        delay_stop,
        durations_include_communication,
    )
    communication = CommunicationModel(
        () if communication_durations is None else tuple(communication_durations)
    )
    communication.validate_profile(work.includes_communication)
    checkpoint = CheckpointModel(checkpoint_every, checkpoint_seconds, restart_seconds)
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if fail_step is not None and (type(fail_step) is not int or not 0 <= fail_step < steps):
        raise ValueError("failure step outside simulation")
    if any(
        not math.isfinite(v) or v < 0 for v in (overhead_seconds, startup_seconds, teardown_seconds)
    ):
        raise ValueError("costs must be finite and nonnegative")
    queue = EventQueue()
    queue.schedule(startup_seconds, "start", 0)
    events = []
    replayed = 0
    injected = False
    while queue:
        item = queue.pop()
        now, event, step = item.time, item.kind, item.step
        events.append({"time": now, "event": event, "step": step})
        if event in ("start", "restart"):
            if step == fail_step and not injected:
                injected = True
                replayed += checkpoint.lost_steps(step)
                events.append({"time": now, "event": "failure", "step": step})
                queue.schedule(
                    now + checkpoint.restart_seconds, "restart", checkpoint.committed_step
                )
            else:
                queue.schedule(
                    now + work.duration(step) + communication.duration(step) + overhead_seconds,
                    "complete",
                    step,
                )
        elif event == "complete":
            if checkpoint.due(step + 1, steps):
                queue.schedule(now + checkpoint.write_seconds, "checkpoint", step + 1)
            elif step + 1 < steps:
                queue.schedule(now, "start", step + 1)
        elif event == "checkpoint":
            checkpoint.commit(step)
            if step < steps:
                queue.schedule(now, "start", step)
    return SimulationResult("simulated", queue.now + teardown_seconds, steps, replayed, events)
