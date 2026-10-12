"""FIFO shared service-slot model for overlapping synchronous training jobs.

Profiled step duration is treated as service time for each worker. This is an
explicit approximation: profiles include communication and are not pure CPU
service demand. Slots are abstract capacity, not calibrated physical CPU cores.
"""

import math
from collections import deque

from tensorfleet.simulator.event_queue import EventQueue


def simulate_shared_capacity(durations, jobs, world_size=2, slots=2):
    if (
        not durations
        or any(not math.isfinite(d) or d <= 0 for d in durations)
        or not jobs
        or type(world_size) is not int
        or world_size < 1
        or type(slots) is not int
        or slots < 1
        or any(type(steps) is not int or steps < 1 for steps in jobs)
    ):
        raise ValueError("positive profile, job steps, worker count and slots required")
    waiting = deque((job, 0, rank) for job in range(len(jobs)) for rank in range(world_size))
    running = EventQueue()
    remaining = {(job, 0): world_size for job in range(len(jobs))}
    finished = {}
    now = 0.0
    events = []
    while waiting or running:
        while waiting and len(running) < slots:
            job, step, rank = waiting.popleft()
            running.schedule(
                now + durations[step % len(durations)], "worker-complete", step, job=job, rank=rank
            )
            events.append(
                {"time": now, "event": "worker-start", "job": job, "step": step, "rank": rank}
            )
        completed = running.pop()
        now, job, step, rank = completed.time, completed.job, completed.step, completed.rank
        events.append(
            {"time": now, "event": "worker-complete", "job": job, "step": step, "rank": rank}
        )
        remaining[job, step] -= 1
        if remaining[job, step] == 0:
            if step + 1 == jobs[job]:
                finished[job] = now
            else:
                remaining[job, step + 1] = world_size
                waiting.extend((job, step + 1, rank) for rank in range(world_size))
    return {
        "kind": "simulated",
        "total_seconds": now,
        "job_completion_seconds": finished,
        "slots": slots,
        "events": events,
        "capacity_model": "uncalibrated FIFO service slots",
    }
