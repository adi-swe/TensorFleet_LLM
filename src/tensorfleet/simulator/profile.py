"""Measured cost profiles from complete, successful single-attempt runs."""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimingProfile:
    source: str
    config: dict
    environment: dict
    step_seconds: list[float]
    rank_step_seconds: list[list[float]]
    overhead_seconds: list[float]
    checkpoint_seconds: float
    startup_seconds: float
    teardown_seconds: float

    def to_dict(self):
        return asdict(self)


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def load_timing_profile(directory):
    directory = Path(directory).resolve()
    events = read_records(directory / "events.jsonl")
    if len(events) != 1 or events[0]["failed"]:
        raise ValueError("calibration requires a successful single-attempt run")
    run = json.loads((directory / "run.json").read_text())
    records = read_records(directory / "metrics-attempt-0.jsonl")
    if [record["step"] for record in records] != list(range(run["config"]["steps"])) or any(
        record["kind"] != "measured" for record in records
    ):
        raise ValueError("calibration requires a full measured run from step zero")
    lifecycle = read_records(directory / "lifecycle-attempt-0.jsonl")
    startup = next(record["startup_seconds"] for record in lifecycle if record["event"] == "ready")
    loop = next(
        record["training_loop_seconds"] for record in lifecycle if record["event"] == "complete"
    )
    costs = [record["checkpoint_seconds"] for record in records if record["checkpoint_seconds"] > 0]
    # Include rank-zero log writes in profiled orchestration overhead.
    log_cost = max(0.0, loop - sum(record["step_wall_seconds"] for record in records)) / len(
        records
    )
    return TimingProfile(
        str(directory),
        run["config"],
        run["environment"],
        [record["step_seconds"] for record in records],
        [record["rank_step_seconds"] for record in records],
        [record["overhead_seconds"] + log_cost for record in records],
        statistics.mean(costs),
        startup,
        max(0.0, events[0]["elapsed_seconds"] - startup - loop),
    )
