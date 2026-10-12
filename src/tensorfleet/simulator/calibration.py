"""Load measured profiles and configure cost-model predictions without launching training."""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from tensorfleet.simulator.engine import simulate


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


def predict(
    profile, config, *, contention_factor=1.0, restart_seconds=None, checkpoint_seconds=None
):
    if profile.config.get("straggler_seconds", 0) > 0:
        raise ValueError("use an uncontended profile to avoid double-counting delays")
    # The overhead mean is independent of checkpoint cadence; checkpoint cost is explicit.
    return simulate(
        profile.step_seconds,
        config["steps"],
        config["checkpoint_every"],
        config.get("fail_step"),
        restart_seconds=(
            profile.startup_seconds + profile.teardown_seconds
            if restart_seconds is None
            else restart_seconds
        ),
        checkpoint_seconds=(
            profile.checkpoint_seconds if checkpoint_seconds is None else checkpoint_seconds
        ),
        overhead_seconds=statistics.mean(profile.overhead_seconds),
        startup_seconds=profile.startup_seconds,
        teardown_seconds=profile.teardown_seconds,
        contention_factor=contention_factor,
        worker_delay_seconds=config.get("straggler_seconds", 0.0),
        delay_start=config.get("straggler_start", 0),
        delay_stop=config.get("straggler_stop"),
    )
