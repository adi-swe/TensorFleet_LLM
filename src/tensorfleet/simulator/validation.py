"""Held-out prediction errors with separate loop and process wall-clock scopes."""

import json
import math
import statistics
from pathlib import Path

from tensorfleet.simulator.engine import simulate
from tensorfleet.simulator.profile import load_timing_profile, read_records


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


def validate_profile(calibration, holdouts, tolerance=0.25):
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    profile = load_timing_profile(calibration)
    if profile.config.get("straggler_seconds", 0) > 0:
        raise ValueError("use an uncontended calibration profile to avoid double-counting delays")
    cases = []
    seen = {str(Path(calibration).resolve())}
    # Operational settings may differ. Model/data/backend/hardware must match.
    ignored = {
        "steps",
        "checkpoint_every",
        "fail_step",
        "fail_rank",
        "max_restarts",
        "attempt_timeout",
        "collective_timeout",
        "straggler_seconds",
        "straggler_rank",
        "straggler_start",
        "straggler_stop",
    }
    for directory in holdouts:
        path = Path(directory).resolve()
        if str(path) in seen:
            raise ValueError("validation requires distinct held-out runs")
        seen.add(str(path))
        other = load_timing_profile(path)
        if {k: v for k, v in profile.config.items() if k not in ignored} != {
            k: v for k, v in other.config.items() if k not in ignored
        } or profile.environment != other.environment:
            raise ValueError("held-out model/data/device/environment differs from calibration")
        result = predict(profile, other.config)
        predicted_loop = result.total_seconds - profile.startup_seconds - profile.teardown_seconds
        lifecycle = read_records(path / "lifecycle-attempt-0.jsonl")
        measured_loop = next(
            r["training_loop_seconds"] for r in lifecycle if r["event"] == "complete"
        )
        measured_total = read_records(path / "events.jsonl")[0]["elapsed_seconds"]
        cases.append(
            {
                "run": str(path),
                "prediction_kind": "simulated",
                "observation_kind": "measured",
                "predicted_loop_seconds": predicted_loop,
                "measured_loop_seconds": measured_loop,
                "loop_relative_error": abs(predicted_loop - measured_loop) / measured_loop,
                "predicted_total_seconds": result.total_seconds,
                "measured_total_seconds": measured_total,
                "total_relative_error": abs(result.total_seconds - measured_total) / measured_total,
            }
        )
    if not cases:
        raise ValueError("at least one held-out run is required")
    loop_errors = [c["loop_relative_error"] for c in cases]
    total_errors = [c["total_relative_error"] for c in cases]
    return {
        "kind": "validation",
        "calibration": profile.source,
        "tolerance": tolerance,
        "passed": all(error <= tolerance for error in loop_errors + total_errors),
        "loop_mean_absolute_relative_error": statistics.mean(loop_errors),
        "total_mean_absolute_relative_error": statistics.mean(total_errors),
        "max_loop_relative_error": max(loop_errors),
        "max_total_relative_error": max(total_errors),
        "cases": cases,
        "limitations": "No confidence bound from a single campaign; CPU service-slot contention "
        "and recovery delay are not calibrated by this single-attempt validation.",
    }


def run_validation_campaign(output, steps=30, repeats=3, delay_seconds=0.005, tolerance=0.25):
    from dataclasses import replace

    from tensorfleet.config import TrainingConfig
    from tensorfleet.orchestration.supervisor import run_training

    if type(repeats) is not int or repeats < 1 or type(steps) is not int or steps < 2:
        raise ValueError("campaign needs at least two steps and one repeat")
    if (
        not math.isfinite(tolerance)
        or tolerance <= 0
        or not math.isfinite(delay_seconds)
        or delay_seconds < 0
    ):
        raise ValueError("campaign requires positive tolerance and nonnegative finite delay")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = TrainingConfig(steps=steps)
    calibration = output / "calibration"
    run_training(config, calibration)
    holdouts = []
    for scenario in ("baseline", "checkpoint-every-step", "straggler"):
        for index in range(repeats):
            run = output / f"{scenario}-{index}"
            candidate = (
                replace(config, checkpoint_every=1)
                if scenario == "checkpoint-every-step"
                else replace(config, straggler_seconds=delay_seconds, straggler_start=steps // 3)
                if scenario == "straggler"
                else config
            )
            run_training(candidate, run)
            holdouts.append(run)
    report = validate_profile(calibration, holdouts, tolerance)
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    (output / "profile.json").write_text(
        json.dumps(load_timing_profile(calibration).to_dict(), indent=2)
    )
    return report
