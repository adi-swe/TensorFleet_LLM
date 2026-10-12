"""Launch real calibration and held-out runs, then evaluate predictions."""

import json
import math
from pathlib import Path

from tensorfleet.experiments.evaluate import validate_profile
from tensorfleet.experiments.failure_scenarios import calibration_scenarios
from tensorfleet.simulator.calibration import load_timing_profile


def run_validation_campaign(output, steps=30, repeats=3, delay_seconds=0.005, tolerance=0.25):
    from tensorfleet.config import TrainingConfig
    from tensorfleet.controller.executor import run_training

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
    for scenario, candidate in calibration_scenarios(config, delay_seconds):
        for index in range(repeats):
            run = output / f"{scenario}-{index}"
            run_training(candidate, run)
            holdouts.append(run)
    report = validate_profile(calibration, holdouts, tolerance)
    (output / "validation.json").write_text(json.dumps(report, indent=2))
    (output / "profile.json").write_text(
        json.dumps(load_timing_profile(calibration).to_dict(), indent=2)
    )
    return report
