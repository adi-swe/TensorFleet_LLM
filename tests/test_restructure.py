import json
import os
import subprocess
import sys
from itertools import repeat
from pathlib import Path

import pytest

from tensorfleet.config import TrainingConfig
from tensorfleet.controller.candidate_search import search_checkpoint_candidates
from tensorfleet.experiments.failure_scenarios import with_straggler, with_worker_failure
from tensorfleet.simulator.calibration import TimingProfile
from tensorfleet.simulator.checkpoint_model import CheckpointModel
from tensorfleet.simulator.communication_model import CommunicationModel
from tensorfleet.simulator.compute_model import ComputeModel
from tensorfleet.simulator.engine import simulate
from tensorfleet.simulator.event_queue import EventQueue


def profile():
    return TimingProfile(
        "test-profile",
        {"steps": 6, "checkpoint_every": 2},
        {},
        [1.0],
        [[1.0, 1.0]],
        [0.0],
        0.25,
        0.0,
        0.0,
    )


def test_queue_stable_ties_and_monotonic_clock():
    queue = EventQueue()
    queue.schedule(2.0, "second", 1)
    queue.schedule(1.0, "z-first", 0, job=3)
    queue.schedule(1.0, "a-next", 0, job=1)
    assert queue.pop().kind == "z-first"  # insertion order, not lexical event/job order
    assert queue.pop().kind == "a-next"
    with pytest.raises(ValueError, match="precede"):
        queue.schedule(0.5, "past", 0)
    assert queue.pop().time == 2.0
    assert not queue
    with pytest.raises(ValueError):
        queue.schedule(float("nan"), "bad", 0)


def test_cost_model_composition_prevents_double_counting():
    work = ComputeModel(
        [1.0, 2.0], contention_factor=2, worker_delay_seconds=0.5, delay_start=1, delay_stop=2
    )
    assert [work.duration(step) for step in range(3)] == [2, 4.5, 2]
    communication = CommunicationModel([0.1, 0.2])
    with pytest.raises(ValueError, match="already included"):
        communication.validate_profile(True)
    with pytest.raises(ValueError, match="already included"):
        simulate([1, 2], 3, communication_durations=[0.1])
    result = simulate(
        [1, 2],
        3,
        communication_durations=[0.1, 0.2],
        durations_include_communication=False,
        checkpoint_seconds=0.5,
    )
    assert result.total_seconds == pytest.approx(5.4)  # 4 work + .4 communication + 1 checkpoint
    assert result.kind == "simulated"
    with pytest.raises(ValueError):
        CommunicationModel([-1])


def test_checkpoint_commit_and_rollback_events():
    checkpoint = CheckpointModel(interval=2, write_seconds=3, restart_seconds=5)
    assert checkpoint.due(2, 5) and checkpoint.due(5, 5)
    assert not checkpoint.due(1, 5)
    assert checkpoint.committed_step == 0
    checkpoint.commit(2)
    assert checkpoint.lost_steps(3) == 1
    with pytest.raises(ValueError):
        checkpoint.commit(1)
    result = simulate(
        [1],
        4,
        checkpoint_seconds=3,
        fail_step=3,
        restart_seconds=5,
        startup_seconds=2,
        teardown_seconds=1,
    )
    assert result.total_seconds == 19
    assert result.replayed_steps == 1
    commits = [event for event in result.events if event["event"] == "checkpoint"]
    assert commits == [
        {"time": 7, "event": "checkpoint", "step": 2},
        {"time": 18, "event": "checkpoint", "step": 4},
    ]


def test_candidates_rank_tradeoff_without_mutating_profile():
    measured = profile()
    original = measured.to_dict()
    no_failure = search_checkpoint_candidates(measured, [1, 2, 3])
    with_failure = search_checkpoint_candidates(measured, [1, 2, 3], fail_step=5)
    assert [item.checkpoint_every for item in no_failure] == [3, 2, 1]
    assert [item.checkpoint_every for item in with_failure] == [1, 2, 3]
    assert all(item.kind == "simulated" for item in with_failure)
    assert measured.to_dict() == original
    assert len(search_checkpoint_candidates(measured, [2, 2])) == 1
    with pytest.raises(ValueError, match="budget"):
        search_checkpoint_candidates(measured, repeat(1), max_candidates=2)
    with pytest.raises(ValueError):
        search_checkpoint_candidates(measured, [])


def test_failure_scenarios_validate_configuration():
    base = TrainingConfig(steps=8)
    failed = with_worker_failure(base, step=3, rank=0)
    delayed = with_straggler(base, seconds=0.01, start=2, stop=4)
    assert (failed.fail_step, failed.fail_rank) == (3, 0)
    assert base.fail_step is None and base.straggler_seconds == 0
    assert (delayed.straggler_start, delayed.straggler_stop) == (2, 4)
    with pytest.raises(ValueError):
        with_worker_failure(base, step=8)
    with pytest.raises(ValueError):
        with_straggler(base, seconds=-1)


def test_offline_modules_do_not_import_training_or_torch():
    root = Path(__file__).resolve().parents[1]
    script = """
import importlib, sys
for name in ("engine", "event_queue", "compute_model", "communication_model",
             "checkpoint_model", "calibration", "contention"):
    importlib.import_module("tensorfleet.simulator." + name)
import tensorfleet.controller.candidate_search
import tensorfleet.experiments.evaluate
assert "torch" not in sys.modules
assert not any(name.startswith("tensorfleet.training") for name in sys.modules)
assert "tensorfleet.experiments.runner" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        timeout=10,
    )


def test_candidate_cli_uses_measured_profile(tmp_path):
    # A minimal complete measured fixture verifies entry-point wiring without timing noise.
    config = {"steps": 2, "checkpoint_every": 1}
    (tmp_path / "run.json").write_text(json.dumps({"config": config, "environment": {}}))
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"failed": False, "elapsed_seconds": 4.0}) + "\n"
    )
    (tmp_path / "lifecycle-attempt-0.jsonl").write_text(
        json.dumps({"event": "ready", "startup_seconds": 1})
        + "\n"
        + json.dumps({"event": "complete", "training_loop_seconds": 2.2})
        + "\n"
    )
    records = [
        {
            "kind": "measured",
            "step": step,
            "step_seconds": 1,
            "rank_step_seconds": [1, 1],
            "step_wall_seconds": 1.1,
            "checkpoint_seconds": 0.1,
            "overhead_seconds": 0,
        }
        for step in range(2)
    ]
    (tmp_path / "metrics-attempt-0.jsonl").write_text("\n".join(map(json.dumps, records)))
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tensorfleet",
            "search-candidates",
            "--profile",
            str(tmp_path),
            "--checkpoint-intervals",
            "1",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    data = json.loads(result.stdout)
    assert data["kind"] == "simulated" and data["applied"] is False
    assert data["candidates"][0]["checkpoint_every"] == 2
