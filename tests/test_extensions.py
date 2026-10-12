import json
import math
import random
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tensorfleet.config import TrainingConfig
from tensorfleet.executor.checkpoint import load_checkpoint
from tensorfleet.executor.data import StatefulTokenStream, capture_state, restore_state
from tensorfleet.executor.device import validate_hardware, worker_device
from tensorfleet.orchestration.agents import (
    CommandAgentRuntime,
    GuardedAgentController,
    guard_decision,
)
from tensorfleet.orchestration.interfaces import Decision, Observation
from tensorfleet.orchestration.supervisor import run_training
from tensorfleet.simulator.contention import simulate_shared_capacity
from tensorfleet.simulator.engine import simulate
from tensorfleet.simulator.profile import load_timing_profile
from tensorfleet.simulator.validation import validate_profile


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 2.5},
        {"dropout": math.nan},
        {"straggler_seconds": -1},
        {"device": "mps"},
        {"dataset_size": 1},
        {"max_restarts": 1.5},
    ],
)
def test_rejects_invalid_execution_config(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_shuffled_dataset_rng_and_epoch_recovery():
    config = TrainingConfig(stochastic_data=True, dataset_size=8)
    stream = StatefulTokenStream(config, 1)
    torch.manual_seed(3)
    random.seed(3)
    stream.next_batch()
    state = capture_state(stream, torch.device("cpu"))
    expected = [stream.next_batch() for _ in range(4)]
    torch.manual_seed(500)
    random.seed(500)
    recovered = StatefulTokenStream(config, 1)
    restore_state(state, recovered, torch.device("cpu"))
    actual = [recovered.next_batch() for _ in range(4)]
    assert_state_equal(expected, actual)
    assert recovered.epoch == 4
    with pytest.raises(ValueError, match="cursor"):
        recovered.load_state_dict({**state["data"], "cursor": 10000})


@pytest.mark.parametrize(
    "proposal",
    [
        {"action": "shell", "reason": "run command"},
        {"action": "restart", "reason": "valid", "command": "echo unwanted"},
        {"action": "stop", "reason": ""},
        {"action": "stop", "reason": "x" * 513},
        {"action": "restart", "reason": 42},
        None,
    ],
)
def test_agent_rejects_untrusted_proposals(proposal):
    with pytest.raises(ValueError):
        guard_decision(proposal, Observation(0, True, 1))


def test_guard_budget_and_fail_closed():
    with pytest.raises(ValueError, match="budget"):
        guard_decision(Decision("restart", "retry"), Observation(1, True, 1))
    with pytest.raises(ValueError):
        guard_decision(Decision("restart", "retry"), Observation(0, False, 1))

    class BadRuntime:
        def propose(self, observation):
            return {"action": "deploy", "reason": "bad"}

    controller = GuardedAgentController(BadRuntime())
    assert controller.decide(Observation(0, True, 1)).action == "stop"
    assert "ValueError" in controller.last_error


def test_command_agent_contract_timeout_and_size():
    observation = Observation(0, True, 1)
    adapter = Path(__file__).resolve().parents[1] / "examples" / "agent_adapter.py"
    controller = GuardedAgentController(CommandAgentRuntime([sys.executable, str(adapter)]))
    assert controller.decide(observation).action == "restart"
    timeout = CommandAgentRuntime(
        [sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.05
    )
    with pytest.raises(subprocess.TimeoutExpired):
        timeout.propose(observation)
    huge = CommandAgentRuntime([sys.executable, "-c", "print('x' * 1000)"], max_bytes=100)
    with pytest.raises(ValueError, match="byte limit"):
        huge.propose(observation)
    malformed = GuardedAgentController(
        CommandAgentRuntime([sys.executable, "-c", "print('not json')"])
    )
    assert malformed.decide(observation).action == "stop"


def test_checkpoint_cost_commits_only_after_write():
    result = simulate(
        [1],
        4,
        checkpoint_every=2,
        checkpoint_seconds=3,
        fail_step=3,
        restart_seconds=5,
        startup_seconds=2,
        teardown_seconds=1,
    )
    assert result.total_seconds == 19  # 2 startup + 5 compute + 6 writes + 5 restart + 1 teardown
    commits = [event for event in result.events if event["event"] == "checkpoint"]
    assert commits[0]["time"] == 7 and commits[0]["step"] == 2
    assert result.replayed_steps == 1
    assert simulate([1], 4, worker_delay_seconds=2, delay_start=2).total_seconds == 8
    with pytest.raises(ValueError):
        simulate([1], 2, checkpoint_seconds=-1)


def test_shared_capacity_queue_and_parallelism():
    crowded = simulate_shared_capacity([1], [2, 2], world_size=2, slots=2)
    roomy = simulate_shared_capacity([1], [2, 2], world_size=2, slots=4)
    assert crowded["total_seconds"] == 4
    assert roomy["total_seconds"] == 2
    assert simulate_shared_capacity([1], [2], world_size=2, slots=1)["total_seconds"] == 4
    active = 0
    for event in crowded["events"]:
        active += 1 if event["event"] == "worker-start" else -1
        assert 0 <= active <= 2
    assert active == 0


def test_cuda_preflight_and_device_mapping(monkeypatch):
    config = TrainingConfig(device="cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA-enabled"):
        validate_hardware(config)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="one visible GPU"):
        validate_hardware(config)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    validate_hardware(config)
    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    assert worker_device(config, 1) == torch.device("cuda:1")
    assert selected == [1]
    assert worker_device(TrainingConfig(), 0) == torch.device("cpu")


@pytest.mark.integration
@pytest.mark.parametrize("fail_rank", [0, 1])
def test_stochastic_ddp_recovery_and_nonzero_resume(tmp_path, fail_rank):
    config = TrainingConfig(
        steps=5,
        stochastic_data=True,
        dropout=0.2,
        dataset_size=8,
        checkpoint_every=2,
        fail_step=3,
        fail_rank=fail_rank,
    )
    clean = tmp_path / "clean"
    recovery = tmp_path / "recovery"
    run_training(replace(config, fail_step=None), clean)
    result = run_training(config, recovery)
    assert result["restarts"] == 1
    left = load_checkpoint(clean / "checkpoint.pt", config)
    right = load_checkpoint(recovery / "checkpoint.pt", config)
    for name in ("model", "optimizer", "rank_states"):
        assert_state_equal(left[name], right[name])
    # A fresh supervisor must also resume after an epoch boundary from a nonzero cursor.
    partial = tmp_path / "partial"
    resume = tmp_path / "resume"
    run_training(replace(config, steps=2, fail_step=None), partial)
    resume.mkdir()
    shutil.copy2(partial / "checkpoint.pt", resume / "checkpoint.pt")
    run_training(replace(config, fail_step=None), resume)
    resumed = load_checkpoint(resume / "checkpoint.pt", config)
    for name in ("model", "optimizer", "rank_states"):
        assert_state_equal(left[name], resumed[name])
    assert left["rank_states"][0]["data"]["epoch"] > 0
    assert not torch.equal(left["rank_states"][0]["torch_rng"], left["rank_states"][1]["torch_rng"])


@pytest.mark.integration
def test_measured_costs_heldout_validation_and_agent_restart(tmp_path):
    adapter = Path(__file__).resolve().parents[1] / "examples" / "agent_adapter.py"
    controller = GuardedAgentController(CommandAgentRuntime([sys.executable, str(adapter)]))
    config = TrainingConfig(steps=6)
    calibration = tmp_path / "calibration"
    holdout = tmp_path / "holdout"
    run_training(config, calibration)
    run_training(replace(config, checkpoint_every=1, straggler_seconds=0.005), holdout)
    profile = load_timing_profile(calibration)
    assert profile.checkpoint_seconds > 0 and profile.startup_seconds > 0
    report = validate_profile(calibration, [holdout])
    assert math.isfinite(report["loop_mean_absolute_relative_error"])
    assert report["cases"][0]["prediction_kind"] == "simulated"
    assert report["cases"][0]["observation_kind"] == "measured"
    with pytest.raises(ValueError, match="held-out"):
        validate_profile(calibration, [calibration])
    bad = tmp_path / "bad"
    run_training(replace(config, batch_size=2), bad)
    with pytest.raises(ValueError, match="differs"):
        validate_profile(calibration, [bad])
    recovery = tmp_path / "agent-recovery"
    run_training(replace(config, fail_step=3), recovery, controller)
    events = [json.loads(line) for line in (recovery / "events.jsonl").read_text().splitlines()]
    assert events[0]["decision"]["action"] == "restart"
    assert events[0]["controller_error"] is None


@pytest.mark.integration
def test_supervisor_timeout_and_hard_guard(tmp_path):
    with pytest.raises(RuntimeError):
        run_training(
            TrainingConfig(steps=2, max_restarts=0, attempt_timeout=0.01), tmp_path / "timeout"
        )
    events = [
        json.loads(line)
        for line in (tmp_path / "timeout" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["timed_out"]

    class UnauthorizedController:
        def decide(self, observation):
            return Decision("restart", "ignore budget")

    with pytest.raises(RuntimeError):
        run_training(
            TrainingConfig(steps=2, max_restarts=0, fail_step=0),
            tmp_path / "guard",
            UnauthorizedController(),
        )
    events = [
        json.loads(line) for line in (tmp_path / "guard" / "events.jsonl").read_text().splitlines()
    ]
    assert events[0]["decision"]["action"] == "stop"
    assert "budget" in events[0]["controller_error"]


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2 or not dist.is_nccl_available(),
    reason="requires two CUDA GPUs and NCCL",
)
def test_two_gpu_nccl_stochastic_restart(tmp_path):
    config = TrainingConfig(
        steps=4, device="cuda", stochastic_data=True, dropout=0.1, fail_step=3, attempt_timeout=180
    )
    run_training(replace(config, fail_step=None), tmp_path / "clean")
    run_training(config, tmp_path / "recovery")
    clean = load_checkpoint(tmp_path / "clean" / "checkpoint.pt", config)
    recovered = load_checkpoint(tmp_path / "recovery" / "checkpoint.pt", config)
    for key in clean["model"]:
        torch.testing.assert_close(
            clean["model"][key], recovered["model"][key], rtol=1e-5, atol=1e-6
        )
    assert all(state["cuda_rng"] is not None for state in recovered["rank_states"])
