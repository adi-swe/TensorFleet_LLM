import json
import math
from dataclasses import replace

import pytest
import torch

from tensorfleet.config import TrainingConfig
from tensorfleet.controller.executor import run_training
from tensorfleet.controller.observations import Observation
from tensorfleet.controller.policies import RestartController
from tensorfleet.simulator.calibration import load_profile
from tensorfleet.simulator.engine import simulate
from tensorfleet.training.checkpoint import atomic_save, load_checkpoint
from tensorfleet.training.model import TinyTransformer, synthetic_batch


def test_config_and_synthetic_batch():
    with pytest.raises(ValueError):
        TrainingConfig(d_model=31)
    with pytest.raises(ValueError):
        TrainingConfig(fail_step=10)
    config = TrainingConfig()
    x, y = synthetic_batch(config, 0, 0)
    assert torch.equal(y, (x + 1) % config.vocab_size)
    assert torch.equal(x, synthetic_batch(config, 0, 0)[0])
    assert not torch.equal(x, synthetic_batch(config, 0, 1)[0])
    model = TinyTransformer(config)
    assert model(x).shape == (config.batch_size, config.sequence_length, config.vocab_size)
    changed = x.clone()
    changed[:, -1] = (changed[:, -1] + 3) % config.vocab_size
    assert torch.equal(model(x)[:, :-1], model(changed)[:, :-1])


def test_checkpoint_validation_and_atomic_failure(tmp_path, monkeypatch):
    config = TrainingConfig()
    path = tmp_path / "checkpoint.pt"
    payload = {
        "version": 1,
        "config": config.compatibility(),
        "next_step": 2,
        "model": {"weight": torch.ones(2)},
    }
    atomic_save(payload, path)
    assert load_checkpoint(path, config)["next_step"] == 2
    with pytest.raises(ValueError, match="incompatible"):
        load_checkpoint(path, replace(config, batch_size=8))

    def fail(*args):
        raise OSError("disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        atomic_save({"next_step": 3}, path)
    assert load_checkpoint(path, config)["next_step"] == 2
    assert list(tmp_path.iterdir()) == [path]


def test_controller_budget():
    controller = RestartController()
    assert controller.decide(Observation(0, True, 1)).action == "restart"
    assert controller.decide(Observation(1, True, 1)).action == "stop"
    assert controller.decide(Observation(0, False, 1)).action == "stop"


def test_discrete_events_replay_and_profile(tmp_path):
    result = simulate([1, 2], 4, checkpoint_every=2, fail_step=3, restart_seconds=5)
    assert result.kind == "simulated"
    assert result.total_seconds == 12  # six normal seconds + one replay + five restart
    assert result.replayed_steps == 1
    assert [e["time"] for e in result.events] == sorted(e["time"] for e in result.events)
    assert simulate([2], 3).total_seconds == 6
    for durations in ([], [0], [-1], [math.nan]):
        with pytest.raises(ValueError):
            simulate(durations, 2)
    with pytest.raises(ValueError):
        load_profile(tmp_path)
    (tmp_path / "metrics-attempt-0.jsonl").write_text(
        json.dumps({"kind": "measured", "step": 0, "step_seconds": 2}) + "\n"
    )
    assert load_profile(tmp_path) == [2]


@pytest.mark.integration
def test_real_ddp_failure_restart_matches_uninterrupted(tmp_path):
    config = TrainingConfig(steps=5, checkpoint_every=2, fail_step=3)
    recovered = tmp_path / "recovered"
    clean = tmp_path / "clean"
    result = run_training(config, recovered)
    assert result["restarts"] == 1
    run_training(replace(config, fail_step=None), clean)
    recovered_state = load_checkpoint(recovered / "checkpoint.pt", config)
    clean_state = load_checkpoint(clean / "checkpoint.pt", config)
    assert recovered_state["next_step"] == clean_state["next_step"] == 5
    for key, value in clean_state["model"].items():
        torch.testing.assert_close(recovered_state["model"][key], value, rtol=0, atol=0)
    # DDP's averaged gradients should match one process using the global batch.
    torch.manual_seed(config.seed)
    reference = TinyTransformer(config)
    optimizer = torch.optim.AdamW(reference.parameters(), lr=config.learning_rate)
    for step in range(config.steps):
        batches = [synthetic_batch(config, step, rank) for rank in range(config.world_size)]
        inputs = torch.cat([batch[0] for batch in batches])
        targets = torch.cat([batch[1] for batch in batches])
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            reference(inputs).reshape(-1, config.vocab_size), targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
    # Adam amplifies near-zero attention key-bias rounding differences. Compare
    # moving gradient statistics and functional outputs across batch reduction orders.
    for key, value in optimizer.state_dict()["state"].items():
        for name in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                clean_state["optimizer"]["state"][key][name], value[name], rtol=1e-5, atol=1e-8
            )
    clean_model = TinyTransformer(config)
    clean_model.load_state_dict(clean_state["model"])
    torch.testing.assert_close(clean_model(inputs), reference(inputs), rtol=1e-5, atol=1e-5)
    for key, value in clean_state["optimizer"]["state"].items():
        for name, tensor in value.items():
            torch.testing.assert_close(
                recovered_state["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
            )
    events = [json.loads(line) for line in (recovered / "events.jsonl").read_text().splitlines()]
    assert events[0]["failed"] and 86 in events[0]["exit_codes"]
    assert events[1]["exit_codes"] == [0, 0]
    records = [
        json.loads(line)
        for line in (recovered / "metrics-attempt-1.jsonl").read_text().splitlines()
    ]
    assert records[0]["step"] == 2  # rollback to the last committed checkpoint
    for record in records:
        assert record["kind"] == "measured"
        assert math.isfinite(record["loss"])
        assert record["step_seconds"] > 0 and record["worker_rss_bytes_sum"] > 0
        assert record["tokens_per_second"] > 0
    assert len(load_profile(recovered)) == 5
    assert not (recovered / ".run.lock").exists()
    with pytest.raises(ValueError, match="fresh output"):
        run_training(config, recovered)


@pytest.mark.integration
def test_exhausted_restart_budget_and_resume(tmp_path):
    config = TrainingConfig(steps=2, fail_step=0, max_restarts=0)
    with pytest.raises(RuntimeError, match="worker group failed"):
        run_training(config, tmp_path)
    assert load_checkpoint(tmp_path / "checkpoint.pt", config)["next_step"] == 0
    resume = tmp_path / "resume"
    resume.mkdir()
    import shutil

    shutil.copy2(tmp_path / "checkpoint.pt", resume / "checkpoint.pt")
    result = run_training(replace(config, fail_step=None), resume)
    assert result["status"] == "completed"
    assert load_checkpoint(resume / "checkpoint.pt", config)["next_step"] == 2
