import json
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

from tensorfleet.controller.executor import run_training
from tensorfleet.training.checkpoint import load_checkpoint
from tensorfleet.training.device import backend_name, resolve_config
from tensorfleet.training.group import WorkerGroup


def capabilities(monkeypatch, *, cuda=False, count=0, mps=False, nccl=True):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    monkeypatch.setattr(dist, "is_nccl_available", lambda: nccl)
    monkeypatch.setattr(dist, "is_gloo_available", lambda: True)


@pytest.mark.parametrize(
    "cuda,count,mps,device,workers,backend",
    [
        (True, 4, True, "cuda", 2, "nccl"),
        (True, 1, False, "cuda", 1, None),
        (False, 0, True, "mps", 1, None),
        (False, 0, False, "cpu", 2, "gloo"),
    ],
)
def test_platform_resolution(monkeypatch, cuda, count, mps, device, workers, backend):
    capabilities(monkeypatch, cuda=cuda, count=count, mps=mps)
    config = resolve_config()
    assert (config.device, config.world_size, backend_name(config)) == (device, workers, backend)
    assert config.fail_rank == config.straggler_rank == workers - 1


def test_explicit_selection_never_falls_back(monkeypatch):
    capabilities(monkeypatch, mps=True)
    assert resolve_config(device="cpu").world_size == 2
    with pytest.raises(ValueError, match="CUDA"):
        resolve_config(device="cuda", world_size=2)
    with pytest.raises(ValueError, match="one worker"):
        resolve_config(device="mps", world_size=2)
    with pytest.raises(ValueError, match="rank"):
        resolve_config(device="mps", fail_rank=1)
    capabilities(monkeypatch)
    with pytest.raises(ValueError, match="Apple GPU"):
        resolve_config(device="mps")


def test_single_cuda_does_not_require_nccl(monkeypatch):
    capabilities(monkeypatch, cuda=True, count=1, nccl=False)
    assert backend_name(resolve_config()) is None


def test_single_process_does_not_initialize_collectives(monkeypatch):
    config = resolve_config(device="cpu", world_size=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("single-worker training must not use distributed collectives")

    for name in (
        "init_process_group",
        "destroy_process_group",
        "barrier",
        "all_reduce",
        "all_gather",
        "all_gather_object",
    ):
        monkeypatch.setattr(dist, name, forbidden)
    with WorkerGroup(config, 0, "unused") as group:
        model = torch.nn.Linear(2, 2)
        assert group.wrap(model) is model
        group.barrier()
        assert group.gather_states({"state": 1}) == [{"state": 1}]
        values = torch.tensor([2.0])
        group.sum_metrics(values)
        assert group.durations(values) == [2.0]


@pytest.mark.integration
def test_single_cpu_execution(tmp_path):
    config = resolve_config(device="cpu", world_size=1, steps=2)
    result = run_training(config, tmp_path)
    assert result["device"] == "cpu" and result["backend"] is None
    assert result["world_size"] == 1
    assert load_checkpoint(tmp_path / "checkpoint.pt", config)["next_step"] == 2


@pytest.mark.integration
@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires an Apple MPS GPU")
def test_real_mps_stochastic_recovery(tmp_path):
    config = resolve_config(
        device="mps", steps=5, stochastic_data=True, dropout=0.2, dataset_size=8, fail_step=3
    )
    baseline = tmp_path / "baseline"
    recovery = tmp_path / "recovery"
    run_training(replace(config, fail_step=None), baseline)
    result = run_training(config, recovery)
    assert (result["device"], result["world_size"], result["backend"], result["restarts"]) == (
        "mps",
        1,
        None,
        1,
    )
    clean = load_checkpoint(baseline / "checkpoint.pt", config)
    resumed = load_checkpoint(recovery / "checkpoint.pt", config)
    assert resumed["next_step"] == 5
    for name, value in clean["model"].items():
        torch.testing.assert_close(resumed["model"][name], value, rtol=1e-5, atol=1e-6)
    for key, state in clean["optimizer"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                resumed["optimizer"]["state"][key][name], value, rtol=1e-5, atol=1e-6
            )
    torch.testing.assert_close(
        resumed["rank_states"][0]["mps_rng"], clean["rank_states"][0]["mps_rng"], rtol=0, atol=0
    )
    for name in ("cursor", "epoch"):
        assert resumed["rank_states"][0]["data"][name] == clean["rank_states"][0]["data"][name]
    records = [
        json.loads(line) for line in (recovery / "metrics-attempt-1.jsonl").read_text().splitlines()
    ]
    assert records[0]["step"] == 2
    assert all(r["mps_allocated_bytes"] > 0 and r["step_seconds"] > 0 for r in records)
    metadata = json.loads((recovery / "run.json").read_text())
    assert metadata["execution"] == {"device": "mps", "backend": None, "world_size": 1}
