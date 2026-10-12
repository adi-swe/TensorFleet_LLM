"""Durable, same-filesystem atomic checkpoint replacement."""

import os
import tempfile
from pathlib import Path

import torch


def atomic_save(payload: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: Path, config):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = config.compatibility()
    saved = dict(payload.get("config", {}))
    if payload.get("version") == 1:
        for name, default in {
            "device": "cpu",
            "dropout": 0.0,
            "stochastic_data": False,
            "dataset_size": 64,
        }.items():
            saved.setdefault(name, default)
    if payload.get("version") not in {1, 2} or saved != expected:
        raise ValueError("checkpoint version/configuration is incompatible")
    if payload["version"] == 1 and (config.stochastic_data or config.dropout > 0):
        raise ValueError("stochastic recovery requires a version 2 checkpoint")
    if payload["version"] == 2:
        states = payload.get("rank_states", [])
        if len(states) != config.world_size:
            raise ValueError("checkpoint rank state count mismatch")
        for state in states:
            if not isinstance(state, dict) or "torch_rng" not in state or "python_rng" not in state:
                raise ValueError("checkpoint is missing rank RNG state")
            if config.stochastic_data and state.get("data") is None:
                raise ValueError("checkpoint is missing dataset state")
            if config.device == "mps" and state.get("mps_rng") is None:
                raise ValueError("checkpoint is missing MPS RNG state")
            if config.device == "cuda" and state.get("cuda_rng") is None:
                raise ValueError("checkpoint is missing CUDA RNG state")
    if not 0 <= payload["next_step"] <= config.steps:
        raise ValueError("checkpoint progress exceeds requested steps")
    return payload
