"""Capability-based platform selection and explicit device validation."""

import torch
import torch.distributed as dist

from tensorfleet.config import TrainingConfig


def resolve_config(device="auto", world_size=None, fail_rank=None, straggler_rank=None, **kwargs):
    """Resolve user choices once; persist concrete settings in run/checkpoint metadata.

    Auto prefers CUDA, then Apple MPS, then CPU. Defaults are at most two CUDA
    workers, one MPS worker, or two CPU workers. Explicit requests never fall back.
    """
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if world_size is None:
        if device == "cuda":
            world_size = (
                min(2, max(1, torch.cuda.device_count())) if dist.is_nccl_available() else 1
            )
        else:
            world_size = 1 if device == "mps" else 2
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    last_rank = min(1, world_size - 1)
    config = TrainingConfig(
        device=device,
        world_size=world_size,
        fail_rank=last_rank if fail_rank is None else fail_rank,
        straggler_rank=last_rank if straggler_rank is None else straggler_rank,
        **kwargs,
    )
    validate_hardware(config)
    return config


def backend_name(config):
    if config.world_size == 1:
        return None
    return "nccl" if config.device == "cuda" else "gloo"


def validate_hardware(config):
    if config.device == "cuda":
        if not torch.cuda.is_available() or (
            config.world_size > 1 and not dist.is_nccl_available()
        ):
            raise ValueError(
                "CUDA execution requires a CUDA-enabled PyTorch build and NCCL for DDP"
            )
        if torch.cuda.device_count() < config.world_size:
            raise ValueError("CUDA DDP requires one visible GPU per worker")
    elif config.device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "MPS execution requires an available Apple GPU and an MPS-enabled PyTorch build"
            )
        if config.world_size != 1:
            raise ValueError(
                "MPS supports one worker; use CPU/Gloo for distributed training on this Mac"
            )
    elif config.world_size > 1 and not dist.is_gloo_available():
        raise ValueError("CPU distributed execution requires Gloo")


def worker_device(config, rank):
    if config.device == "cuda":
        torch.cuda.set_device(rank)
        return torch.device("cuda", rank)
    return torch.device(config.device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
