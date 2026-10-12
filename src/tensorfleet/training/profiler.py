"""Measured step metrics and JSONL output, independent of optimizer execution."""

import json

import psutil
import torch


def write_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


class StepProfiler:
    def __init__(self, config, group, device, path, attempt):
        self.config = config
        self.group = group
        self.device = device
        self.path = path
        self.attempt = attempt

    def collect(self, loss, elapsed):
        device, group = self.device, self.group
        # MPS does not support float64. Single-worker metrics stay on CPU.
        metrics_device = torch.device("cpu") if device.type == "mps" else device
        values = torch.tensor(
            [
                loss.item(),
                psutil.Process().memory_info().rss,
                torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            ],
            dtype=torch.float64,
            device=metrics_device,
        )
        group.sum_metrics(values)
        duration = torch.tensor([elapsed], dtype=torch.float64, device=metrics_device)
        rank_seconds = group.durations(duration)
        return values, rank_seconds

    def record(self, step, measurements, wall, checkpoint_seconds):
        config, group, device = self.config, self.group, self.device
        values, rank_seconds = measurements
        if self.group.rank == 0:
            latency = max(rank_seconds)
            write_record(
                self.path,
                {
                    "kind": "measured",
                    "device": config.device,
                    "backend": group.backend,
                    "world_size": config.world_size,
                    "attempt": self.attempt,
                    "step": step,
                    "loss": values[0].item() / config.world_size,
                    "step_seconds": latency,
                    "rank_step_seconds": rank_seconds,
                    "step_wall_seconds": wall,
                    "checkpoint_seconds": checkpoint_seconds,
                    "overhead_seconds": max(0.0, wall - max(rank_seconds) - checkpoint_seconds),
                    "injected_delay_seconds": config.straggler_seconds
                    if step >= config.straggler_start
                    and (config.straggler_stop is None or step < config.straggler_stop)
                    else 0.0,
                    "tokens_per_second": config.world_size
                    * config.batch_size
                    * config.sequence_length
                    / latency,
                    "worker_rss_bytes_sum": int(values[1].item()),
                    "mps_allocated_bytes": torch.mps.current_allocated_memory()
                    if device.type == "mps"
                    else 0,
                    "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()
                    if device.type == "mps"
                    else 0,
                    "cuda_allocated_bytes_sum": int(values[2].item()),
                    "cuda_peak_allocated_bytes_sum": int(values[3].item()),
                },
            )
