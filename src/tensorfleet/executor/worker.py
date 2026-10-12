"""Profiled CPU, CUDA or MPS training with rank-local stochastic recovery."""

import json
import os
import random
import time
from pathlib import Path

import psutil
import torch

from tensorfleet.executor.checkpoint import atomic_save, load_checkpoint
from tensorfleet.executor.data import StatefulTokenStream, capture_state, restore_state
from tensorfleet.executor.device import synchronize, worker_device
from tensorfleet.executor.group import WorkerGroup
from tensorfleet.executor.model import TinyTransformer, synthetic_batch


def write_record(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")


def train_worker(rank, config, output, rendezvous, attempt, attempt_started=None):
    attempt_started = attempt_started if attempt_started is not None else time.monotonic()
    torch.set_num_threads(config.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = worker_device(config, rank)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    with WorkerGroup(config, rank, rendezvous) as group:
        model = TinyTransformer(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        checkpoint = Path(output) / "checkpoint.pt"
        start = 0
        payload = None
        if checkpoint.exists():
            payload = load_checkpoint(checkpoint, config)
            model.load_state_dict(payload["model"])
            optimizer.load_state_dict(payload["optimizer"])
            start = payload["next_step"]
        execution_model = group.wrap(model)
        stream = StatefulTokenStream(config, rank) if config.stochastic_data else None
        torch.manual_seed(config.seed + 30000 + rank)
        random.seed(config.seed + 40000 + rank)
        # Construction may consume random numbers: restore only after it finishes.
        if payload is not None and payload["version"] == 2:
            restore_state(payload["rank_states"][rank], stream, device)

        def save(next_step):
            began = time.perf_counter()
            states = group.gather_states(capture_state(stream, device))
            if rank == 0:
                atomic_save(
                    {
                        "version": 2,
                        "config": config.compatibility(),
                        "next_step": next_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "rank_states": states,
                    },
                    checkpoint,
                )
            group.barrier()
            synchronize(device)
            return time.perf_counter() - began

        initial_checkpoint = save(0) if payload is None else 0.0
        group.barrier()
        synchronize(device)
        lifecycle = Path(output) / f"lifecycle-attempt-{attempt}.jsonl"
        if rank == 0:
            write_record(
                lifecycle,
                {
                    "kind": "measured",
                    "event": "ready",
                    "attempt": attempt,
                    "next_step": start,
                    "startup_seconds": time.monotonic() - attempt_started,
                    "initial_checkpoint_seconds": initial_checkpoint,
                },
            )
        loop_started = time.perf_counter()
        metrics = Path(output) / f"metrics-attempt-{attempt}.jsonl"
        for step in range(start, config.steps):
            if attempt == 0 and step == config.fail_step and rank == config.fail_rank:
                write_record(
                    Path(output) / f"failure-attempt-{attempt}.jsonl",
                    {
                        "seconds_since_start": time.monotonic() - attempt_started,
                        "rank": rank,
                        "step": step,
                    },
                )
                os._exit(86)
            began = time.perf_counter()
            delay = 0.0
            if (
                rank == config.straggler_rank
                and step >= config.straggler_start
                and (config.straggler_stop is None or step < config.straggler_stop)
            ):
                delay = config.straggler_seconds
                time.sleep(delay) if delay else None
            inputs, targets = (
                stream.next_batch() if stream is not None else synthetic_batch(config, step, rank)
            )
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(
                execution_model(inputs).reshape(-1, config.vocab_size), targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            synchronize(device)
            elapsed = time.perf_counter() - began
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
            checkpoint_seconds = 0.0
            if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps:
                checkpoint_seconds = save(step + 1)
            else:
                group.barrier()
            wall = time.perf_counter() - began
            if rank == 0:
                latency = max(rank_seconds)
                write_record(
                    metrics,
                    {
                        "kind": "measured",
                        "device": config.device,
                        "backend": group.backend,
                        "world_size": config.world_size,
                        "attempt": attempt,
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
        if rank == 0:
            write_record(
                lifecycle,
                {
                    "kind": "measured",
                    "event": "complete",
                    "attempt": attempt,
                    "training_loop_seconds": time.perf_counter() - loop_started,
                },
            )
