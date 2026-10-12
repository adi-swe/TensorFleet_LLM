"""Spawn, monitor, reap, and restart the whole local worker group."""

import json
import multiprocessing as mp
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import psutil

from tensorfleet.controller.observations import Observation
from tensorfleet.controller.policies import RestartController
from tensorfleet.controller.safety_validator import guard_decision
from tensorfleet.training.ddp_executor import train_worker
from tensorfleet.training.device import backend_name, validate_hardware


def run_training(config, output, controller=None):
    validate_hardware(config)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    controller = controller or RestartController()
    # Exclusive run ownership prevents concurrent checkpoint and metric writers.
    lock = output / ".run.lock"
    with lock.open("x"):
        pass
    try:
        if (output / "events.jsonl").exists() or any(output.glob("metrics-attempt-*.jsonl")):
            raise ValueError("use a fresh output directory; copy checkpoint.pt there to resume")
        import torch

        (output / "run.json").write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "execution": {
                        "device": config.device,
                        "backend": backend_name(config),
                        "world_size": config.world_size,
                    },
                    "environment": {
                        "torch": torch.__version__,
                        "platform": platform.platform(),
                        "machine": platform.machine(),
                        "cpu_count": psutil.cpu_count(),
                        "memory_bytes": psutil.virtual_memory().total,
                        "cuda_version": torch.version.cuda,
                        "cuda_devices": [
                            torch.cuda.get_device_name(rank) for rank in range(config.world_size)
                        ]
                        if config.device == "cuda"
                        else [],
                    },
                },
                indent=2,
            )
        )
        context = mp.get_context("spawn")
        for attempt in range(config.max_restarts + 1):
            with tempfile.TemporaryDirectory(prefix="tensorfleet-") as temporary:
                rendezvous = (Path(temporary) / "store").as_uri()
                began = time.monotonic()
                workers = [
                    context.Process(
                        target=train_worker,
                        args=(rank, config, str(output), rendezvous, attempt, began),
                    )
                    for rank in range(config.world_size)
                ]
                timed_out = False
                try:
                    for worker in workers:
                        worker.start()
                    while any(worker.is_alive() for worker in workers):
                        if any(worker.exitcode not in (None, 0) for worker in workers):
                            break
                        if time.monotonic() - began > config.attempt_timeout:
                            timed_out = True
                            break
                        time.sleep(0.05)
                finally:
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                    for worker in workers:
                        if worker.pid is not None:
                            worker.join(timeout=5)
                            if worker.is_alive():
                                worker.kill()
                                worker.join()
                codes = [worker.exitcode for worker in workers]
                failed = timed_out or any(code != 0 for code in codes)
                observation = Observation(attempt, failed, config.max_restarts)
                error = None
                decision = None
                if failed:
                    try:
                        decision = guard_decision(controller.decide(observation), observation)
                        error = getattr(controller, "last_error", None)
                    except Exception as rejected:
                        from tensorfleet.controller.observations import Decision

                        decision = Decision("stop", "controller decision rejected")
                        error = f"{type(rejected).__name__}: {rejected}"[:512]
                with (output / "events.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "kind": "measured",
                                "attempt": attempt,
                                "exit_codes": codes,
                                "failed": failed,
                                "timed_out": timed_out,
                                "elapsed_seconds": time.monotonic() - began,
                                "decision": asdict(decision) if decision else None,
                                "controller_error": error,
                            }
                        )
                        + "\n"
                    )
                if not failed:
                    return {
                        "status": "completed",
                        "device": config.device,
                        "backend": backend_name(config),
                        "world_size": config.world_size,
                        "restarts": attempt,
                        "steps": config.steps,
                        "output": str(output),
                    }
                if decision.action != "restart":
                    raise RuntimeError(f"worker group failed: exit codes {codes}")
        raise RuntimeError("worker group exhausted restart budget")
    finally:
        lock.unlink(missing_ok=True)
