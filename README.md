# TensorFleet-LLM

Simulation-guided orchestration for model training. The CLI automatically selects
**CUDA, Apple MPS, or CPU** based on available hardware. It trains a small causal
Transformer, checkpoints and recovers workers, and independently simulates
training from measured step durations. Distributed runs use PyTorch DDP with
CPU/Gloo or CUDA/NCCL; Apple MPS uses one supervised GPU worker.
The existing `tensorfleet_literature` evidence validator remains available.

## Project structure

Production code retains the standard `src/` layout; tests stay at repository root.

```text
src/tensorfleet/
├── training/
│   ├── ddp_executor.py
│   ├── checkpoint.py
│   ├── profiler.py
│   └── data.py, device.py, group.py, model.py
├── simulator/
│   ├── engine.py
│   ├── event_queue.py
│   ├── compute_model.py
│   ├── communication_model.py
│   ├── checkpoint_model.py
│   ├── calibration.py
│   └── contention.py
├── controller/
│   ├── observations.py
│   ├── policies.py
│   ├── candidate_search.py
│   ├── safety_validator.py
│   ├── executor.py
│   └── agent_runtime.py
├── experiments/
│   ├── runner.py
│   ├── failure_scenarios.py
│   └── evaluate.py
└── cli.py, config.py, __main__.py
tests/
```

`training` performs real work and records measurements; `controller` supervises
workers and validates policy proposals; `simulator` composes event and cost models;
`experiments` launches real campaigns and evaluates held-out observations.
See [ADR-0002](docs/adr/0002-modular-package-boundaries.md) for ownership and migration.

Python imports have moved from `tensorfleet.executor` to `tensorfleet.training`
and from `tensorfleet.orchestration` to `tensorfleet.controller`:

```python
from tensorfleet.training.device import resolve_config
from tensorfleet.controller.executor import run_training
from tensorfleet.simulator.calibration import load_timing_profile, predict
from tensorfleet.experiments.evaluate import validate_profile
from tensorfleet.experiments.runner import run_validation_campaign
```

CLI commands and checkpoint/metrics schemas remain compatible. The old Python
module paths are removed; no compatibility shims are installed.

## Quick start: macOS Apple Silicon

Use native arm64 Python 3.10 or newer (3.12/3.13 recommended for this foundation)
and a native PyTorch wheel. No Docker or cluster setup is required. An available
Apple GPU is selected automatically through PyTorch MPS.
From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -c "import torch; print('Apple GPU available:', torch.backends.mps.is_available())"
python -m pytest -q

# Selects the Apple GPU on an MPS-capable Mac:
tensorfleet train --output runs/auto --steps 10
# Explicit two-process distributed CPU runs:
tensorfleet train --device cpu --output runs/baseline --steps 10
tensorfleet train --device cpu --output runs/recovery --steps 10 --fail-step 5 --fail-rank 1
tensorfleet simulate --profile runs/baseline --steps 20
tensorfleet simulate --profile runs/baseline --steps 20 --fail-step 5 --restart-seconds 1
```

`python -m tensorfleet` is equivalent to `tensorfleet`. Each command needs a fresh
output directory. Failure steps are zero-based and injection happens only on the
first supervisor attempt. The default budget permits one whole-group restart.
To demonstrate exhausted recovery:

```bash
tensorfleet train --output runs/exhausted --steps 4 --fail-step 1 --max-restarts 0
```

Resume in a fresh directory with the last committed checkpoint:

```bash
mkdir -p runs/resumed
cp runs/recovery/checkpoint.pt runs/resumed/checkpoint.pt
tensorfleet train --device cpu --output runs/resumed --steps 20
```

A MacBook Air M4 with 16 GB memory can start with these deliberately small defaults:
32 hidden dimensions, one Transformer layer, four attention heads, batch size four
per worker, 16 tokens per sequence, vocabulary 64, and one CPU compute thread per
worker. The CLI uses one Apple GPU worker; explicit CPU/Gloo uses two processes.
These are development settings, not an M4 performance benchmark. Increase steps
before model size. GPU launch overhead can dominate a model this small.

## Platform selection

`train --device auto` is the CLI default. Selection occurs once before launch:

- NVIDIA CUDA available: up to two visible GPUs; NCCL/DDP when using two. A single
  GPU, or a build without NCCL, uses ordinary single-worker CUDA training.
- Apple MPS available: one Apple GPU worker with no distributed process group.
- Otherwise: two CPU workers using Gloo/DDP.

Explicit choices are `--device cpu`, `--device cuda`, and `--device mps`.
`--world-size` overrides the default where supported. MPS requires one worker;
requesting multiple MPS workers raises an error. Explicit unavailable devices
raise errors. Training never changes device after an execution failure.
Failure and straggler ranks default to rank 1 for multiple workers, rank 0 for one.
`run.json`, training metrics and the CLI result record the resolved device,
backend and worker count; single-worker runs report `backend: null`.

```bash
tensorfleet train --device mps --output runs/apple-gpu --steps 10 --fail-step 5
python -m pytest -q -m mps
```

Global batch size is `batch-size × world-size`, so automatic worker selection can
change training semantics. Pin device and world size for controlled comparisons.
Resume checkpoints with their original device, worker count and data settings.
The Python `TrainingConfig()` default remains CPU/Gloo for reproducible existing
experiments; use `tensorfleet.training.device.resolve_config(device="auto", ...)`
for capability-based selection in Python. The current supervisor and command-agent
lifecycle are supported on macOS/Linux; Windows has not been validated.

## Profiling, simulation accuracy and contention

Run an independent calibration plus three held-out repetitions for each scenario:

```bash
tensorfleet benchmark-simulation --output runs/accuracy --steps 100 --repeats 3
```

This campaign explicitly uses two CPU/Gloo workers and measures normal training, checkpointing every step, and a controlled 5 ms
straggler delay. `validation.json` reports training-loop and total-runtime errors
separately; `profile.json` contains measured cost components. The default 25%
error gate applies to every case and exits with code 2 if any case fails. These
sample errors are not statistical guarantees. Use a fresh output directory.

```bash
tensorfleet validate-simulation --calibration runs/accuracy/calibration \
  --holdout runs/accuracy/baseline-0 runs/accuracy/checkpoint-every-step-0
# Includes measured checkpoint, metrics, startup and teardown costs:
tensorfleet simulate --profile runs/accuracy/calibration --cost-profile --steps 100
# Abstract capacity queueing for two simultaneous jobs; not calibrated CPU cores:
tensorfleet simulate-contention --profile runs/accuracy/calibration \
  --job-steps 10 10 --world-size 2 --slots 2
```

The simple `simulate` mode still accepts legacy profiles and excludes lifecycle
costs unless `--cost-profile` is supplied. `--checkpoint-seconds` and
`--restart-seconds` override assumed costs. `--contention-factor` scales training
service duration. The shared-capacity model is explicitly **uncalibrated**;
injected delays model controlled stragglers rather than measured physical CPU
contention. Longer projections cycle the profile and need separate validation.

## Offline candidate search and cost models

Rank explicit checkpoint intervals using a completed calibration run:

```bash
tensorfleet search-candidates --profile runs/accuracy/calibration \
  --checkpoint-intervals 1 2 4 8 --fail-step 5
```

Results are labeled `simulated` and `applied: false`. Search is bounded to 128
input candidates and cannot change live training. Checkpoint frequency trades
write cost against work replayed after the specified failure.

The default work profile includes DDP communication. `communication_model.py`
prevents adding communication costs again. If separately measured isolated compute
and communication samples are available, the Python API can compose them:

```python
from tensorfleet.simulator.engine import simulate

# Illustrative assumed seconds; not measurements or a calibrated network model.
projection = simulate([0.01], steps=10, communication_durations=[0.002],
                      durations_include_communication=False)
```

`event_queue.py` provides stable chronological ordering; `compute_model.py` owns
profile cycling and explicit slowdown/delay assumptions; `checkpoint_model.py`
commits progress only after a modeled write finishes. The training profiler does
not yet isolate pure compute and communication times.

## Stochastic recovery

```bash
tensorfleet train --output runs/stochastic --steps 12 --stochastic-data \
  --dropout 0.2 --dataset-size 64 --fail-step 5
mkdir -p runs/stochastic-resume
cp runs/stochastic/checkpoint.pt runs/stochastic-resume/checkpoint.pt
tensorfleet train --output runs/stochastic-resume --steps 20 --stochastic-data \
  --dropout 0.2 --dataset-size 64
```

Checkpoints store each rank's shuffled dataset permutation/cursor/epoch, CPU and
Python RNG, and CUDA or MPS RNG when applicable. Tests verify bit-exact CPU model, optimizer and
rank-state recovery with dropout, augmentation, epoch changes and either rank
failing. MPS integration tests exercise stochastic recovery with numerical
tolerances for model and optimizer values and exact RNG restoration. This stream has no asynchronous prefetch or NumPy transforms.

## Guarded agent controllers

```bash
tensorfleet train --output runs/agent-recovery --steps 10 --fail-step 5 \
  --agent-command '.venv/bin/python examples/agent_adapter.py' --agent-timeout 5
```

The provided adapter is a deterministic executable example, not an LLM. Replace
its proposal-generation body with a provider request to connect an LLM. The
supervisor sends JSON observations on stdin, accepts only `restart`/`stop` with a
short reason, and enforces the hard restart budget independently. Malformed,
late, oversized or forbidden proposals stop safely and are audited. Executable
adapters run with your permissions and must be trusted; proposal guards do not
sandbox the adapter itself. No shell is used to invoke it.

## Optional CUDA/NCCL (Linux GPU host)

Install a CUDA-enabled PyTorch wheel for the host using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/),
then install this project. For two visible GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 tensorfleet train --device cuda --world-size 2 \
  --output runs/cuda --steps 20 --stochastic-data --dropout 0.1 --fail-step 5
python -m pytest -q -m cuda
```

Each rank owns one GPU. CUDA measurements synchronize kernels; checkpoints retain
rank CUDA RNG and optimizer state. Missing CUDA, NCCL for multi-GPU execution, or
too few GPUs fails before launch. The CUDA integration test skips on Apple Silicon; CUDA cannot be verified
on a MacBook Air. Numerical recovery across GPUs uses tolerances.

## Outputs and semantics

- `checkpoint.pt`: versioned model, AdamW and per-rank RNG/data state plus next step and compatible config.
  Writes use a temporary file, file fsync, atomic replacement, and directory fsync.
- `metrics-attempt-N.jsonl`: **measured** global mean loss, maximum worker step latency,
  global tokens/second, and sum of sampled worker RSS in bytes.
- `lifecycle-attempt-N.jsonl`: measured startup and training-loop duration.
- `run.json`: resolved device/backend/worker count, configuration and environment metadata.
- `events.jsonl`: attempt wall time, exit codes, timeout and audited controller decision.
- Simulation JSON: **simulated** elapsed seconds, replay count, and event trace.

Step latency includes batch creation, forward/backward, DDP synchronization, and
optimizer update. It excludes metrics collectives, checkpoint I/O, startup, and the
post-step barrier. Those costs are profiled separately; CUDA and MPS timing
include device synchronization. MPS reports sampled tensor-allocated and driver-allocated
memory separately; these are not peak memory and should not be added to RSS on
unified-memory machines. CUDA reports allocator memory separately. Throughput is global tokens divided by that maximum latency.
RSS is current process resident memory sampled after each update, not peak memory,
allocator memory, or physical system memory (shared pages can be counted twice).
Failed attempts can contain updates later rolled back; keep attempt IDs when
analyzing observations. The simulator chooses the latest measurement for each step.

## Design and verification

- [Architecture diagram and contracts](docs/architecture.md)
- [Architecture decision records](docs/adr/README.md)
- [ADR-0001: Platform-aware training architecture](docs/adr/0001-platform-aware-training-architecture.md)
- [ADR-0002: Modular package boundaries](docs/adr/0002-modular-package-boundaries.md)

Architecture documentation is reviewed after every implementation step under
[repository guidance](AGENTS.md); reviews are recorded in the architecture maintenance log.

See [architecture and limitations](docs/architecture.md) for checkpoint, agent and simulation contracts. Unit tests validate causal
masking, deterministic batches, checkpoint compatibility and failed writes,
controller budgets, and simulation event accounting. Integration tests exercise
real two-worker DDP, forced failure, whole-group teardown/restart, exact model and
optimizer recovery, and checkpoint restoration in a new run.

```bash
python -m pytest -q -m 'not integration'  # fast units
python -m pytest -q -m integration        # real workers
python -m ruff check src tests examples
```

[CI](.github/workflows/tests.yml) runs the full suite on Linux with Python
3.10/3.11/3.12 and macOS with Python 3.13. It also checks formatting and installs
the built wheel to smoke-test both CLI entry points outside the source tree.
Linux runners use CPU PyTorch wheels; GPU tests skip when their hardware is
unavailable. NumPy is a runtime dependency for PyTorch's checkpoint-state object
collectives. JUnit test reports are retained as workflow artifacts.
The hosted macOS Python 3.12 lane stalled during process supervision; its cause
remains unresolved. Use Python 3.13 for the tested macOS setup.

## Literature evidence registry

The dependency-free validator checks completeness, URL format, year range,
record count, and duplicate citations. It does not infer scientific validity.

```bash
python -m tensorfleet_literature.validate path/to/matrix.jsonl
python -m tensorfleet_literature.validate path/to/matrix.jsonl --minimum 1 --maximum 15
```

Each JSONL record includes citation, title, year, source URL, hardware, model,
quantization, adapter placement/rank, training budget/seeds, memory methodology,
evaluation metrics, released artifacts, findings, and limitations.
