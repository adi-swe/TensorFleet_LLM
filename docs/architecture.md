# TensorFleet architecture and recovery contracts

## Component boundaries

The diagram represents the implemented system. Single-worker paths do not use DDP.
CUDA/NCCL is implemented but has not been run on this Apple host; the capacity
simulation remains uncalibrated. Decision rationale lives in
[ADR-0001](adr/0001-platform-aware-training-architecture.md).

```mermaid
flowchart TB
    CLI["CLI: train / simulate / validate / benchmark"]
    RESOLVE["Device resolution and configuration validation<br/>auto: CUDA → MPS → CPU"]

    subgraph ORCH["Orchestration"]
        SUP["Supervisor<br/>spawn, monitor, reap, bounded restart"]
        CTRL["RestartController or GuardedAgentController"]
        GUARD["Supervisor proposal guard<br/>restart or stop; hard budget"]
        ADAPTER["Optional trusted command agent<br/>JSON proposals; finite timeout"]
        SUP -->|"failure observation"| CTRL
        CTRL -->|"proposal"| GUARD
        GUARD -->|"validated decision"| SUP
        CTRL -.->|"request proposal"| ADAPTER
        ADAPTER -.->|"response"| CTRL
    end

    subgraph EXEC["Training executor: real work"]
        WORKER["Supervised worker process"]
        MODEL["TinyTransformer + AdamW<br/>synthetic or shuffled stateful data"]
        GROUP["WorkerGroup"]
        CPU["Multiple CPU workers<br/>PyTorch DDP / Gloo"]
        CUDA["Multiple CUDA workers<br/>PyTorch DDP / NCCL"]
        SINGLE["One CPU, CUDA or MPS worker<br/>ordinary model; no process group"]
        WORKER --> MODEL
        WORKER --> GROUP
        GROUP -->|"CPU; more than one worker"| CPU
        GROUP -->|"CUDA; more than one worker"| CUDA
        GROUP -->|"one worker; required for MPS"| SINGLE
    end

    subgraph FILES["Local persisted artifacts"]
        CKPT[("Atomic checkpoint<br/>model, optimizer, rank RNG/data, next step")]
        OBS[("Measured observations<br/>metrics, lifecycle, events, resolved configuration")]
    end

    subgraph SIM["Simulation and validation: separate from executor"]
        PROFILE["Measured profile loader"]
        ENGINE["Discrete-event simulator<br/>steps, checkpoints, rollback, lifecycle costs"]
        CAPACITY["Shared-capacity FIFO model<br/>uncalibrated service slots"]
        PRED[("Simulated results and event traces")]
        VALIDATE["Held-out validation<br/>loop and total-runtime prediction errors"]
        CAMPAIGN["Benchmark campaign<br/>real calibration and held-out runs"]
        PROFILE --> ENGINE
        PROFILE --> CAPACITY
        ENGINE --> PRED
        CAPACITY --> PRED
        ENGINE -->|"cost-model prediction"| VALIDATE
        CAMPAIGN -->|"distinct run paths"| VALIDATE
    end

    CLI -->|"train"| RESOLVE
    RESOLVE --> SUP
    SUP -->|"launch or restart whole group"| WORKER
    WORKER -->|"exit or timeout observed"| SUP
    WORKER -->|"rank-zero commit"| CKPT
    CKPT -->|"restore each rank before continuation"| WORKER
    WORKER -->|"step and lifecycle measurements"| OBS
    SUP -->|"attempt events and configuration"| OBS
    OBS --> PROFILE
    OBS -->|"held-out measurements"| VALIDATE
    CLI -->|"simulate / simulate-contention"| PROFILE
    CLI -->|"validate-simulation"| VALIDATE
    CLI -->|"benchmark-simulation"| CAMPAIGN
    CAMPAIGN -->|"launch real CPU calibration and held-out runs"| SUP
```

Solid arrows show implemented control or data flow; dashed arrows show the optional
agent adapter exchange. Simulated results are not currently fed into live control
decisions. The agent adapter is implemented; the example does not call an LLM.

Training, simulation, orchestration and agent adapters remain separate. PyTorch
DDP performs gradient synchronization. The simulator never launches training;
the benchmark command explicitly launches real runs before computing predictions.

## Executor and stochastic recovery

Each local worker is spawned with a fresh shared FileStore rendezvous per attempt.
The CLI resolves available capabilities in CUDA → MPS → CPU order. Defaults
are at most two CUDA workers (one without NCCL), one MPS worker, or two CPU workers.
The Python TrainingConfig default remains two CPU workers; resolve_config exposes
automatic selection for Python callers. Concrete settings are persisted.

The WorkerGroup abstraction initializes distributed state only for more than one
worker. Distributed CPU uses Gloo. Distributed CUDA uses NCCL, binds rank N to
visible GPU N before process-group construction, and uses CUDA tensors for metrics
collectives. Single-worker CPU/CUDA/MPS uses the ordinary model and optimizer.
MPS rejects world sizes above one and performs no distributed collectives. MPS
training synchronizes the device before measuring elapsed time; float64 metric
aggregation stays on CPU because MPS does not support float64 tensors. CUDA timings
synchronize the device at the measurement boundary. Hardware validation rejects
missing CUDA/NCCL for DDP or insufficient visible GPUs before spawning workers. There is
no device fallback after selection. Deterministic algorithms are required; unsupported
operations raise errors. CUDA recovery is tested with numerical tolerances, not
claimed bit-exact across hardware, software versions, or DDP reduction orders.

The default dataset remains stateless arithmetic token sequences. With
`--stochastic-data`, a finite deterministic corpus is sharded by rank and shuffled
at each epoch with a dedicated generator. The stream owns its permutation,
generator state, epoch and cursor. Batch augmentation draws from Python and global
PyTorch CPU RNG; optional Transformer dropout exercises model RNG. The stream can
cross epochs inside a batch. No DataLoader prefetch or NumPy RNG is used. MPS RNG is restored through
torch.mps.set_rng_state after construction, alongside CPU/Python and data state.

Version 2 checkpoints contain model, AdamW state, configuration, next global step,
and all ranks' CPU/Python RNG, data stream and optional CUDA or MPS RNG states. Rank
states are gathered only at checkpoint boundaries from trusted local workers.
Rank zero writes a temporary file, fsyncs, atomically replaces the checkpoint and
fsyncs the directory. A barrier prevents advancement before commit. On restore,
model/optimizer/DDP and data stream are constructed first, then each rank restores
its own RNG and stream state before consuming its next batch. Adding prefetch,
NumPy transformations or additional generators requires checkpointing their state.

Version 1 deterministic CPU checkpoints remain readable using Phase 1 defaults.
They cannot restore stochastic training. Configuration changes affecting
model/data/backend identity are rejected. Cross-device checkpoint migration is
not a supported reproducible-resume operation. Changing total steps, checkpoint
cadence, timeouts or straggler injection is allowed. Resume in a fresh output
directory by copying only `checkpoint.pt`.

## Supervisor and agent guard

The supervisor detects a worker exit or attempt deadline, terminates/reaps all
workers and consults its controller. Collective timeouts bound peer waits. Both
the controller and supervisor enforce the restart budget. Per-directory exclusive
locks prevent simultaneous writers. A hard-killed supervisor can leave a stale
lock; verify no workers remain before manually removing it.

The controller action set is deliberately `restart` or `stop`. The proposal guard
checks exact schema, a nonempty reason of at most 512 characters, failed-group
state, and the hard restart budget. It guards every controller, including rule
policies. Invalid actions or exceptions stop safely and are recorded in
`events.jsonl`; controllers cannot change model settings or execute shell actions
through a decision. `GuardedAgentController` calls an `AgentRuntime` adapter and
fails closed on rejected proposals.

`CommandAgentRuntime` sends an observation as JSON on stdin to an explicitly
selected trusted local command. The command returns JSON on stdout. It is invoked
without a shell, has a finite timeout and response-size check, and its process
group is reaped on completion/timeout. Response files avoid unbounded in-memory
capture, but the command is **not sandboxed**, has the user's permissions, and
output limits are checked after completion. A Python runtime implementation used
programmatically must provide its own finite deadline. The included deterministic
adapter is a runnable contract demonstration, not an LLM. A provider-specific
adapter may make an LLM request and return the same limited proposal schema.

## Measurement scope

Each measured step reports global mean loss, maximum per-rank training duration,
all rank durations, global tokens/second, summed sampled RSS, checkpoint time,
orchestration overhead and rank-zero wall time. Training duration covers batch
creation/augmentation, optional injected delay, device transfers, forward,
backward, DDP synchronization, optimizer update and accelerator synchronization.
Checkpoint time includes rank-state gathering, atomic saving and commit barrier.
Overhead covers metrics collectives and remaining barrier costs. Rank-zero log
writes are measured as training-loop residual when building a cost profile.
Summed RSS can double-count shared pages and is not peak RSS. CUDA allocated and
peak allocated bytes are summed per worker and are distinct from RSS and total GPU
memory. MPS reports current allocated tensor bytes and driver-allocated bytes,
not peaks. These cannot be added to RSS to estimate physical usage on Apple unified
memory. Model and optimizer state size remains fixed during these experiments.

Lifecycle records measure time from supervisor spawn to ready (including initial
checkpoint) and the training loop. Supervisor events measure each attempt's
wall time through worker teardown and decision. These are separate from individual
step metrics. Attempts containing rolled-back work retain their own files.

## Discrete events and contention

The event engine schedules step start/completion, checkpoint completion, failure
and restart. Checkpoint progress commits only when its modeled write completes.
Explicit checkpoint, overhead, startup, teardown and restart costs contribute to
elapsed time. Failure occurs once before a chosen step and replays uncommitted
steps. A scalar slowdown and step-range delay represent specified contention or
straggler scenarios. Restart cost inferred from a clean profile is an assumption;
failure detection and restart warm-up still need calibration with failed runs.

The separate FIFO service-slot model schedules concurrent jobs' worker service
and waits for all workers before each job advances. Capacity limits introduce
queueing. **Slots are abstract service capacity, not physical CPU cores.** Profiled
DDP durations include communication and are only a coarse proxy for service
demand. The capacity model is unit-tested but has not been calibrated against
actual overlapping jobs or CPU stress. Injected sleep delays validate a controlled
straggler scenario, not a general CPU contention model.

## Accuracy validation

Calibration requires a full successful single-attempt run from step zero.
Held-out runs must be distinct and match the model/data/device configuration and
recorded environment. Baseline, checkpoint-every-step and injected-straggler
campaigns execute independently and compare predictions with measured training
loop and total attempt time. Absolute relative error is reported for each scope;
a configurable gate checks every case and returns CLI exit code 2 on failure.
A low total-runtime error cannot hide a high training-loop error.

The cost model uses measured durations and mean checkpoint/overhead costs;
profiles cycle for projections longer than the calibration, including cold-step
samples. This is unsuitable for unvalidated long-horizon extrapolation. Hardware,
worker-count and model-size changes require new calibration. Campaign errors are
sample results, not confidence bounds or guaranteed prediction accuracy. Output
labels always distinguish `measured`, `simulated` and validation comparisons.

## Remaining milestones

- Calibrate actual overlapping jobs, pure CPU service demand, network and filesystem
  contention; separate cold start and steady state, and publish confidence intervals.
- Calibrate failure detection, teardown and restart warm-up using failed-run profiles.
- Validate CUDA/NCCL on a two-GPU Linux host and add a GPU CI lane.
- Add an actual provider adapter and offline agent evaluations before expanding actions.
- Add real datasets, prefetch-aware cursor recovery and eventually multi-node execution.

## Primary references

- [PyTorch DDP tutorial](https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch distributed and NCCL device selection](https://docs.pytorch.org/docs/stable/distributed.html)
- [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [MPS RNG restoration](https://docs.pytorch.org/docs/stable/generated/torch.mps.set_rng_state.html)
- [CUDA RNG state](https://docs.pytorch.org/docs/stable/cuda.html)
- [TorchElastic](https://docs.pytorch.org/docs/stable/elastic/run.html)
- [Ray Train](https://docs.ray.io/en/latest/train/train.html)
- [Vidur](https://github.com/microsoft/vidur)

TensorFleet takes inspiration from these systems; it does not claim to implement
Ray Train, TorchElastic membership, or Vidur's GPU/operator simulation model.


## Architecture maintenance log

Review this diagram, the affected contracts and the [ADR index](adr/README.md)
after every meaningful implementation step, as required by [AGENTS.md](../AGENTS.md).
Update the diagram when structure or behavior changes; otherwise record that it
was reviewed and remains accurate. Preserve decision history through superseding
ADRs when a decision changes.

- **2026-10-12 — Architecture documentation foundation:** recorded the existing
  platform-aware executor, stochastic checkpoint recovery, guarded orchestration
  and separate simulation/validation design in ADR-0001. Replaced the ASCII sketch
  with the canonical Mermaid diagram. Added persistent maintenance guidance and
  navigation links. Runtime architecture is unchanged by this documentation step.
  Verification: Mermaid parser accepted the diagram, all 29 local documentation
  links/anchors resolved, and `git diff --check` passed. Runtime tests were not
  rerun because this step changes documentation and contributor guidance only.

- **2026-10-12 — Phase 1 commit preparation:** reviewed ADR-0001 and the canonical
  diagram against the completed implementation; architecture remains unchanged.
  The latest runtime verification remains 41 tests passed and one CUDA test skipped,
  with successful MPS failure recovery and passing lint. Documentation-only work
  since that verification passed link, Mermaid and whitespace checks.
