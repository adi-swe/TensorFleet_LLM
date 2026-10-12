# ADR-0001: Platform-aware training with separate simulation and guarded orchestration

- **Status:** Accepted
- **Date:** 2026-10-12
- **Scope:** Current single-host TensorFleet implementation
- **Decision basis:** Phase 1 requirements and subsequent requests for stochastic
  recovery, guarded agents, CUDA/NCCL and automatic Apple GPU support
- **Related diagram:** [Current architecture](../architecture.md#component-boundaries)

## Context

TensorFleet is a research foundation for simulation-guided orchestration of model
training. It must execute real training and recover from worker failures while
keeping predicted performance distinguishable from measured behavior. Development
must work on an Apple Silicon Mac with 16 GB memory and remain useful on CPU-only
and NVIDIA GPU hosts without introducing cluster infrastructure.

Initial requirements established two-process CPU/Gloo training. Subsequent work
added stochastic state recovery, measured cost profiles, simulation validation and
guarded controllers. The platform-selection request requires using the Apple GPU
without pretending that the existing distributed backends support MPS collectives.

## Decision

1. **Separate executor, orchestration, simulator and agent adapter boundaries.**
   The supervisor owns worker lifecycle; the executor owns model updates and
   checkpoint state; the simulator consumes measured profiles. The benchmark
   command explicitly coordinates real calibration/hold-out runs and validation.
   Simulators do not launch workers or automatically control live training.

2. **Resolve devices from capabilities before launching workers.**
   CLI `auto` prefers CUDA, then MPS, then CPU. Default execution uses up to two
   visible CUDA GPUs when NCCL is available, one CUDA GPU otherwise, one MPS
   worker on Apple GPUs, or two CPU workers. Explicit requests are validated and
   never silently replaced. The resolved configuration is persisted. Python
   `TrainingConfig()` retains its two-worker CPU default; `resolve_config` provides
   automatic selection to Python callers.

3. **Use PyTorch DDP only for multi-worker execution.**
   CPU workers synchronize through Gloo and CUDA workers through NCCL. A shared
   `WorkerGroup` interface also supports ordinary single-worker CPU/CUDA/MPS
   execution with no distributed process group. MPS rejects multiple workers.
   There is no custom gradient synchronization or attempt to mix these backends.

4. **Recover the entire worker group from an atomic checkpoint.**
   The spawn-based supervisor detects exits/timeouts, reaps the group and enforces
   a bounded restart budget. Checkpoints contain model, optimizer, next step,
   compatible configuration and each rank's data/RNG state, including the selected
   accelerator RNG. Rank zero commits through temporary-file write, fsync, atomic
   replacement and directory fsync; distributed peers wait for commit. Restore RNG
   and data state after constructing the executor. Cross-device/world-size migration
   is outside the reproducible-resume contract.

5. **Treat agent output as a limited proposal.**
   Controllers may propose only `restart` or `stop`. The supervisor validates
   schema, state and budget independently; invalid proposals stop safely and are
   audited. A trusted command adapter exchanges JSON with a finite timeout and
   response-size check. This adapter runs with user permissions and is not a
   security sandbox. The included example is deterministic; a live LLM provider
   has not been integrated.

6. **Keep measurements and simulation provenance explicit.**
   Persist measured step, checkpoint, overhead and lifecycle costs with run/device
   metadata. Device timings synchronize accelerator work. Simulation results are
   labeled `simulated`; held-out validation reports training-loop and total-runtime
   error separately. The FIFO capacity model is an uncalibrated approximation.
   Injected straggler delays do not establish accuracy for real resource contention.

7. **Maintain the architecture alongside implementation.**
   After every meaningful implementation step, review this decision set, update
   the canonical Mermaid diagram and affected contracts, and append a dated
   maintenance note. Record unchanged architecture explicitly. New decisions get
   new ADRs; changed decisions supersede earlier records without erasing rationale.

## Alternatives considered

- **CPU-only execution:** simpler and reproducible for the original DDP baseline,
  but would leave the user's Apple GPU unused. It remains an explicit option.
- **One distributed backend for all devices:** would imply unsupported MPS/DDP
  behavior or introduce custom synchronization. Use a single-worker MPS path.
- **Custom gradient synchronization:** adds distributed correctness risk without
  supporting the present research objective. Use established PyTorch primitives.
- **Individual worker restart:** complicates collective membership and state
  consistency. Whole-group restart is sufficient for fixed-size local runs.
- **Agent-issued executable actions:** expands authority beyond the recovery
  contract. Restrict proposals and keep execution inside the supervisor.
- **A combined simulator/executor or unlabeled timing output:** obscures evidence
  and risks reporting predictions as measurements. Keep explicit component and
  artifact boundaries.
- **Kubernetes, cloud services or multi-node infrastructure:** unnecessary for the
  current local experiments. Revisit with a new ADR when scaling requires them.

## Consequences

The same training loop supports local CPU, CUDA and Apple GPU execution while
retaining a reproducible two-worker CPU baseline. Platform selection is visible
and failures do not silently change the chosen backend. Automatic worker selection
changes global batch size, so controlled comparisons must pin configuration.

Checkpoint state and atomic commits make stochastic rollback testable, at the cost
of checkpoint synchronization and replayed work. Fixed world size and trusted local
workers simplify the recovery boundary. Prefetch-aware datasets, additional RNG
sources and elastic membership require further decisions.

CPU and MPS paths have been exercised locally. The two-GPU CUDA integration test
exists but was skipped on the current Apple host; CUDA runtime behavior remains
unverified here. MPS accelerator recovery uses numerical tolerances for model and
optimizer state. Actual contention, failure-detection costs and GPU simulation
accuracy still need calibration. Current process supervision/agent lifecycle targets
macOS and Linux; Windows has not been validated.

## Validation and implementation references

- [Configuration](../../src/tensorfleet/config.py), [device selection](../../src/tensorfleet/executor/device.py)
  and [worker group](../../src/tensorfleet/executor/group.py)
- [Training executor](../../src/tensorfleet/executor/worker.py), [checkpointing](../../src/tensorfleet/executor/checkpoint.py)
  and [stateful data](../../src/tensorfleet/executor/data.py)
- [Supervisor](../../src/tensorfleet/orchestration/supervisor.py), [controller interfaces](../../src/tensorfleet/orchestration/interfaces.py)
  and [agent guards](../../src/tensorfleet/orchestration/agents.py)
- [Simulator](../../src/tensorfleet/simulator/engine.py), [capacity model](../../src/tensorfleet/simulator/contention.py)
  and [held-out validation](../../src/tensorfleet/simulator/validation.py)
- [Phase 1 tests](../../tests/test_phase1.py), [extension tests](../../tests/test_extensions.py)
  and [platform tests](../../tests/test_platforms.py)

Documentation maintenance is governed by [AGENTS.md](../../AGENTS.md). Current
verification and diagram reviews are recorded in the
[architecture maintenance log](../architecture.md#architecture-maintenance-log).
