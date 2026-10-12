# ADR-0002: Separate training, simulator, controller and experiment modules

- **Status:** Accepted
- **Date:** 2026-10-12
- **Refines:** Module organization in [ADR-0001](0001-platform-aware-training-architecture.md)
- **Scope:** Package ownership and offline simulation composition

## Context

The user requested explicit training, simulator, controller and experiments layers.
Existing workers mixed profiling with optimizer execution, controller contracts and
validation lived together, and simulation event/cost accounting was embedded in one
function. Named modules must own working functionality rather than empty stubs.

## Decision

Keep the Python `src/` layout, with packages under `src/tensorfleet` and tests at
repository root. Use `training/ddp_executor.py` for numerical work,
`training/checkpoint.py` for persistence and `training/profiler.py` for measurements.
Retain data, model, device and worker-group helpers within training.

Separate controller observations, policies, proposal validation and process
execution. Retain the trusted external command adapter in `controller/agent_runtime.py`.
Candidate search evaluates checkpoint cadences offline using the simulator; it
must not apply its recommendations to a live executor automatically.

Simulation owns its stable event queue, profiled compute/work cost, communication
cost contract, checkpoint state/cost, calibration and event engine. Existing DDP
profiles already include communication; separate communication costs require
explicit isolated compute samples to prevent double counting. New cost inputs are
assumptions until measured; this refactor does not calibrate a network model.

Experiments own real campaign launches, reusable failure/straggler configurations
and held-out measured-versus-simulated evaluation. The simulator does not import
training or experiment runners. The experiment runner invokes the controller
executor explicitly. Tests continue to exercise real CPU and MPS recovery.

## Alternatives and consequences

Keeping monolithic modules would preserve import paths but obscure the requested
ownership. Empty files would satisfy only the appearance of the proposed layout.
Instead, move/extract existing behavior and provide working offline candidate and
cost-model APIs. Old executor/orchestration and simulator profile/validation imports
are replaced; this pre-1.0 refactor does not add compatibility shims. CLI commands
and checkpoint/metrics formats remain compatible. Supplemental modules are retained
where deleting them would mix responsibilities.

This ADR refines module organization without superseding the platform, recovery,
agent-authority or measurement-provenance decisions in ADR-0001. Future changes to
those decisions require a superseding ADR.

## Validation

Validate event ordering, cost composition, checkpoint rollback, candidate ranking,
import boundaries and existing CPU/MPS integration tests. Verify CLI entry points,
architecture links, Mermaid syntax and lint after the complete migration. Record
actual outcomes in the [architecture maintenance log](../architecture.md#architecture-maintenance-log).
