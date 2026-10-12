# Architecture decision records

ADRs explain why TensorFleet adopts an architecture. The current component diagram
and operational contracts live in [the architecture document](../architecture.md).

## Index

- [ADR-0001: Platform-aware training with separate simulation and guarded orchestration](0001-platform-aware-training-architecture.md) — Accepted, 2026-10-12.

- [ADR-0002: Separate training, simulator, controller and experiment modules](0002-modular-package-boundaries.md) — Accepted, 2026-10-12.

## Record and maintenance convention

Use `NNNN-short-decision-title.md` with the next available number. Each record
contains a title, date, status, context, decision, considered alternatives,
consequences, validation and related implementation links.

Review architecture documentation after every meaningful implementation step as
specified in [AGENTS.md](../../AGENTS.md). Update the current diagram/contracts and
append a dated maintenance-log entry, including when architecture is unchanged.
Create a new ADR when a new decision is made. If a decision changes, mark the old
record superseded, link its replacement, and preserve its original rationale.
Correct factual errors or clarify existing records with an explicit amendment note.
