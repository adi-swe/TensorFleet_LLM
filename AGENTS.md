# TensorFleet repository guidance

## Architecture documentation after every implementation step

- After each meaningful implementation step (feature, fix, refactor, configuration
  change, or documentation milestone), review `docs/architecture.md` and the ADRs
  indexed in `docs/adr/README.md` before starting the next step or reporting completion.
  A step means a completed unit of work, not each tool call or optimizer update.
- Update the Mermaid diagram in `docs/architecture.md` in the same step whenever
  components, interfaces, control/data flow, execution platforms, or persistence change.
  Keep this Markdown block as the single canonical diagram; do not maintain a duplicate.
- Update the affected architecture contracts and append a dated maintenance-log entry
  describing the change and its verification. If the architecture is unchanged,
  explicitly record that review instead of making cosmetic diagram edits.
- Add a numbered ADR for a new architectural decision. For a changed decision,
  create a superseding ADR and cross-link both records. Preserve accepted decision
  history; corrections and clarifications may update an existing ADR with a note.
- Keep the ADR index and README links current. Use relative links within repository
  documentation. Check links, Mermaid syntax when changed, and `git diff --check`.
- Diagrams describe implemented behavior. Label future work and unverified paths
  explicitly. Never present simulations as measured results or imply that MPS uses DDP.
