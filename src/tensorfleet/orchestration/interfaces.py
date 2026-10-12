"""Controller boundaries shared by future rule policies and agent adapters."""

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class Observation:
    attempt: int
    failed: bool
    max_restarts: int


@dataclass(frozen=True)
class Decision:
    action: Literal["restart", "stop"]
    reason: str


class Controller(Protocol):
    def decide(self, observation: Observation) -> Decision: ...


class AgentRuntime(Protocol):
    """A future LLM adapter returns a decision; the supervisor owns execution."""

    def propose(self, observation: Observation) -> Decision | dict[str, object]: ...


class RestartController:
    def decide(self, observation: Observation) -> Decision:
        restart = observation.failed and observation.attempt < observation.max_restarts
        return Decision("restart" if restart else "stop", "bounded whole-group recovery")
