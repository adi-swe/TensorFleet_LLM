"""Policy interfaces and bounded rule/agent decisions."""

from typing import Protocol

from tensorfleet.controller.observations import Decision, Observation
from tensorfleet.controller.safety_validator import guard_decision


class Controller(Protocol):
    def decide(self, observation: Observation) -> Decision: ...


class AgentRuntime(Protocol):
    """A future LLM adapter returns a decision; the supervisor owns execution."""

    def propose(self, observation: Observation) -> Decision | dict[str, object]: ...


class RestartController:
    def decide(self, observation: Observation) -> Decision:
        restart = observation.failed and observation.attempt < observation.max_restarts
        return Decision("restart" if restart else "stop", "bounded whole-group recovery")


class GuardedAgentController:
    def __init__(self, runtime):
        self.runtime = runtime
        self.last_error = None

    def decide(self, observation):
        self.last_error = None
        try:
            return guard_decision(self.runtime.propose(observation), observation)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"[:512]
            return Decision("stop", "agent proposal rejected; stop safely")
