"""Validate limited controller proposals against the hard supervisor budget."""

from dataclasses import asdict

from tensorfleet.controller.observations import Decision


def guard_decision(proposal, observation):
    if isinstance(proposal, Decision):
        proposal = asdict(proposal)
    if (
        not isinstance(proposal, dict)
        or set(proposal) != {"action", "reason"}
        or proposal["action"] not in ("restart", "stop")
        or not isinstance(proposal["reason"], str)
        or not 1 <= len(proposal["reason"].strip()) <= 512
    ):
        raise ValueError("proposal must contain only restart/stop action and a short reason")
    if proposal["action"] == "restart" and (
        not observation.failed or observation.attempt >= observation.max_restarts
    ):
        raise ValueError("restart violates supervisor state or hard restart budget")
    return Decision(proposal["action"], proposal["reason"].strip())
