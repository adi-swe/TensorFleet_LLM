"""Bounded offline checkpoint-cadence search; recommendations never execute actions."""

from dataclasses import asdict, dataclass

from tensorfleet.simulator.calibration import predict


@dataclass(frozen=True)
class Candidate:
    checkpoint_every: int
    predicted_seconds: float
    replayed_steps: int
    kind: str = "simulated"

    def to_dict(self):
        return asdict(self)


def search_checkpoint_candidates(profile, intervals, *, fail_step=None, max_candidates=128):
    """Rank explicit cadences under a fixed profile/configuration and optional failure."""
    if type(max_candidates) is not int or not 1 <= max_candidates <= 128:
        raise ValueError("candidate budget must be an integer between 1 and 128")
    candidates = []
    seen = set()
    for index, interval in enumerate(intervals):
        if index >= max_candidates:
            raise ValueError("checkpoint search exceeds candidate budget")
        if type(interval) is not int or interval < 1:
            raise ValueError("checkpoint candidates must be positive integers")
        if interval in seen:
            continue
        seen.add(interval)
        config = {**profile.config, "checkpoint_every": interval, "fail_step": fail_step}
        result = predict(profile, config)
        candidates.append(Candidate(interval, result.total_seconds, result.replayed_steps))
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    return sorted(
        candidates, key=lambda candidate: (candidate.predicted_seconds, candidate.checkpoint_every)
    )
