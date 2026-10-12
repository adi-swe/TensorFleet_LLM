"""Immutable observations and decisions exchanged with orchestration policies."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Observation:
    attempt: int
    failed: bool
    max_restarts: int


@dataclass(frozen=True)
class Decision:
    action: Literal["restart", "stop"]
    reason: str
