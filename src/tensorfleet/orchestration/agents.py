"""Guarded proposals from external agent commands; no shell or executable actions."""

import json
import math
import os
import signal
import subprocess
import tempfile
from dataclasses import asdict

from tensorfleet.orchestration.interfaces import Decision


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


class CommandAgentRuntime:
    """JSON observation on stdin, JSON proposal on stdout, finite deadline.

    The command is a user-selected local adapter (which may call an LLM provider).
    It runs with the user's privileges, not in a security sandbox. Only its returned
    decision passes the action guard; it must not perform executor actions itself.
    """

    def __init__(self, command, timeout=5.0, max_bytes=65536):
        if (
            not command
            or isinstance(command, str)
            or any(not isinstance(arg, str) for arg in command)
            or not math.isfinite(timeout)
            or timeout <= 0
            or max_bytes <= 0
        ):
            raise ValueError("agent command, timeout and response limit must be positive")
        self.command = list(command)
        self.timeout = timeout
        self.max_bytes = max_bytes

    def propose(self, observation):
        with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=output,
                stderr=errors,
                start_new_session=True,
            )
            try:
                process.communicate(json.dumps(asdict(observation)).encode(), timeout=self.timeout)
                if process.returncode != 0:
                    raise ValueError(f"agent adapter exited with code {process.returncode}")
                if output.tell() > self.max_bytes:
                    raise ValueError("agent response exceeds byte limit")
                output.seek(0)
                return json.loads(output.read(self.max_bytes).decode())
            finally:
                # Reap descendants as well, including adapters that time out.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


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
