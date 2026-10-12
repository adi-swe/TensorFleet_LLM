"""JSON command adapter for trusted external agent implementations."""

import json
import math
import os
import signal
import subprocess
import tempfile
from dataclasses import asdict


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
