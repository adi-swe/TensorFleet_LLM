"""Runnable deterministic adapter example, not an LLM agent.

Replace the proposal-generation body with a provider call to use an LLM.
The supervisor validates the returned JSON and alone performs process restarts.
"""

import json
import sys

observation = json.load(sys.stdin)
restart = observation["failed"] and observation["attempt"] < observation["max_restarts"]
print(json.dumps({"action": "restart" if restart else "stop", "reason": "bounded adapter policy"}))
