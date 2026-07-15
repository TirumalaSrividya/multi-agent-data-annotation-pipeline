"""Abstract base class shared by all agents: gives every agent a named
logger and a consistent `run()` entry point convention, without imposing
any particular interface on inputs/outputs (each agent's contract is
defined by the schemas it consumes/returns, see src/schemas.py)."""
from __future__ import annotations

import logging
from abc import ABC


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.name}")

    def _log_error(self, msg: str, exc: Exception) -> None:
        self.logger.error("%s: %s: %s", msg, type(exc).__name__, exc)
