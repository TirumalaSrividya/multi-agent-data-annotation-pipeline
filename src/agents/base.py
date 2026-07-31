"""
Base class for all agents.
Provides shared logging functionality
"""
from __future__ import annotations

import logging
from abc import ABC


class BaseAgent(ABC):
    name: str = "base_agent"

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.name}")

    def _log_error(self, msg: str, exc: Exception) -> None:
        self.logger.error("%s: %s: %s", msg, type(exc).__name__, exc)
