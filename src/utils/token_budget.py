"""Simple token-budget tracker so the Annotator agent stops calling the LLM
once a configured budget is exhausted, instead of running unbounded and
racking up cost/latency on an ever-growing unlabelled pool."""
from __future__ import annotations

import logging

logger = logging.getLogger("utils.token_budget")


class BudgetExceededError(Exception):
    """Raised when an operation would exceed the remaining token budget."""


class TokenBudgetTracker:
    def __init__(self, budget: int):
        self.budget = budget
        self.used = 0

    @property
    def remaining(self) -> int:
        return max(self.budget - self.used, 0)

    def has_budget(self, estimated_cost: int = 0) -> bool:
        return self.remaining > estimated_cost

    def consume(self, tokens: int) -> None:
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        self.used += tokens
        if self.used >= self.budget:
            logger.warning("Token budget exhausted: used=%d budget=%d", self.used, self.budget)

    def __repr__(self) -> str:
        return f"TokenBudgetTracker(used={self.used}/{self.budget}, remaining={self.remaining})"
