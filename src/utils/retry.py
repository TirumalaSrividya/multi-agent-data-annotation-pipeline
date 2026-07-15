"""
Lightweight retry-with-exponential-backoff decorator used to make LLM API
calls resilient to transient network failures and provider rate limits,
without pulling in a heavy third-party dependency.
"""
from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger("utils.retry")


def retry_with_backoff(
    exceptions: Tuple[Type[BaseException], ...],
    max_retries: int = 5,
    base_delay_s: float = 1.5,
    max_delay_s: float = 30.0,
    jitter: bool = True,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(
                            "Giving up on %s after %d attempts: %s", fn.__name__, attempt, exc
                        )
                        raise
                    delay = min(base_delay_s * (2 ** (attempt - 1)), max_delay_s)
                    if jitter:
                        delay += random.uniform(0, base_delay_s)
                    logger.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.1fs",
                        fn.__name__, attempt, max_retries, exc, delay,
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


class RetryableError(Exception):
    """Raised internally to signal a transient error worth retrying."""


class FatalError(Exception):
    """Raised for errors that should never be retried (e.g. bad request)."""
