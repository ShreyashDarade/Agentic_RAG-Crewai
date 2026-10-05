"""Retry policy: full-jitter backoff, a per-client retry budget, bounded Retry-After (ADR-0009).

Every source of time and randomness is injectable so tests never sleep.
"""

from __future__ import annotations

import asyncio
import email.utils
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

__all__ = ["RetryPolicy", "TokenBucket", "parse_retry_after"]


class TokenBucket:
    """Retries spend tokens; a failing server therefore sees a bounded retry rate, not a storm."""

    def __init__(self, capacity: float, refill_per_second: float, clock: Callable[[], float]) -> None:
        self._capacity = capacity
        self._rate = refill_per_second
        self._clock = clock
        self._tokens = capacity
        self._stamp = clock()

    def try_take(self) -> bool:
        now = self._clock()
        self._tokens = min(self._capacity, self._tokens + (now - self._stamp) * self._rate)
        self._stamp = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


def parse_retry_after(value: str | None, *, now: Callable[[], float] = time.time) -> float | None:
    """``Retry-After`` as seconds from now: delay-seconds or an HTTP-date; anything else is ignored."""
    if value is None:
        return None
    value = value.strip()
    if value.isascii() and value.isdigit():  # str.isdigit() is also true for "²", which float() rejects
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - now())


@dataclass
class RetryPolicy:
    max_retries: int = 2
    base_delay: float = 0.5
    max_delay: float = 8.0
    retry_after_cap: float = 60.0
    bucket_capacity: float = 10.0
    bucket_refill_per_second: float = 1.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    uniform: Callable[[float, float], float] = random.uniform
    monotonic: Callable[[], float] = time.monotonic
    bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        if self.max_retries < 0 or self.base_delay < 0 or self.max_delay < self.base_delay:
            raise ValueError("invalid retry policy")
        self.bucket = TokenBucket(self.bucket_capacity, self.bucket_refill_per_second, self.monotonic)

    def backoff(self, attempt: int) -> float:
        """Full jitter: uniform in [0, min(cap, base * 2**attempt)]."""
        return self.uniform(0.0, min(self.max_delay, self.base_delay * (2**attempt)))
