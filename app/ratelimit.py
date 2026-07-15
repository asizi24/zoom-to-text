"""
In-process rate limiting — stdlib token bucket, keyed by client IP.

In-process is sufficient for the same reason the SSE hub and the cancellation
registry are (app/events.py, app/cancellation.py): exactly one uvicorn worker,
so every request shares this module's state. All access happens on the event
loop — no locking needed.

Usage (FastAPI dependency):

    @router.post("/thing", dependencies=[Depends(rate_limit("tasks", "rate_limit_tasks_per_minute"))])

The per-minute budget is read from settings lazily (at each scope's first use)
so tests can tune it; settings.rate_limit_enabled is consulted per request so
the whole mechanism can be toggled without touching the app. The test suite
disables it globally in conftest and re-enables it only in the rate-limit tests.
"""
import logging
import time

from fastapi import HTTPException, Request

from app.config import settings

logger = logging.getLogger(__name__)

# Safety cap on tracked clients. A small whitelisted user base will never get
# near this; if an open proxy floods us with spoofed IPs, the stalest bucket
# is evicted rather than growing without bound.
_MAX_BUCKETS_PER_SCOPE = 1024


class _TokenBucket:
    __slots__ = ("rate", "burst", "tokens", "last")

    def __init__(self, per_minute: int):
        self.rate = per_minute / 60.0   # tokens per second
        self.burst = float(per_minute)  # full budget available immediately
        self.tokens = self.burst
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._buckets: dict[str, _TokenBucket] = {}

    def allow(self, key: str) -> bool:
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= _MAX_BUCKETS_PER_SCOPE:
                stalest = min(self._buckets, key=lambda k: self._buckets[k].last)
                del self._buckets[stalest]
            bucket = self._buckets[key] = _TokenBucket(self.per_minute)
        return bucket.allow()


# One limiter per scope ("auth", "tasks", …), created on first request so the
# per-minute setting is read after any test-time monkeypatching.
_limiters: dict[str, RateLimiter] = {}


def reset() -> None:
    """Drop all limiter state — used by tests to start each case clean."""
    _limiters.clear()


def rate_limit(scope: str, per_minute_setting: str):
    """Build a FastAPI dependency enforcing the named per-minute budget."""

    async def dependency(request: Request) -> None:
        if not settings.rate_limit_enabled:
            return
        limiter = _limiters.get(scope)
        if limiter is None:
            limiter = _limiters[scope] = RateLimiter(getattr(settings, per_minute_setting))
        key = request.client.host if request.client else "unknown"
        if not limiter.allow(key):
            logger.warning(f"Rate limit hit: scope={scope} client={key}")
            raise HTTPException(
                status_code=429,
                detail="יותר מדי בקשות — המתן רגע ונסה שוב.",
            )

    return dependency
