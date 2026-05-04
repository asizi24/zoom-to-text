"""
Pure-Python in-memory sliding-window rate limiter.

Exposes a `limiter` object whose `.limit(key_func)` decorator mirrors the
slowapi interface used in routes.py — no external deps, no starlette Config
file reads (which break on Windows with non-ASCII .env files).
"""
import collections
import time
from functools import wraps

from fastapi import Request
from fastapi.responses import JSONResponse


class _InMemoryLimiter:
    """Sliding-window rate limiter keyed by client IP."""

    def __init__(self):
        # {ip: deque of monotonic timestamps}
        self._windows: dict[str, collections.deque] = {}

    def limit(self, limit_spec):
        """
        Decorator factory.  `limit_spec` is either a "N/minute" string or a
        callable(request) -> str.  Resolved at *request time* so tests can
        monkeypatch settings.rate_limit_per_minute freely.
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                spec = limit_spec(request) if callable(limit_spec) else limit_spec
                max_calls, window_seconds = self._parse_spec(spec)

                if max_calls > 0:
                    key = request.client.host if request.client else "unknown"
                    now = time.monotonic()
                    cutoff = now - window_seconds

                    dq = self._windows.setdefault(key, collections.deque())
                    while dq and dq[0] < cutoff:
                        dq.popleft()

                    if len(dq) >= max_calls:
                        return JSONResponse(
                            status_code=429,
                            content={"error": f"Rate limit exceeded: {spec}"},
                        )
                    dq.append(now)

                return await func(request, *args, **kwargs)
            return wrapper
        return decorator

    @staticmethod
    def _parse_spec(spec: str) -> tuple[int, float]:
        """Parse "N/minute" | "N/second" | "N/hour" → (N, window_seconds)."""
        parts = spec.split("/")
        n = int(parts[0])
        unit = parts[1].lower() if len(parts) > 1 else "minute"
        seconds = {"second": 1.0, "minute": 60.0, "hour": 3600.0}
        return n, seconds.get(unit, 60.0)


limiter = _InMemoryLimiter()
