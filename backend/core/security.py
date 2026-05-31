"""
Security dependency seams: optional API-key auth and a simple rate limiter.

Both are *seams* — disabled by default so local dev is frictionless, and wired
as FastAPI dependencies so they can be attached per-router or per-route. The
rate limiter is in-process (single worker); swap its store for Redis to scale.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, Request

from core.config import Settings, get_settings
from core.exceptions import AuthError, RateLimitError


def _bearer_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization", "")
    return auth[7:] if auth.startswith("Bearer ") else None


def require_api_key(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """No-op when API_AUTH_TOKEN is empty; otherwise require a matching key."""
    expected = settings.API_AUTH_TOKEN
    if not expected:
        return
    provided = request.headers.get("X-API-Key") or _bearer_token(request)
    if provided != expected:
        raise AuthError("Missing or invalid API key.")


class RateLimiter:
    """Fixed-window per-client limiter. Single-process; a seam for a real one."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def reset(self) -> None:
        self._hits.clear()

    def __call__(
        self, request: Request, settings: Settings = Depends(get_settings)
    ) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        now = time.monotonic()
        key = request.client.host if request.client else "unknown"
        window = self._hits[key]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= settings.RATE_LIMIT_PER_MINUTE:
            raise RateLimitError("Rate limit exceeded. Please slow down.")
        window.append(now)


# Shared instance used as a dependency: Depends(rate_limiter)
rate_limiter = RateLimiter()
