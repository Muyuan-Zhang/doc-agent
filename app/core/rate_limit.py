from functools import partial
from typing import Callable

from fastapi import HTTPException, Request, status


def rate_limiter(key_suffix: str, limit: int, window_seconds: int) -> Callable:
    """Return a FastAPI dependency that enforces a per-client rate limit.

    Uses RedisClient.increment_with_ttl (atomic INCR + EXPIRE via Lua) so the
    counter is safe under concurrent requests and survives process restarts.
    """
    async def _check(request: Request) -> None:
        try:
            redis = request.app.state.redis
        except AttributeError:
            return  # state not ready; endpoint will return 503 itself
        # Use X-User-ID header when available; fall back to remote IP.
        client_id = request.headers.get("X-User-ID") or (
            request.client.host if request.client else "unknown"
        )
        # Hash-tag keeps the key on a single Redis cluster slot.
        key = f"{{rl:{client_id}}}:{key_suffix}"
        count = await redis.increment_with_ttl(key, window_seconds)
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per {window_seconds}s",
            )

    return _check
