"""IP-based rate limiting middleware using Redis atomic increment.

Limits each client IP to `settings.agent_rate_limit_rpm` requests per minute.
Returns 429 with the standard AppError JSON envelope on exceed.
"""
import logging

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings
from app.core.exceptions import _error_body

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        redis = getattr(request.app.state, "redis", None)
        if redis is None:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        key = f"{{rate:{client_ip}}}:rpm"
        try:
            count = await redis.increment_with_ttl(key, ttl_seconds=60)
            if count > settings.agent_rate_limit_rpm:
                return JSONResponse(
                    status_code=429,
                    content=_error_body("RATE_LIMITED", "Too many requests"),
                )
        except Exception as exc:
            logger.warning("Rate-limit check failed, allowing request: %s", exc)

        return await call_next(request)
