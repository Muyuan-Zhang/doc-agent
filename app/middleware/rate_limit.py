"""IP-based rate limiting middleware using Redis atomic increment.

Limits each client IP to `settings.agent_rate_limit_rpm` requests per minute.
Returns 429 with a standard AppError JSON body on exceed.
"""
import json
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.config import settings

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
                body = json.dumps({"error": {"code": "RATE_LIMITED", "message": "Too many requests"}})
                return Response(content=body, status_code=429, media_type="application/json")
        except Exception as exc:
            logger.warning("Rate-limit check failed, allowing request: %s", exc)

        return await call_next(request)
