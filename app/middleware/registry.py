from fastapi import FastAPI

from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIdMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware

# Insertion order = outermost-first; RequestIdMiddleware must stay innermost.
_MIDDLEWARE_STACK = [
    SecurityHeadersMiddleware,
    RateLimitMiddleware,
    RequestIdMiddleware,
]


def register_middlewares(app: FastAPI) -> None:
    # add_middleware wraps in reverse, so we reverse the list to preserve declared order.
    for middleware_cls in reversed(_MIDDLEWARE_STACK):
        app.add_middleware(middleware_cls)
