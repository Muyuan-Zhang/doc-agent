from fastapi import FastAPI

from app.middleware.request_id import RequestIdMiddleware

# Insertion order = outermost-first; RequestIdMiddleware must stay innermost.
# Add RateLimitMiddleware / AuthMiddleware above it when implementing M4.
_MIDDLEWARE_STACK = [
    # Placeholder: AuthMiddleware (M4)
    # Placeholder: RateLimitMiddleware (M4)
    RequestIdMiddleware,
]


def register_middlewares(app: FastAPI) -> None:
    # add_middleware wraps in reverse, so we reverse the list to preserve declared order.
    for middleware_cls in reversed(_MIDDLEWARE_STACK):
        app.add_middleware(middleware_cls)
