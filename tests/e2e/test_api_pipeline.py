"""
E2E tests for the complete M0 ASGI pipeline.

Exercises: client → RequestIdMiddleware → exception handlers → router → response.
No real infrastructure — client state is injected after create_app() but before
any request, bypassing the lifespan connect logic.

Covers gaps not addressed by unit tests:
  - Middleware integration with real routes (X-Request-ID header lifecycle)
  - Error message safety for health degradation paths
  - Full pipeline shape for liveness and readiness responses
"""
import re

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


def _make_ping(ok: bool = True) -> AsyncMock:
    m = AsyncMock()
    m.ping = AsyncMock(return_value=ok)
    return m


def _app(*, postgres=True, redis=True, milvus=True, mq=True, llm=True):
    from app import create_app
    app = create_app()
    app.state.postgres = _make_ping(postgres)
    app.state.redis = _make_ping(redis)
    app.state.milvus = _make_ping(milvus)
    app.state.mq = _make_ping(mq)
    app.state.llm = _make_ping(llm)
    return app


def _app_with_exc(client_name: str, exc: Exception):
    app = _app()
    failing = getattr(app.state, client_name)
    failing.ping = AsyncMock(side_effect=exc)
    return app


# ---------------------------------------------------------------------------
# Request ID middleware — integration
# ---------------------------------------------------------------------------

class TestRequestIdPropagation:
    async def test_response_always_carries_request_id_header(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health")
        assert "x-request-id" in r.headers

    async def test_missing_header_generates_uuid(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health")
        assert _UUID_RE.match(r.headers["x-request-id"]), r.headers["x-request-id"]

    async def test_valid_header_is_echoed_unchanged(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health", headers={"X-Request-ID": "my-req-abc-123"})
        assert r.headers["x-request-id"] == "my-req-abc-123"

    async def test_unsafe_header_is_replaced_with_uuid(self):
        """Special chars outside [a-zA-Z0-9-_] must not be echoed."""
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health", headers={"X-Request-ID": "bad!chars@here"})
        rid = r.headers["x-request-id"]
        assert _UUID_RE.match(rid), f"expected UUID replacement, got {rid!r}"

    async def test_oversized_header_is_replaced_with_uuid(self):
        """Values exceeding 64 chars must not be echoed."""
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health", headers={"X-Request-ID": "a" * 65})
        rid = r.headers["x-request-id"]
        assert _UUID_RE.match(rid), f"expected UUID replacement, got {rid!r}"

    async def test_request_id_present_on_503_response(self):
        """Middleware must decorate error responses too."""
        app = _app(postgres=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert r.status_code == 503
        assert "x-request-id" in r.headers

    async def test_distinct_requests_get_distinct_ids(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r1 = await c.get("/health")
            r2 = await c.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]


# ---------------------------------------------------------------------------
# Error response safety — internal details must not leak
# ---------------------------------------------------------------------------

class TestErrorResponseSafety:
    async def test_ping_exception_detail_absent_from_response(self):
        secret = "postgresql+asyncpg://admin:s3cr3t@internal-host:5432/prod"
        app = _app_with_exc("postgres", RuntimeError(secret))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert secret not in r.text

    async def test_ping_exception_replaced_with_safe_message(self):
        app = _app_with_exc("redis", ConnectionError("redis://admin:pass@host"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert r.json()["checks"]["redis"] == "error: service unavailable"

    async def test_app_error_body_has_no_stack_trace(self):
        """AppError responses must not expose traceback text."""
        from app.core.exceptions import NotFoundError, register_exception_handlers
        from fastapi import FastAPI
        inner = FastAPI()
        register_exception_handlers(inner)

        @inner.get("/trigger")
        async def trigger():
            raise NotFoundError("secret-resource-id")

        async with AsyncClient(transport=ASGITransport(app=inner), base_url="http://test") as c:
            r = await c.get("/trigger")
        assert r.status_code == 404
        assert "Traceback" not in r.text
        assert "secret-resource-id" not in r.json()["error"].get("code", "")

    async def test_app_error_request_id_present_in_body(self):
        from app.core.exceptions import ServiceUnavailableError, register_exception_handlers
        from fastapi import FastAPI
        inner = FastAPI()
        register_exception_handlers(inner)

        @inner.get("/trigger")
        async def trigger():
            raise ServiceUnavailableError("db down")

        async with AsyncClient(transport=ASGITransport(app=inner), base_url="http://test") as c:
            r = await c.get("/trigger")
        assert "request_id" in r.json()["error"]


# ---------------------------------------------------------------------------
# Full pipeline — liveness
# ---------------------------------------------------------------------------

class TestLivenessPipeline:
    async def test_200_with_correct_body_and_middleware(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["service"] == "doc-agent"
        assert "x-request-id" in r.headers

    async def test_request_id_echoed_through_liveness(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health", headers={"X-Request-ID": "liveness-e2e"})
        assert r.headers["x-request-id"] == "liveness-e2e"


# ---------------------------------------------------------------------------
# Full pipeline — readiness
# ---------------------------------------------------------------------------

class TestReadinessPipeline:
    async def test_all_ok_returns_200_with_middleware(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert all(v == "ok" for v in body["checks"].values())
        assert "x-request-id" in r.headers

    async def test_single_degraded_client_returns_503(self):
        app = _app(milvus=False)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["status"] == "degraded"
        assert r.json()["checks"]["milvus"].startswith("error:")
        assert all(
            r.json()["checks"][n] == "ok"
            for n in ("postgres", "redis", "mq", "llm")
        )

    async def test_exc_in_ping_returns_503_with_safe_message(self):
        app = _app_with_exc("mq", RuntimeError("internal crash details"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert r.status_code == 503
        assert r.json()["checks"]["mq"] == "error: service unavailable"
        assert "internal crash details" not in r.text

    async def test_kb_version_present_in_readiness_body(self):
        app = _app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            r = await c.get("/health/ready")
        assert "kb_version" in r.json()
