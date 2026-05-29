"""
E2E tests for the F0 frontend pipeline.

Covers:
  - SecurityHeadersMiddleware injects CSP / X-Frame-Options / X-Content-Type-Options
    on every response (static, API, 404)
  - QueryRequest.session_id pattern validation rejects invalid inputs
  - Valid SESSION_ID (UUID) flows through query → job_id → stream

No real infrastructure — state is injected into app after create_app().
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import ASGITransport, AsyncClient

from tests.e2e.conftest import make_app, make_redis_mock


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_mq_mock() -> MagicMock:
    mq = MagicMock()
    mq.publish = AsyncMock(return_value="1-0")
    mq.ping = AsyncMock(return_value=True)
    mq.connect = AsyncMock()
    mq.disconnect = AsyncMock()
    return mq


def _make_redis_with_job(status: str = "queued", answer: str = "", error: str = "") -> MagicMock:
    redis = make_redis_mock()
    redis.client.hset = AsyncMock()
    redis.client.expire = AsyncMock()
    redis.client.hgetall = AsyncMock(
        return_value={"status": status, "answer": answer, "error": error}
    )
    return redis


# ── Security header assertions ────────────────────────────────────────────────

def _assert_security_headers(headers) -> None:
    assert "content-security-policy" in headers, "CSP header missing"
    assert "x-frame-options" in headers, "X-Frame-Options missing"
    assert "x-content-type-options" in headers, "X-Content-Type-Options missing"

    csp = headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "object-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert headers["x-frame-options"] == "DENY"
    assert headers["x-content-type-options"] == "nosniff"


# ── SecurityHeadersMiddleware coverage ───────────────────────────────────────

class TestSecurityHeadersOnStaticRoutes:
    async def test_root_has_security_headers(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/")
        _assert_security_headers(resp.headers)

    async def test_static_css_has_security_headers(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/static/style.css")
        _assert_security_headers(resp.headers)

    async def test_static_js_has_security_headers(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/static/app.js")
        _assert_security_headers(resp.headers)


class TestSecurityHeadersOnAPIRoutes:
    async def test_health_has_security_headers(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/health")
        _assert_security_headers(resp.headers)

    async def test_404_has_security_headers(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/does-not-exist")
        assert resp.status_code == 404
        _assert_security_headers(resp.headers)


# ── session_id validation ─────────────────────────────────────────────────────

class TestQueryRequestSessionIdValidation:
    """POST /agent/query now enforces ^[a-zA-Z0-9_-]{1,64}$ on session_id."""

    async def _post_query(self, session_id: str) -> int:
        app = make_app(mq=_make_mq_mock(), redis=_make_redis_with_job())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": session_id,
                "query": "hello",
            })
        return resp.status_code

    async def test_uuid_session_id_accepted(self):
        status = await self._post_query("550e8400-e29b-41d4-a716-446655440000")
        assert status == 202

    async def test_alphanumeric_session_id_accepted(self):
        status = await self._post_query("sess1")
        assert status == 202

    async def test_hyphenated_session_id_accepted(self):
        status = await self._post_query("sess-123-abc")
        assert status == 202

    async def test_underscore_session_id_accepted(self):
        status = await self._post_query("user_session_42")
        assert status == 202

    async def test_empty_session_id_rejected(self):
        status = await self._post_query("")
        assert status == 422

    async def test_session_id_with_space_rejected(self):
        status = await self._post_query("bad session")
        assert status == 422

    async def test_session_id_with_special_chars_rejected(self):
        status = await self._post_query("sess<script>")
        assert status == 422

    async def test_session_id_too_long_rejected(self):
        status = await self._post_query("a" * 65)
        assert status == 422

    async def test_session_id_64_chars_accepted(self):
        status = await self._post_query("a" * 64)
        assert status == 202


# ── Full UI-facing pipeline ───────────────────────────────────────────────────

class TestChatPipeline:
    """Simulate the flow a browser performs: POST /agent/query → GET /agent/stream/{job_id}"""

    async def test_query_returns_job_id(self):
        app = make_app(mq=_make_mq_mock(), redis=_make_redis_with_job())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.post("/agent/query", json={
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "summarise the document",
                "top_k": 5,
            })
        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    async def test_stream_delivers_done_answer(self):
        redis = _make_redis_with_job(status="done", answer="The document covers X.")
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            # Step 1: submit
            qresp = await c.post("/agent/query", json={
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "query": "what does it describe?",
            })
            job_id = qresp.json()["job_id"]

            # Step 2: stream
            sresp = await c.get(f"/agent/stream/{job_id}")

        assert sresp.status_code == 200
        assert sresp.headers["content-type"].startswith("text/event-stream")
        body = sresp.text
        assert "The document covers X." in body
        assert "[DONE]" in body

    async def test_stream_propagates_error_event(self):
        redis = _make_redis_with_job(status="error", error="LLM timeout")
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            qresp = await c.post("/agent/query", json={
                "session_id": "sess-abc",
                "query": "what is this?",
            })
            job_id = qresp.json()["job_id"]
            sresp = await c.get(f"/agent/stream/{job_id}")

        assert "[ERROR]" in sresp.text
        assert "LLM timeout" in sresp.text

    async def test_stream_has_security_headers(self):
        redis = _make_redis_with_job(status="done", answer="ok")
        app = make_app(mq=_make_mq_mock(), redis=redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            qresp = await c.post("/agent/query", json={
                "session_id": "sess-abc",
                "query": "test query",
            })
            job_id = qresp.json()["job_id"]
            sresp = await c.get(f"/agent/stream/{job_id}")

        _assert_security_headers(sresp.headers)


# ── SPA root + static verify CSP allows self-hosted assets ───────────────────

class TestCSPAllowsSelfHostedAssets:
    """Confirm the CSP allows the static assets the SPA page references."""

    async def test_script_src_allows_self(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/")
        csp = resp.headers["content-security-policy"]
        assert "script-src 'self'" in csp

    async def test_style_src_allows_self(self):
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/")
        csp = resp.headers["content-security-policy"]
        assert "style-src 'self'" in csp

    async def test_connect_src_allows_self_for_sse(self):
        """EventSource and fetch both use connect-src."""
        app = make_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            resp = await c.get("/")
        csp = resp.headers["content-security-policy"]
        assert "connect-src 'self'" in csp
