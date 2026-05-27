"""
Tests for GET /health (liveness) and GET /health/ready (readiness).

Priority area 2:
- /health always returns 200 with {"status": "ok"}.
- /health/ready returns 200 when all four clients ping successfully.
- /health/ready returns 503 when any single client fails ping.
- /health/ready returns 503 when all four clients fail ping.
- /health/ready returns 503 for every individual client failure
  and the response body names the failing client correctly.
- /health/ready returns 200 even when ping raises an exception
  (the endpoint catches it and marks that check as error).
  Wait — re-reading the implementation: exceptions are caught and
  result in "error: <exc>", which causes 503. Covered below.
- Response body always contains "checks", "status", "kb_version".
"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Fixture: build a throwaway FastAPI app with mocked clients on state
# ---------------------------------------------------------------------------

def _make_ping_mock(returns: bool = True) -> AsyncMock:
    return AsyncMock(return_value=returns)


def _make_client(ping_ok: bool = True) -> AsyncMock:
    m = AsyncMock()
    m.ping = AsyncMock(return_value=ping_ok)
    return m


async def _get_app_with_state(mysql_ok=True, redis_ok=True, milvus_ok=True, mq_ok=True):
    """
    Build a FastAPI app (via create_app) and populate app.state
    with mocked clients, bypassing the lifespan so tests stay unit-level.
    """
    from app import create_app

    app = create_app()

    # Inject mocked clients directly onto state — no real connections.
    app.state.mysql = _make_client(mysql_ok)
    app.state.redis = _make_client(redis_ok)
    app.state.milvus = _make_client(milvus_ok)
    app.state.mq = _make_client(mq_ok)

    return app


# ---------------------------------------------------------------------------
# /health  — liveness
# ---------------------------------------------------------------------------

class TestLiveness:
    async def test_liveness_returns_200(self):
        from app import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200

    async def test_liveness_body_contains_status_ok(self):
        from app import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        body = response.json()
        assert body["status"] == "ok"

    async def test_liveness_body_contains_service_name(self):
        from app import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        body = response.json()
        assert "service" in body


# ---------------------------------------------------------------------------
# /health/ready — all healthy
# ---------------------------------------------------------------------------

class TestReadinessAllHealthy:
    async def test_all_healthy_returns_200(self):
        app = await _get_app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 200

    async def test_all_healthy_status_is_ok(self):
        app = await _get_app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["status"] == "ok"

    async def test_all_healthy_checks_all_ok(self):
        app = await _get_app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        assert checks == {"mysql": "ok", "redis": "ok", "milvus": "ok", "mq": "ok"}

    async def test_all_healthy_body_contains_kb_version(self):
        app = await _get_app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert "kb_version" in response.json()


# ---------------------------------------------------------------------------
# /health/ready — individual client failures → 503
# ---------------------------------------------------------------------------

class TestReadinessSingleClientFailure:
    @pytest.mark.parametrize("failing_client", ["mysql", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_returns_503(self, failing_client: str):
        """Each individual client failing ping independently causes 503."""
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    @pytest.mark.parametrize("failing_client", ["mysql", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_status_is_degraded(self, failing_client: str):
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["status"] == "degraded"

    @pytest.mark.parametrize("failing_client", ["mysql", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_names_failing_check(self, failing_client: str):
        """The failing client's check value must start with 'error:'."""
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        assert checks[failing_client].startswith("error:")

    @pytest.mark.parametrize("failing_client", ["mysql", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_other_checks_still_ok(self, failing_client: str):
        """Healthy clients still show 'ok' when one client fails."""
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        for name, value in checks.items():
            if name != failing_client:
                assert value == "ok", f"Expected {name}=ok, got {value!r}"


# ---------------------------------------------------------------------------
# /health/ready — all clients fail
# ---------------------------------------------------------------------------

class TestReadinessAllFailing:
    async def test_all_failing_returns_503(self):
        app = await _get_app_with_state(
            mysql_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    async def test_all_failing_status_is_degraded(self):
        app = await _get_app_with_state(
            mysql_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["status"] == "degraded"

    async def test_all_failing_all_checks_are_errors(self):
        app = await _get_app_with_state(
            mysql_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        for name, value in checks.items():
            assert value.startswith("error:"), f"Expected error for {name}, got {value!r}"


# ---------------------------------------------------------------------------
# /health/ready — ping raises an exception (not just returns False)
# ---------------------------------------------------------------------------

class TestReadinessPingException:
    async def test_ping_exception_returns_503(self):
        """If a client's ping() raises instead of returning False, still 503."""
        from app import create_app

        app = create_app()
        app.state.mysql = AsyncMock()
        app.state.mysql.ping = AsyncMock(side_effect=RuntimeError("connection refused"))
        app.state.redis = _make_client(True)
        app.state.milvus = _make_client(True)
        app.state.mq = _make_client(True)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    async def test_ping_exception_names_the_client_in_checks(self):
        from app import create_app

        app = create_app()
        app.state.mysql = AsyncMock()
        app.state.mysql.ping = AsyncMock(side_effect=RuntimeError("connection refused"))
        app.state.redis = _make_client(True)
        app.state.milvus = _make_client(True)
        app.state.mq = _make_client(True)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        assert checks["mysql"].startswith("error:")
        assert "connection refused" in checks["mysql"]
