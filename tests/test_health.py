"""
Tests for GET /health (liveness) and GET /health/ready (readiness).
"""
import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock


def _make_client(ping_ok: bool = True) -> AsyncMock:
    m = AsyncMock()
    m.ping = AsyncMock(return_value=ping_ok)
    return m


async def _get_app_with_state(postgres_ok=True, redis_ok=True, milvus_ok=True, mq_ok=True):
    from app import create_app
    app = create_app()
    app.state.postgres = _make_client(postgres_ok)
    app.state.redis = _make_client(redis_ok)
    app.state.milvus = _make_client(milvus_ok)
    app.state.mq = _make_client(mq_ok)
    return app


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
        assert response.json()["status"] == "ok"

    async def test_liveness_body_contains_service_name(self):
        from app import create_app
        app = create_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")
        assert "service" in response.json()


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
        assert checks == {"postgres": "ok", "redis": "ok", "milvus": "ok", "mq": "ok"}

    async def test_all_healthy_body_contains_kb_version(self):
        app = await _get_app_with_state()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert "kb_version" in response.json()


class TestReadinessSingleClientFailure:
    @pytest.mark.parametrize("failing_client", ["postgres", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_returns_503(self, failing_client: str):
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    @pytest.mark.parametrize("failing_client", ["postgres", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_status_is_degraded(self, failing_client: str):
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["status"] == "degraded"

    @pytest.mark.parametrize("failing_client", ["postgres", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_names_failing_check(self, failing_client: str):
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["checks"][failing_client].startswith("error:")

    @pytest.mark.parametrize("failing_client", ["postgres", "redis", "milvus", "mq"])
    async def test_single_client_ping_false_other_checks_still_ok(self, failing_client: str):
        kwargs = {f"{failing_client}_ok": False}
        app = await _get_app_with_state(**kwargs)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        for name, value in response.json()["checks"].items():
            if name != failing_client:
                assert value == "ok", f"Expected {name}=ok, got {value!r}"


class TestReadinessAllFailing:
    async def test_all_failing_returns_503(self):
        app = await _get_app_with_state(
            postgres_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    async def test_all_failing_status_is_degraded(self):
        app = await _get_app_with_state(
            postgres_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.json()["status"] == "degraded"

    async def test_all_failing_all_checks_are_errors(self):
        app = await _get_app_with_state(
            postgres_ok=False, redis_ok=False, milvus_ok=False, mq_ok=False
        )
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        for name, value in response.json()["checks"].items():
            assert value.startswith("error:"), f"Expected error for {name}, got {value!r}"


class TestReadinessPingException:
    async def test_ping_exception_returns_503(self):
        from app import create_app
        app = create_app()
        app.state.postgres = AsyncMock()
        app.state.postgres.ping = AsyncMock(side_effect=RuntimeError("connection refused"))
        app.state.redis = _make_client(True)
        app.state.milvus = _make_client(True)
        app.state.mq = _make_client(True)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        assert response.status_code == 503

    async def test_ping_exception_names_the_client_in_checks(self):
        from app import create_app
        app = create_app()
        app.state.postgres = AsyncMock()
        app.state.postgres.ping = AsyncMock(side_effect=RuntimeError("connection refused"))
        app.state.redis = _make_client(True)
        app.state.milvus = _make_client(True)
        app.state.mq = _make_client(True)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health/ready")
        checks = response.json()["checks"]
        assert checks["postgres"].startswith("error:")
        assert "connection refused" in checks["postgres"]
