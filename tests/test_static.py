"""
Tests for static file serving and the SPA root route (GET /).
"""
import pytest
from httpx import ASGITransport, AsyncClient


def _get_app():
    from app import create_app
    return create_app()


class TestSPARoot:
    async def test_root_returns_200(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/")
        assert response.status_code == 200

    async def test_root_content_type_is_html(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/")
        assert "text/html" in response.headers["content-type"]

    async def test_root_body_contains_doctype(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/")
        assert "<!doctype html>" in response.text.lower()

    async def test_root_body_contains_app_js_link(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/")
        assert "/static/app.js" in response.text

    async def test_root_body_contains_style_css_link(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/")
        assert "/static/style.css" in response.text


class TestStaticAssets:
    async def test_style_css_returns_200(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/static/style.css")
        assert response.status_code == 200

    async def test_style_css_content_type(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/static/style.css")
        assert "text/css" in response.headers["content-type"]

    async def test_app_js_returns_200(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/static/app.js")
        assert response.status_code == 200

    async def test_app_js_content_type(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/static/app.js")
        assert "javascript" in response.headers["content-type"]

    async def test_unknown_static_path_returns_404(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/static/nonexistent.xyz")
        assert response.status_code == 404


class TestAPIRoutesUnaffected:
    async def test_health_still_returns_200(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/health")
        assert response.status_code == 200

    async def test_health_body_unchanged(self):
        app = _get_app()
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/health")
        assert response.json()["status"] == "ok"
