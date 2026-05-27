"""Unit tests for exception hierarchy and handlers — P0."""
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.core.exceptions import (
    AppError,
    NotFoundError,
    ServiceUnavailableError,
    ValidationError as AppValidationError,
    register_exception_handlers,
)


def _make_test_app() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)
    return app


class TestAppErrorClasses:
    def test_app_error_defaults(self):
        err = AppError("something failed")
        assert err.status_code == 500
        assert err.code == "INTERNAL_ERROR"
        assert err.message == "something failed"

    def test_not_found_error_status(self):
        err = NotFoundError("resource missing")
        assert err.status_code == 404
        assert err.code == "NOT_FOUND"

    def test_validation_error_status(self):
        err = AppValidationError("bad input")
        assert err.status_code == 422
        assert err.code == "VALIDATION_ERROR"

    def test_service_unavailable_error_status(self):
        err = ServiceUnavailableError("service down")
        assert err.status_code == 503
        assert err.code == "SERVICE_UNAVAILABLE"

    def test_app_error_is_exception(self):
        assert issubclass(AppError, Exception)

    def test_subclasses_inherit_from_app_error(self):
        assert issubclass(NotFoundError, AppError)
        assert issubclass(AppValidationError, AppError)
        assert issubclass(ServiceUnavailableError, AppError)


class TestAppErrorHandler:
    async def test_app_error_returns_correct_http_status(self):
        app = _make_test_app()

        @app.get("/test")
        async def route():
            raise NotFoundError("item not found")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/test")
        assert response.status_code == 404

    async def test_app_error_body_has_error_code(self):
        app = _make_test_app()

        @app.get("/test")
        async def route():
            raise NotFoundError("item not found")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/test")
        assert response.json()["error"]["code"] == "NOT_FOUND"

    async def test_app_error_body_has_message(self):
        app = _make_test_app()

        @app.get("/test")
        async def route():
            raise ServiceUnavailableError("db offline")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/test")
        assert response.json()["error"]["message"] == "db offline"

    async def test_app_error_body_has_request_id_field(self):
        app = _make_test_app()

        @app.get("/test")
        async def route():
            raise NotFoundError("missing")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/test")
        assert "request_id" in response.json()["error"]

    async def test_service_unavailable_returns_503(self):
        app = _make_test_app()

        @app.get("/test")
        async def route():
            raise ServiceUnavailableError("down")

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.get("/test")
        assert response.status_code == 503


def _make_request() -> "Request":
    from starlette.requests import Request
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/crash",
        "query_string": b"",
        "headers": [],
        "asgi": {"version": "3.0"},
    }
    return Request(scope)


class TestUnhandledErrorHandler:
    # Starlette's ServerErrorMiddleware always re-raises after calling the
    # Exception handler (by design). Testing through HTTP transport would
    # require raise_server_exceptions=False which this httpx version doesn't
    # support on ASGITransport. Instead, we call the handler function directly.

    async def test_handler_returns_500(self):
        app = _make_test_app()
        handler = app.exception_handlers[Exception]
        response = await handler(_make_request(), ValueError("internal crash"))
        assert response.status_code == 500

    async def test_handler_returns_generic_message(self):
        """Exception details must NOT leak into the response body."""
        import json
        app = _make_test_app()
        handler = app.exception_handlers[Exception]
        response = await handler(_make_request(), ValueError("secret internal detail"))
        body = json.loads(response.body)
        assert body["error"]["message"] == "An unexpected error occurred"
        assert "secret internal detail" not in body["error"]["message"]

    async def test_handler_code_is_internal_error(self):
        import json
        app = _make_test_app()
        handler = app.exception_handlers[Exception]
        response = await handler(_make_request(), RuntimeError("crash"))
        body = json.loads(response.body)
        assert body["error"]["code"] == "INTERNAL_ERROR"

    async def test_handler_response_has_request_id_field(self):
        import json
        app = _make_test_app()
        handler = app.exception_handlers[Exception]
        response = await handler(_make_request(), RuntimeError("crash"))
        body = json.loads(response.body)
        assert "request_id" in body["error"]
