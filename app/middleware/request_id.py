import re
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging_config import request_id_var, trace_id_var

_SAFE_ID = re.compile(r"^[a-zA-Z0-9\-_]{1,64}$")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        raw = request.headers.get("X-Request-ID", "")
        request_id = raw if _SAFE_ID.match(raw) else str(uuid.uuid4())
        # OpenTelemetry traceparent header — populated by OTLP propagator when integrated
        raw_trace = request.headers.get("X-Trace-ID", "")
        trace_id = raw_trace if _SAFE_ID.match(raw_trace) else ""

        req_token = request_id_var.set(request_id)
        trace_token = trace_id_var.set(trace_id)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(req_token)
            trace_id_var.reset(trace_token)
