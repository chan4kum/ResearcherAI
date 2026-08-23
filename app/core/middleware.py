import asyncio
import socket
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import HTTP_REQUEST_DURATION_SECONDS, HTTP_REQUESTS_TOTAL
from app.core.resilience.rate_limiter import RateLimitMiddleware
from app.models.schemas import ErrorDetail, ErrorResponse

import structlog

logger = get_logger("app.core.middleware")



class RequestIdMiddleware(BaseHTTPMiddleware):
    """Ensure every incoming HTTP request has a unique correlation ID and structured telemetry."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        client_ip = request.client.host if request.client else "unknown"

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            http_method=request.method,
            http_path=request.url.path,
            client_ip=client_ip,
        )

        start_time = time.perf_counter()
        try:
            response: Response = await call_next(request)
            process_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

            response.headers["x-request-id"] = request_id
            response.headers["x-process-time-ms"] = str(process_time_ms)
            response.headers["x-pod-name"] = socket.gethostname()

            HTTP_REQUESTS_TOTAL.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=str(response.status_code),
            ).inc()
            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe((time.perf_counter() - start_time))

            logger.info(
                "http_request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=process_time_ms,
                request_id=request_id,
                client_ip=client_ip,
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()




class TimeoutMiddleware(BaseHTTPMiddleware):
    """Enforce maximum request processing timeout."""

    def __init__(self, app: Any, timeout_seconds: float = 30.0) -> None:
        super().__init__(app)
        self._default_timeout = timeout_seconds

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
        timeout_header = request.headers.get("x-request-timeout")
        try:
            timeout_sec = float(timeout_header) if timeout_header else self._default_timeout
        except ValueError:
            timeout_sec = self._default_timeout

        try:
            return await asyncio.wait_for(call_next(request), timeout=timeout_sec)
        except TimeoutError:
            logger.warning(
                "http_request_timeout",
                path=request.url.path,
                timeout_seconds=timeout_sec,
            )
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="REQUEST_TIMEOUT",
                    message=f"Request processing timed out after {timeout_sec}s",
                    request_id=request_id,
                )
            )
            response = JSONResponse(
                status_code=504,
                content=error_response.model_dump(),
            )
            response.headers["x-request-id"] = request_id
            return response


def register_middlewares(app: FastAPI, settings: Settings | None = None) -> None:
    """Register application-level middlewares."""
    active_settings = settings or get_settings()

    # 1. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Rate Limiting Middleware
    if active_settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            settings=active_settings,
        )

    # 3. Timeout Middleware
    app.add_middleware(
        TimeoutMiddleware,
        timeout_seconds=active_settings.request_timeout_seconds,
    )

    # 4. Request ID Correlation Middleware
    app.add_middleware(RequestIdMiddleware)
