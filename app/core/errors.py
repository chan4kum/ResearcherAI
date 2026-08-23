import asyncio
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger
from app.models.schemas import ErrorDetail, ErrorResponse

logger = get_logger("app.core.errors")


class AppException(Exception):
    """Base application exception with error codes and HTTP status codes."""

    def __init__(
        self,
        message: str,
        code: str = "INTERNAL_ERROR",
        status_code: int = 500,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details


class NotFoundException(AppException):
    """Resource not found exception."""

    def __init__(self, message: str = "Resource not found", details: Any = None) -> None:
        super().__init__(message=message, code="NOT_FOUND", status_code=404, details=details)


class BadRequestException(AppException):
    """Bad request exception."""

    def __init__(self, message: str = "Bad request", details: Any = None) -> None:
        super().__init__(message=message, code="BAD_REQUEST", status_code=400, details=details)


class RequestTimeoutException(AppException):
    """Request execution timeout exception."""

    def __init__(self, message: str = "Request execution timed out", details: Any = None) -> None:
        super().__init__(message=message, code="REQUEST_TIMEOUT", status_code=504, details=details)


class UnauthorizedException(AppException):
    """Authentication required or credentials invalid exception."""

    def __init__(self, message: str = "Authentication required", details: Any = None) -> None:
        super().__init__(message=message, code="UNAUTHORIZED", status_code=401, details=details)


class ForbiddenException(AppException):
    """Authorization permission denied exception."""

    def __init__(
        self, message: str = "Access forbidden: insufficient permissions", details: Any = None
    ) -> None:
        super().__init__(message=message, code="FORBIDDEN", status_code=403, details=details)


class RateLimitException(AppException):
    """Client request rate limit exceeded exception."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Please retry after the specified delay.",
        retry_after_seconds: int = 60,
        details: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details or {"retry_after_seconds": retry_after_seconds},
        )
        self.retry_after_seconds = retry_after_seconds


class BudgetExceededException(AppException):
    """Agent loop computation or tool call budget exceeded exception."""

    def __init__(
        self,
        message: str = "Agent execution budget exceeded to prevent runaway cost.",
        details: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            code="BUDGET_EXCEEDED",
            status_code=429,
            details=details,
        )


class CircuitBreakerOpenException(AppException):
    """Downstream dependency protected by open circuit breaker exception."""

    def __init__(
        self,
        message: str = "Service temporarily unavailable due to downstream protection.",
        details: Any = None,
    ) -> None:
        super().__init__(
            message=message,
            code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            details=details,
        )


class ServiceUnavailableException(AppException):
    """Service or dependency unavailable exception."""

    def __init__(
        self, message: str = "Service temporarily unavailable", details: Any = None
    ) -> None:
        super().__init__(
            message=message, code="SERVICE_UNAVAILABLE", status_code=503, details=details
        )


def _get_request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Handle custom application exceptions."""
    request_id = _get_request_id(request)
    logger.warning(
        "app_exception",
        code=exc.code,
        message=exc.message,
        status_code=exc.status_code,
        request_id=request_id,
    )
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=exc.code,
            message=exc.message,
            details=exc.details,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle Starlette / FastAPI HTTPExceptions."""
    request_id = _get_request_id(request)
    code = f"HTTP_{exc.status_code}"
    message = str(exc.detail) if exc.detail else "An HTTP error occurred"
    logger.warning(
        "http_exception",
        status_code=exc.status_code,
        message=message,
        request_id=request_id,
    )
    error_response = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic request validation errors."""
    request_id = _get_request_id(request)
    logger.info("validation_error", errors=exc.errors(), request_id=request_id)
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=exc.errors(),
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=422,
        content=error_response.model_dump(mode="json"),
    )


async def timeout_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle TimeoutError and asyncio.TimeoutError exceptions."""
    request_id = _get_request_id(request)
    logger.error("request_timeout", error=str(exc), request_id=request_id)
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="REQUEST_TIMEOUT",
            message="The request timed out before completing.",
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=504,
        content=error_response.model_dump(),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for unhandled exceptions."""
    request_id = _get_request_id(request)
    logger.error("unhandled_exception", error=str(exc), request_id=request_id, exc_info=True)
    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected internal error occurred",
            request_id=request_id,
        )
    )
    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all global exception handlers on the FastAPI application."""
    app.add_exception_handler(AppException, app_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(asyncio.TimeoutError, timeout_exception_handler)
    app.add_exception_handler(TimeoutError, timeout_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
