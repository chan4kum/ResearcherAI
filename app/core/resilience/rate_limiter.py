import threading
import time
import uuid
from collections import defaultdict
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.models.schemas import ErrorDetail, ErrorResponse

logger = get_logger("app.core.resilience.rate_limiter")


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter tracking request frequencies per client."""

    def __init__(
        self,
        requests_per_minute: int = 120,
        burst_limit: int = 30,
        window_seconds: float = 60.0,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.window_seconds = window_seconds
        self._history: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check_rate_limit(self, client_key: str) -> tuple[bool, int, float]:
        """Evaluate if the client request is permitted under active rate limits.

        Returns:
            (allowed: bool, remaining_requests: int, retry_after_seconds: float)
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds
        burst_cutoff = now - 1.0  # 1-second burst window

        with self._lock:
            timestamps = self._history[client_key]
            # Prune timestamps older than window
            valid_timestamps = [t for t in timestamps if t > cutoff]
            self._history[client_key] = valid_timestamps

            # 1. Evaluate total window limit
            if len(valid_timestamps) >= self.requests_per_minute:
                oldest = valid_timestamps[0]
                retry_after = max(1.0, round(self.window_seconds - (now - oldest), 1))
                return (False, 0, retry_after)

            # 2. Evaluate short-burst limit
            burst_count = sum(1 for t in valid_timestamps if t > burst_cutoff)
            if burst_count >= self.burst_limit:
                return (False, 0, 1.0)

            # Record this request
            valid_timestamps.append(now)
            remaining = max(0, self.requests_per_minute - len(valid_timestamps))
            return (True, remaining, 0.0)

    def reset(self) -> None:
        """Reset rate limiter state."""
        with self._lock:
            self._history.clear()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI Middleware enforcing client rate limits with standard 429 ErrorResponse."""

    def __init__(
        self,
        app: Any,
        settings: Settings | None = None,
        limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        super().__init__(app)
        self._settings = settings or get_settings()
        self._limiter = limiter or SlidingWindowRateLimiter(
            requests_per_minute=self._settings.rate_limit_per_minute,
            burst_limit=self._settings.rate_limit_burst,
        )
        self._exempt_paths = {
            "/health",
            "/ready",
            "/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/health",
            "/api/v1/health/ready",
            "/api/v1/health/live",
        }

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        if not self._settings.rate_limit_enabled or request.url.path in self._exempt_paths:
            res: Response = await call_next(request)
            return res

        # Derive client identity: API key, Bearer token, or IP address
        client_key = (
            request.headers.get("x-api-key")
            or request.headers.get("authorization")
            or (request.client.host if request.client else "unknown_client")
        )

        allowed, remaining, retry_after = self._limiter.check_rate_limit(client_key)

        if not allowed:
            request_id = getattr(request.state, "request_id", None) or str(uuid.uuid4())
            logger.warning(
                "rate_limit_exceeded",
                client=client_key,
                path=request.url.path,
                retry_after=retry_after,
                request_id=request_id,
            )
            error_response = ErrorResponse(
                error=ErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message=f"Rate limit exceeded. Please retry after {int(retry_after)}s.",
                    details={"retry_after_seconds": int(retry_after)},
                    request_id=request_id,
                )
            )
            error_json_res = JSONResponse(
                status_code=429,
                content=error_response.model_dump(),
            )
            error_json_res.headers["Retry-After"] = str(int(retry_after))
            error_json_res.headers["x-request-id"] = request_id
            return error_json_res

        response: Response = await call_next(request)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
