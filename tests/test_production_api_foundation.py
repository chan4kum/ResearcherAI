import asyncio
from typing import Any

import pytest
from app.config import Settings
from app.core.retry import async_retry, calculate_backoff, execute_with_retry
from app.main import create_app
from app.models.schemas import ErrorResponse, LivenessResponse, ReadinessResponse
from fastapi import APIRouter
from fastapi.testclient import TestClient


def test_valid_request_with_request_id_and_telemetry(settings: Settings) -> None:
    """Verify standard requests return 200 OK with x-request-id and process time headers."""
    app = create_app(settings=settings)
    client = TestClient(app)

    custom_id = "trace-correlation-id-999"
    response = client.get("/health", headers={"x-request-id": custom_id})

    assert response.status_code == 200
    assert response.headers.get("x-request-id") == custom_id
    assert "x-process-time-ms" in response.headers
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == settings.app_version
    assert data["environment"] == settings.app_env


def test_malformed_request_validation_error(settings: Settings) -> None:
    """Verify malformed request payloads return standard 422 ErrorResponse with field details."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # Malformed body missing required 'task' field
    response = client.post(
        "/api/v1/tasks",
        json={"wrong_key": "some value"},
        headers={"x-request-id": "req-malformed-001"},
    )

    assert response.status_code == 422
    data = response.json()
    error_res = ErrorResponse.model_validate(data)
    assert error_res.error.code == "VALIDATION_ERROR"
    assert error_res.error.message == "Request validation failed"
    assert error_res.error.request_id == "req-malformed-001"
    assert isinstance(error_res.error.details, list)


def test_internal_error_handling(settings: Settings) -> None:
    """Verify unhandled exceptions in route handlers return structured 500 ErrorResponse."""
    app = create_app(settings=settings)
    test_router = APIRouter()

    @test_router.get("/test-unhandled-crash")
    async def crash_endpoint() -> dict[str, Any]:
        raise RuntimeError("Simulated critical internal failure")

    app.include_router(test_router)
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get(
        "/test-unhandled-crash",
        headers={"x-request-id": "req-crash-500"},
    )

    assert response.status_code == 500
    data = response.json()
    error_res = ErrorResponse.model_validate(data)
    assert error_res.error.code == "INTERNAL_SERVER_ERROR"
    assert error_res.error.message == "An unexpected internal error occurred"
    assert error_res.error.request_id == "req-crash-500"


def test_timeout_handling_middleware(settings: Settings) -> None:
    """Verify slow requests exceeding timeout threshold return 504 REQUEST_TIMEOUT."""
    app = create_app(settings=settings)
    test_router = APIRouter()

    @test_router.get("/test-slow-operation")
    async def slow_endpoint() -> dict[str, str]:
        await asyncio.sleep(0.5)
        return {"status": "finished"}

    app.include_router(test_router)
    client = TestClient(app)

    # Request with 50ms timeout header
    response = client.get(
        "/test-slow-operation",
        headers={
            "x-request-id": "req-timeout-504",
            "x-request-timeout": "0.05",
        },
    )

    assert response.status_code == 504
    data = response.json()
    error_res = ErrorResponse.model_validate(data)
    assert error_res.error.code == "REQUEST_TIMEOUT"
    assert "timed out" in error_res.error.message.lower()
    assert error_res.error.request_id == "req-timeout-504"


def test_health_backward_compatibility(settings: Settings) -> None:
    """Verify /health and /api/v1/health remain backward compatible."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Root /health
    res1 = client.get("/health")
    assert res1.status_code == 200
    d1 = res1.json()
    assert d1["status"] == "ok"
    assert d1["version"] == settings.app_version
    assert d1["environment"] == settings.app_env

    # 2. Versioned /api/v1/health
    res2 = client.get("/api/v1/health")
    assert res2.status_code == 200
    d2 = res2.json()
    assert d2["status"] == "ok"
    assert d2["version"] == settings.app_version
    assert d2["environment"] == settings.app_env


def test_readiness_endpoint_healthy_and_unhealthy(settings: Settings) -> None:
    """Verify Kubernetes readiness probe (/ready and /health/ready) evaluates subsystem health."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Fully initialized app -> HTTP 200 ready
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    ready_res = ReadinessResponse.model_validate(data)
    assert ready_res.status == "ready"
    assert ready_res.ready is True
    assert "llm_service" in ready_res.checks
    assert "vector_repository" in ready_res.checks
    assert "database" in ready_res.checks
    assert ready_res.checks["llm_service"].status == "healthy"

    # Versioned endpoint check
    res_versioned = client.get("/api/v1/health/ready")
    assert res_versioned.status_code == 200
    assert res_versioned.json()["ready"] is True

    # 2. Uninitialized LLM component -> HTTP 503 not_ready
    app.state.llm_service = None
    res_unhealthy = client.get("/ready")
    assert res_unhealthy.status_code == 503
    data_unhealthy = res_unhealthy.json()
    ready_unhealthy = ReadinessResponse.model_validate(data_unhealthy)
    assert ready_unhealthy.status == "not_ready"
    assert ready_unhealthy.ready is False
    assert ready_unhealthy.checks["llm_service"].status == "unhealthy"


def test_liveness_endpoint(settings: Settings) -> None:
    """Verify Kubernetes liveness probe reports process heartbeat and uptime."""
    app = create_app(settings=settings)
    client = TestClient(app)

    # 1. Root /live
    res = client.get("/live")
    assert res.status_code == 200
    data = res.json()
    live_res = LivenessResponse.model_validate(data)
    assert live_res.status == "alive"
    assert live_res.live is True
    assert live_res.uptime_seconds >= 0.0
    assert live_res.version == settings.app_version
    assert live_res.environment == settings.app_env

    # 2. Versioned /api/v1/health/live
    res_versioned = client.get("/api/v1/health/live")
    assert res_versioned.status_code == 200
    data_versioned = res_versioned.json()
    assert data_versioned["status"] == "alive"
    assert data_versioned["live"] is True


@pytest.mark.asyncio
async def test_async_retry_policy_with_transient_failures() -> None:
    """Verify retry policy handles transient failures and recovers with backoff."""
    attempt_count = 0

    @async_retry(
        max_attempts=3,
        initial_delay=0.01,
        backoff_factor=1.5,
        retryable_exceptions=(ValueError,),
    )
    async def flaky_operation() -> str:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError(f"Transient error on attempt {attempt_count}")
        return "success_recovered"

    result = await flaky_operation()
    assert result == "success_recovered"
    assert attempt_count == 3


@pytest.mark.asyncio
async def test_async_retry_policy_exhaustion() -> None:
    """Verify retry policy exhausts attempts and propagates final exception."""
    attempt_count = 0

    async def always_failing() -> None:
        nonlocal attempt_count
        attempt_count += 1
        raise ConnectionError("Persistent network outage")

    with pytest.raises(ConnectionError, match="Persistent network outage"):
        await execute_with_retry(
            always_failing,
            max_attempts=3,
            initial_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )

    assert attempt_count == 3


def test_calculate_backoff_jitter() -> None:
    """Verify exponential backoff calculation within bounds."""
    b1 = calculate_backoff(attempt=1, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert b1 == 1.0

    b2 = calculate_backoff(attempt=2, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert b2 == 2.0

    b3 = calculate_backoff(attempt=3, initial_delay=1.0, backoff_factor=2.0, jitter=False)
    assert b3 == 4.0

    b_jitter = calculate_backoff(attempt=2, initial_delay=1.0, backoff_factor=2.0, jitter=True)
    assert 1.0 <= b_jitter <= 2.0
