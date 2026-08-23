import time

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from app.config import Settings
from app.models.schemas import (
    ComponentHealthCheck,
    HealthResponse,
    LivenessResponse,
    ReadinessResponse,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns the health status, current version, and environment of the service "
        "(backward compatible)."
    ),
)
async def get_health(request: Request) -> HealthResponse:
    """Return backward-compatible health status."""
    settings: Settings = getattr(request.app.state, "settings", None) or Settings()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        environment=settings.app_env,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Service Readiness probe",
    description=(
        "Evaluates whether internal dependencies and subsystems are initialized "
        "and ready to accept traffic."
    ),
)
async def get_readiness(request: Request) -> JSONResponse:
    """Evaluate subsystem readiness for Kubernetes readiness probes."""
    settings: Settings = getattr(request.app.state, "settings", None) or Settings()
    checks: dict[str, ComponentHealthCheck] = {}
    is_ready = True

    # 1. LLM Service Check
    llm_service = getattr(request.app.state, "llm_service", None)
    if llm_service is not None:
        checks["llm_service"] = ComponentHealthCheck(
            status="healthy", details="LLM adapter initialized"
        )
    else:
        checks["llm_service"] = ComponentHealthCheck(
            status="unhealthy", details="LLM service uninitialized"
        )
        is_ready = False

    # 2. Vector Repository Check
    vector_repo = getattr(request.app.state, "vector_repository", None)
    if vector_repo is not None:
        checks["vector_repository"] = ComponentHealthCheck(
            status="healthy", details="Vector repository accessible"
        )
    else:
        checks["vector_repository"] = ComponentHealthCheck(
            status="unhealthy", details="Vector repository uninitialized"
        )
        is_ready = False

    # 3. Database Manager Check
    db_manager = getattr(request.app.state, "db_manager", None)
    if db_manager is not None:
        checks["database"] = ComponentHealthCheck(
            status="healthy", details="Database manager operational"
        )
    else:
        checks["database"] = ComponentHealthCheck(
            status="unhealthy", details="Database manager uninitialized"
        )
        is_ready = False

    # Diagnostic failure injection for Kubernetes testing
    if getattr(request.app.state, "_simulate_readiness_failure", False):
        checks["diagnostic_override"] = ComponentHealthCheck(
            status="unhealthy", details="Simulated readiness failure for Kubernetes testing"
        )
        is_ready = False

    readiness_status = "ready" if is_ready else "not_ready"
    response_model = ReadinessResponse(
        status=readiness_status,
        ready=is_ready,
        checks=checks,
        version=settings.app_version,
        environment=settings.app_env,
    )

    status_code = status.HTTP_200_OK if is_ready else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(
        status_code=status_code,
        content=response_model.model_dump(),
    )


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Service Liveness probe",
    description="Evaluates process uptime and heartbeat for Kubernetes liveness probes.",
)
async def get_liveness(request: Request) -> JSONResponse:
    """Return process liveness heartbeat with failure simulation support."""
    settings: Settings = getattr(request.app.state, "settings", None) or Settings()
    startup_time = getattr(request.app.state, "startup_time", None) or time.perf_counter()
    uptime_seconds = round(time.perf_counter() - startup_time, 2)

    # Diagnostic failure injection for Kubernetes testing
    if getattr(request.app.state, "_simulate_liveness_failure", False):
        fail_response = LivenessResponse(
            status="deadlock_detected",
            live=False,
            uptime_seconds=uptime_seconds,
            version=settings.app_version,
            environment=settings.app_env,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=fail_response.model_dump(),
        )

    response_model = LivenessResponse(
        status="alive",
        live=True,
        uptime_seconds=uptime_seconds,
        version=settings.app_version,
        environment=settings.app_env,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=response_model.model_dump(),
    )


@router.post(
    "/health/simulate-fail-liveness",
    summary="Simulate liveness failure",
    description="Diagnostic endpoint to simulate deadlock/unresponsiveness for K8s liveness probes.",
)
async def simulate_fail_liveness(request: Request) -> dict[str, str]:
    """Force liveness probe to return 500 error."""
    request.app.state._simulate_liveness_failure = True
    return {"message": "Simulated liveness failure enabled. K8s probe will fail."}


@router.post(
    "/health/simulate-fail-readiness",
    summary="Simulate readiness failure",
    description="Diagnostic endpoint to simulate dependency failure for K8s readiness probes.",
)
async def simulate_fail_readiness(request: Request) -> dict[str, str]:
    """Force readiness probe to return 503 error."""
    request.app.state._simulate_readiness_failure = True
    return {"message": "Simulated readiness failure enabled. Pod removed from service routing."}


@router.post(
    "/health/simulate-recover",
    summary="Recover simulated failures",
    description="Reset diagnostic failure flags to restore healthy probe responses.",
)
async def simulate_recover(request: Request) -> dict[str, str]:
    """Reset simulated failure flags."""
    request.app.state._simulate_liveness_failure = False
    request.app.state._simulate_readiness_failure = False
    return {"message": "Simulated failures cleared. Probes restored to healthy."}

