from fastapi import APIRouter, Request

from app.config import Settings
from app.models.schemas import AppInfoResponse

router = APIRouter()


@router.get(
    "/info",
    response_model=AppInfoResponse,
    summary="Platform metadata",
    description="Returns high-level metadata and configuration info about the platform.",
)
async def get_info(request: Request) -> AppInfoResponse:
    """Return platform metadata."""
    settings: Settings = getattr(request.app.state, "settings", None) or Settings()
    return AppInfoResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        api_version=settings.api_v1_prefix,
    )
