from fastapi import APIRouter

from app.api.v1.router import api_v1_router
from app.config import get_settings

api_router = APIRouter()
settings = get_settings()

api_router.include_router(api_v1_router, prefix=settings.api_v1_prefix)
