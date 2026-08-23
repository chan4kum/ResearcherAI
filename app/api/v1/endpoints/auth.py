from typing import Any

from fastapi import APIRouter, Depends

from app.core.security import get_current_user, require_roles
from app.models.security import Role, UserIdentity, UserResponse

router = APIRouter()


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get authenticated user profile",
    description="Returns the profile and active roles of the authenticated caller.",
)
async def get_my_profile(
    current_user: UserIdentity = Depends(get_current_user),
) -> UserResponse:
    """Return caller's authenticated identity."""
    return UserResponse(
        user_id=current_user.user_id,
        username=current_user.username,
        roles=[r.value for r in current_user.roles],
        is_active=current_user.is_active,
    )


@router.get(
    "/admin-only",
    summary="Admin-only diagnostic endpoint",
    description="Restricted diagnostic endpoint accessible exclusively by administrators.",
)
async def admin_only_action(
    current_user: UserIdentity = Depends(require_roles(Role.ADMIN)),
) -> dict[str, Any]:
    """Execute administrative operations."""
    return {
        "status": "authorized",
        "message": f"Welcome Admin {current_user.username}",
        "user_id": current_user.user_id,
        "roles": [r.value for r in current_user.roles],
    }


@router.get(
    "/researcher-only",
    summary="Researcher workspace endpoint",
    description="Restricted workspace endpoint accessible by Researchers and Administrators.",
)
async def researcher_only_action(
    current_user: UserIdentity = Depends(require_roles(Role.RESEARCHER, Role.ADMIN)),
) -> dict[str, Any]:
    """Execute researcher actions."""
    return {
        "status": "authorized",
        "message": f"Welcome Researcher {current_user.username}",
        "user_id": current_user.user_id,
        "roles": [r.value for r in current_user.roles],
    }
