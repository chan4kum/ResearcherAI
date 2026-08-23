from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Role(StrEnum):
    """User authorization roles within the platform."""

    ADMIN = "admin"
    RESEARCHER = "researcher"
    USER = "user"
    VIEWER = "viewer"


class UserIdentity(BaseModel):
    """Structured authenticated user identity."""

    user_id: str = Field(description="Unique subject identifier for user")
    username: str = Field(description="Human-readable user account handle")
    roles: list[Role] = Field(
        default_factory=lambda: [Role.USER],
        description="Assigned authorization roles",
    )
    is_active: bool = Field(default=True, description="Account active status")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional identity metadata attributes",
    )

    def has_role(self, role: Role | str) -> bool:
        """Check if user has a specific role or superuser admin role."""
        target = role.value if isinstance(role, Role) else role.lower()
        user_roles = [r.value for r in self.roles]
        return target in user_roles or Role.ADMIN.value in user_roles

    def has_any_role(self, *roles: Role | str) -> bool:
        """Check if user has at least one of the specified roles."""
        return any(self.has_role(r) for r in roles)


class UserResponse(BaseModel):
    """Public representation of authenticated user identity."""

    user_id: str = Field(description="Unique user ID")
    username: str = Field(description="Username")
    roles: list[str] = Field(description="List of active roles")
    is_active: bool = Field(description="Active account flag")
