from collections.abc import Callable
from typing import Any

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings
from app.core.errors import ForbiddenException, UnauthorizedException
from app.core.logging import get_logger
from app.models.security import Role, UserIdentity

logger = get_logger("app.core.security")

# Security schemes for OpenAPI docs and extraction
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_token_scheme = HTTPBearer(auto_error=False)


def _get_active_settings(request: Request) -> Settings:
    """Retrieve Settings instance from application state or global fallback."""
    return getattr(request.app.state, "settings", None) or Settings()


async def get_current_user(
    request: Request,
    api_key: str | None = Security(api_key_header_scheme),
    bearer_creds: HTTPAuthorizationCredentials | None = Security(bearer_token_scheme),
) -> UserIdentity:
    """FastAPI dependency to extract and authenticate the requesting user identity."""
    settings = _get_active_settings(request)

    # 1. Bypass check for local development environments if explicitly disabled
    if not settings.security_enabled:
        return UserIdentity(
            user_id="usr_dev_bypass",
            username="dev_admin",
            roles=[Role.ADMIN],
            is_active=True,
        )

    # 2. Extract credential token from Header (Bearer or X-API-Key)
    token: str | None = None
    if bearer_creds and bearer_creds.credentials:
        token = bearer_creds.credentials
    elif api_key:
        token = api_key
    elif "authorization" in request.headers:
        auth_header = request.headers["authorization"]
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
        else:
            token = auth_header.strip()

    if not token:
        logger.info(
            "authentication_missing_credentials",
            path=request.url.path,
            ip=request.client.host if request.client else "unknown",
        )
        raise UnauthorizedException(
            message="Authentication credentials required",
            details={
                "supported_methods": [
                    "Authorization: Bearer <token>",
                    "X-API-Key: <key>",
                ]
            },
        )

    # 3. Lookup user identity in configured auth tokens
    token_data: dict[str, Any] | None = settings.auth_tokens.get(token)
    if not token_data:
        logger.warning("authentication_invalid_token", path=request.url.path)
        raise UnauthorizedException(message="Invalid authentication credentials")

    # 4. Construct and validate UserIdentity
    roles = [Role(r) for r in token_data.get("roles", ["user"])]
    user = UserIdentity(
        user_id=token_data.get("user_id", "unknown_user"),
        username=token_data.get("username", "user"),
        roles=roles,
        is_active=token_data.get("is_active", True),
        metadata=token_data.get("metadata", {}),
    )

    if not user.is_active:
        logger.warning("authentication_inactive_user", user_id=user.user_id)
        raise UnauthorizedException(message="User account is deactivated")

    # Attach authenticated user to request state for downstream telemetry
    request.state.current_user = user
    return user


def require_roles(*required_roles: Role) -> Callable[..., Any]:
    """Dependency factory enforcing that authenticated user has at least one required role."""

    async def role_checker(
        current_user: UserIdentity = Depends(get_current_user),
    ) -> UserIdentity:
        if not current_user.has_any_role(*required_roles):
            logger.warning(
                "authorization_role_denied",
                user_id=current_user.user_id,
                user_roles=[r.value for r in current_user.roles],
                required_roles=[r.value for r in required_roles],
            )
            role_names = [r.value for r in required_roles]
            raise ForbiddenException(
                message=f"Access forbidden: required role(s) {role_names}",
                details={
                    "user_roles": [r.value for r in current_user.roles],
                    "required_roles": role_names,
                },
            )
        return current_user

    return role_checker


# Convenient role dependency shortcuts
require_authenticated_user = Depends(get_current_user)
require_admin = Depends(require_roles(Role.ADMIN))
require_researcher = Depends(require_roles(Role.RESEARCHER, Role.ADMIN))
