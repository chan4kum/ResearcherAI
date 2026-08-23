import threading

from pydantic import BaseModel, Field

from app.core.guardrails.ssrf import validate_safe_url
from app.models.security import Role, UserIdentity


class MCPSafetyPolicy(BaseModel):
    """Safety governance policy controlling MCP server access and tool invocations."""

    allowed_servers: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Whitelist of permitted MCP server identifiers. ['*'] allows all servers.",
    )
    allowed_tools: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Whitelist of permitted tool names. ['*'] allows all discovered tools.",
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description="Maximum execution timeout in seconds per MCP tool invocation.",
    )
    max_invocations: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum allowed tool invocations before execution is throttled/blocked.",
    )
    enforce_whitelist: bool = Field(
        default=True,
        description="Whether to strictly enforce server and tool whitelists.",
    )
    required_roles: dict[str, list[Role]] = Field(
        default_factory=lambda: {
            "admin": [Role.ADMIN],
            "filesystem": [Role.ADMIN],
            "database": [Role.ADMIN, Role.RESEARCHER],
        },
        description="Role-based authorization mappings for sensitive MCP tools/servers.",
    )

    def validate_server_url(self, url: str) -> tuple[bool, str | None]:
        """Verify that remote MCP server endpoint does not violate SSRF constraints."""
        return validate_safe_url(url)

    def is_server_allowed(self, server_name: str) -> bool:
        """Check if target server is permitted under this safety policy."""
        if not self.enforce_whitelist or "*" in self.allowed_servers:
            return True
        return server_name.lower() in [s.lower() for s in self.allowed_servers]

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if target tool is permitted under this safety policy."""
        if not self.enforce_whitelist or "*" in self.allowed_tools:
            return True
        return tool_name.lower() in [t.lower() for t in self.allowed_tools]

    def is_user_authorized_for_mcp_tool(
        self,
        tool_name: str,
        user: UserIdentity | None,
    ) -> tuple[bool, str | None]:
        """Verify role authorization for sensitive MCP tools."""
        req_roles = self.required_roles.get(tool_name.lower())
        if not req_roles:
            return (True, None)

        if user is None:
            return (Role.USER in req_roles or Role.ADMIN in req_roles, None)

        if not any(user.has_role(r) for r in req_roles):
            role_names = [r.value for r in req_roles]
            user_roles = [r.value for r in user.roles]
            return (
                False,
                f"User '{user.username}' with roles {user_roles} lacks required role {role_names} for MCP tool '{tool_name}'.",
            )
        return (True, None)


class MCPInvocationTracker:
    """Thread-safe counter tracking cumulative tool invocations against safety caps."""

    def __init__(self, max_invocations: int = 20) -> None:
        self._max_invocations = max_invocations
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def record_invocation(self, tool_name: str) -> tuple[bool, int, str | None]:
        """Record an invocation of a tool and verify within safety limits.

        Returns:
            (allowed: bool, current_count: int, error_message: str | None)
        """
        with self._lock:
            current_total = sum(self._counts.values())
            if current_total >= self._max_invocations:
                return (
                    False,
                    current_total,
                    f"MCP invocation limit exceeded: {current_total}/{self._max_invocations} "
                    f"maximum calls reached.",
                )

            tool_count = self._counts.get(tool_name, 0) + 1
            self._counts[tool_name] = tool_count
            return (True, tool_count, None)

    def get_total_invocations(self) -> int:
        """Return total recorded invocations."""
        with self._lock:
            return sum(self._counts.values())

    def reset(self) -> None:
        """Reset invocation counter."""
        with self._lock:
            self._counts.clear()
