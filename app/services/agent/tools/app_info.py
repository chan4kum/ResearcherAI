from typing import Any

from app.config import Settings, get_settings
from app.services.agent.tools.base import BaseTool, ToolResult


class AppInfoTool(BaseTool):
    """Tool for querying application runtime metadata and environment information."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @property
    def name(self) -> str:
        return "app_info"

    @property
    def description(self) -> str:
        return (
            "Retrieve current application runtime information and metadata "
            "(application name, version, environment, API version prefix)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            metadata = {
                "app_name": self._settings.app_name,
                "version": self._settings.app_version,
                "environment": self._settings.app_env,
                "api_version": self._settings.api_v1_prefix,
            }
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=metadata,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"Failed to retrieve application metadata: {exc}",
            )
