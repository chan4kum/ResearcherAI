from typing import Any

from app.config import Settings
from app.services.agent.tools.app_info import AppInfoTool
from app.services.agent.tools.base import BaseTool, ToolResult
from app.services.agent.tools.calculator import CalculatorTool


class ToolRegistry:
    """Registry maintaining available agent tools and dispatching tool executions."""

    def __init__(
        self,
        tools: list[BaseTool] | None = None,
        settings: Settings | None = None,
        budget_tracker: Any = None,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._budget_tracker = budget_tracker
        if tools is not None:
            for tool in tools:
                self.register(tool)
        else:
            # Register default safe tools
            self.register(CalculatorTool())
            self.register(AppInfoTool(settings=settings))

    def register(self, tool: BaseTool) -> None:
        """Register a tool instance in the registry."""
        self._tools[tool.name.lower()] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool instance by name."""
        self._tools.pop(name.lower(), None)

    async def register_mcp_client(self, mcp_client: Any) -> list[str]:
        """Discover tools exposed by an MCP client and adapt them into the registry."""
        from app.services.mcp.adapter import MCPToolAdapter

        tool_defs = await mcp_client.list_tools()
        registered_names: list[str] = []
        for defn in tool_defs:
            adapter = MCPToolAdapter(mcp_client=mcp_client, tool_definition=defn)
            self.register(adapter)
            registered_names.append(defn.name)
        return registered_names

    def get(self, name: str) -> BaseTool | None:
        """Retrieve a registered tool by its name."""
        return self._tools.get(name.lower())

    @property
    def available_tool_names(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._tools.keys())

    def list_tools(self) -> list[BaseTool]:
        """Return list of all registered tool instances."""
        return list(self._tools.values())

    def format_tools_description(self) -> str:
        """Format registered tools into a structured description for prompt injection."""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- Tool: `{tool.name}`")
            lines.append(f"  Description: {tool.description}")
            params = tool.parameters_schema.get("properties", {})
            req = tool.parameters_schema.get("required", [])
            lines.append(f"  Parameters: {list(params.keys())} (Required: {req})")
        return "\n".join(lines)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Dispatch execution to the named tool, safely enforcing budget and missing tool errors."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=(
                    f"Tool '{tool_name}' not found. "
                    f"Available registered tools: {self.available_tool_names}"
                ),
            )

        if self._budget_tracker is not None:
            try:
                self._budget_tracker.record_tool_call(tool_name)
            except Exception as exc:
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(exc),
                )

        return tool.execute(**kwargs)
