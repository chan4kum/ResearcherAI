import asyncio
import concurrent.futures
from typing import Any

from app.core.logging import get_logger
from app.services.agent.tools.base import BaseTool, ToolResult
from app.services.mcp.client import MCPClient
from app.services.mcp.models import MCPToolDefinition
from app.services.mcp.safety import MCPInvocationTracker, MCPSafetyPolicy

logger = get_logger("app.services.mcp.adapter")


class MCPToolAdapter(BaseTool):
    """Bridges an MCP-hosted tool into the native BaseTool interface with safety controls."""

    def __init__(
        self,
        mcp_client: MCPClient,
        tool_definition: MCPToolDefinition,
        safety_policy: MCPSafetyPolicy | None = None,
        tracker: MCPInvocationTracker | None = None,
    ) -> None:
        self._client = mcp_client
        self._definition = tool_definition
        self._safety_policy = safety_policy or MCPSafetyPolicy()
        self._tracker = tracker

    @property
    def name(self) -> str:
        return self._definition.name

    @property
    def description(self) -> str:
        return self._definition.description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._definition.inputSchema

    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool by delegating to the MCPClient with safety enforcement."""
        # 1. Server Whitelist Guardrail
        if not self._safety_policy.is_server_allowed(self._client.server_name):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Execution blocked: Server '{self._client.server_name}' is not in "
                    f"the allowed MCP servers whitelist."
                ),
            )

        # 2. Tool Whitelist Guardrail
        if not self._safety_policy.is_tool_allowed(self.name):
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    f"Execution blocked: Tool '{self.name}' is not permitted by "
                    f"active MCP safety policy."
                ),
            )

        # 3. Invocation Limit Tracker
        if self._tracker is not None:
            allowed, _, err_msg = self._tracker.record_invocation(self.name)
            if not allowed:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=err_msg or "MCP invocation limit reached.",
                )

        timeout_sec = self._safety_policy.timeout_seconds

        try:
            # Handle event loop safely for sync BaseTool.execute call
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self._client.call_tool(self.name, kwargs),
                    )
                    mcp_res = future.result(timeout=timeout_sec)
            else:
                mcp_res = asyncio.run(
                    asyncio.wait_for(
                        self._client.call_tool(self.name, kwargs),
                        timeout=timeout_sec,
                    )
                )

            if mcp_res.isError:
                error_msg = (
                    mcp_res.content[0].text
                    if mcp_res.content
                    else "Tool execution failed on MCP server."
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=error_msg,
                )

            output_text = mcp_res.content[0].text if mcp_res.content else ""
            return ToolResult(
                tool_name=self.name,
                success=True,
                output=output_text,
            )

        except (TimeoutError, concurrent.futures.TimeoutError):
            logger.warning("mcp_tool_execution_timeout", tool=self.name, timeout=timeout_sec)
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"MCP Tool execution timed out after {timeout_sec}s.",
            )
        except Exception as exc:
            logger.error("mcp_tool_adapter_execution_error", tool=self.name, error=str(exc))
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"MCP Tool execution exception: {exc}",
            )
