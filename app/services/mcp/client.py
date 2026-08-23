import uuid
from typing import Any

from app.core.logging import get_logger
from app.core.tracing import agent_span
from app.services.mcp.models import (
    MCPCallToolResult,
    MCPRequest,
    MCPResponse,
    MCPToolDefinition,
)
from app.services.mcp.server import LocalMCPServer

logger = get_logger("app.services.mcp.client")


class MCPClient:
    """Client for discovering and invoking tools on Model Context Protocol (MCP) servers."""

    def __init__(self, server: LocalMCPServer) -> None:
        self._server = server
        self._client_id = f"client_{uuid.uuid4().hex[:8]}"
        self._initialized = False

    @property
    def server_name(self) -> str:
        """Name of the connected MCP server."""
        return self._server.server_name

    async def initialize(self) -> dict[str, Any]:
        """Perform MCP protocol initialization handshake."""
        request = MCPRequest(
            id=str(uuid.uuid4()),
            method="initialize",
            params={
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "agentic-ai-mcp-client",
                    "version": "1.0.0",
                },
            },
        )
        response = await self._server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP Initialize failed: {response.error.message}")

        self._initialized = True
        logger.info("mcp_client_initialized", server=self._server.server_name)
        return response.result or {}

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Discover tools exposed by the MCP server."""
        if not self._initialized:
            await self.initialize()

        request = MCPRequest(
            id=str(uuid.uuid4()),
            method="tools/list",
            params={},
        )
        response: MCPResponse = await self._server.handle_request(request)
        if response.error:
            raise RuntimeError(f"MCP tools/list failed: {response.error.message}")

        tools_data = response.result.get("tools", []) if response.result else []
        tools: list[MCPToolDefinition] = [MCPToolDefinition(**t) for t in tools_data]
        logger.info("mcp_tools_discovered", count=len(tools), server=self._server.server_name)
        return tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPCallToolResult:
        """Invoke a tool on the MCP server via JSON-RPC protocol."""
        if not self._initialized:
            await self.initialize()

        request = MCPRequest(
            id=str(uuid.uuid4()),
            method="tools/call",
            params={
                "name": tool_name,
                "arguments": arguments,
            },
        )

        with agent_span(
            "mcp.tool_call",
            tool_name=tool_name,
            extra={"mcp.server": self._server.server_name, "mcp.args_count": len(arguments)},
        ) as span:
            response = await self._server.handle_request(request)
            if response.error:
                span.set_attribute("mcp.error", response.error.message[:200])
                span.set_attribute("mcp.success", False)
                logger.error("mcp_tool_call_error", tool=tool_name, error=response.error.message)
                return MCPCallToolResult(
                    content=[],
                    isError=True,
                )

            span.set_attribute("mcp.success", True)
            result_data = response.result or {}
            return MCPCallToolResult(**result_data)
