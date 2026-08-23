import json
from typing import Any

from app.core.logging import get_logger
from app.services.agent.tools.base import BaseTool
from app.services.mcp.models import (
    MCPCallToolResult,
    MCPContentItem,
    MCPError,
    MCPRequest,
    MCPResponse,
    MCPToolDefinition,
)

logger = get_logger("app.services.mcp.server")


class LocalMCPServer:
    """In-process Model Context Protocol (MCP) server handling JSON-RPC tool dispatches."""

    def __init__(self, server_name: str = "local-mcp-server", version: str = "1.0.0") -> None:
        self.server_name = server_name
        self.version = version
        self._tools: dict[str, BaseTool] = {}
        self._is_initialized = False

    def register_tool(self, tool: BaseTool) -> None:
        """Register a tool instance onto the MCP server."""
        self._tools[tool.name] = tool
        logger.info("mcp_tool_registered", server=self.server_name, tool=tool.name)

    def list_tool_definitions(self) -> list[MCPToolDefinition]:
        """Enumerate all registered tools in MCP standard format."""
        definitions: list[MCPToolDefinition] = []
        for name, tool in self._tools.items():
            definitions.append(
                MCPToolDefinition(
                    name=name,
                    description=tool.description,
                    inputSchema=tool.parameters_schema,
                )
            )
        return definitions

    async def handle_request(self, request_payload: MCPRequest | dict[str, Any]) -> MCPResponse:
        """Process an incoming JSON-RPC 2.0 MCP request."""
        if isinstance(request_payload, dict):
            request = MCPRequest(**request_payload)
        else:
            request = request_payload

        req_id = request.id
        method = request.method
        params = request.params

        logger.debug("mcp_server_request_received", method=method, req_id=req_id)

        # 1. initialize
        if method == "initialize":
            self._is_initialized = True
            return MCPResponse(
                id=req_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {},
                        "prompts": {},
                    },
                    "serverInfo": {
                        "name": self.server_name,
                        "version": self.version,
                    },
                },
            )

        # 2. ping
        if method == "ping":
            return MCPResponse(id=req_id, result={})

        # 3. tools/list
        if method == "tools/list":
            tools = [t.model_dump() for t in self.list_tool_definitions()]
            return MCPResponse(id=req_id, result={"tools": tools})

        # 4. tools/call
        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})

            if not tool_name or tool_name not in self._tools:
                return MCPResponse(
                    id=req_id,
                    error=MCPError(
                        code=-32601,
                        message=f"Tool '{tool_name}' not found on MCP server '{self.server_name}'.",
                    ),
                )

            target_tool = self._tools[tool_name]
            try:
                result = target_tool.execute(**arguments)
                if result.success:
                    output_text = (
                        json.dumps(result.output)
                        if not isinstance(result.output, str)
                        else result.output
                    )
                    call_result = MCPCallToolResult(
                        content=[MCPContentItem(type="text", text=str(output_text))],
                        isError=False,
                    )
                else:
                    call_result = MCPCallToolResult(
                        content=[
                            MCPContentItem(
                                type="text",
                                text=f"Error executing tool '{tool_name}': {result.error}",
                            )
                        ],
                        isError=True,
                    )
                return MCPResponse(id=req_id, result=call_result.model_dump())
            except Exception as exc:
                logger.error("mcp_tool_execution_error", tool=tool_name, error=str(exc))
                call_result = MCPCallToolResult(
                    content=[MCPContentItem(type="text", text=f"Exception: {exc}")],
                    isError=True,
                )
                return MCPResponse(id=req_id, result=call_result.model_dump())

        # Method Not Found
        return MCPResponse(
            id=req_id,
            error=MCPError(
                code=-32601,
                message=f"Unknown method '{method}' on MCP server.",
            ),
        )
