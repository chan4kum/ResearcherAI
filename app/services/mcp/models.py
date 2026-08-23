from typing import Any

from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """JSON-RPC 2.0 Request according to Model Context Protocol (MCP) spec."""

    jsonrpc: str = Field(default="2.0", description="Protocol version, always 2.0")
    id: str | int = Field(description="Unique request ID for correlation")
    method: str = Field(description="Target MCP method name (e.g. tools/list, tools/call)")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Method parameters dictionary",
    )


class MCPError(BaseModel):
    """JSON-RPC 2.0 Error object."""

    code: int = Field(description="Error code integer")
    message: str = Field(description="Human readable error explanation")
    data: Any = Field(default=None, description="Optional error metadata")


class MCPResponse(BaseModel):
    """JSON-RPC 2.0 Response from an MCP server."""

    jsonrpc: str = Field(default="2.0", description="Protocol version, always 2.0")
    id: str | int | None = Field(default=None, description="Correlated request ID")
    result: Any | None = Field(default=None, description="Result payload if successful")
    error: MCPError | None = Field(default=None, description="Error details if failed")


class MCPToolDefinition(BaseModel):
    """Schema and metadata describing an MCP-exposed tool."""

    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human/LLM-readable tool explanation")
    inputSchema: dict[str, Any] = Field(
        default_factory=dict,
        description="JSON Schema defining required and optional arguments",
    )


class MCPContentItem(BaseModel):
    """Standardized content item returned from an MCP tool invocation."""

    type: str = Field(default="text", description="Content type (text, image, resource)")
    text: str = Field(description="Content text payload")


class MCPCallToolResult(BaseModel):
    """Standardized result returned from an MCP tools/call invocation."""

    content: list[MCPContentItem] = Field(
        default_factory=list,
        description="Content items returned by the tool",
    )
    isError: bool = Field(
        default=False,
        description="Whether tool execution resulted in an error",
    )
