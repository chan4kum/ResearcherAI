from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """Specification of a tool call decision made by the agent."""

    tool_name: str = Field(description="The name of the tool to execute")
    tool_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )


class ToolResult(BaseModel):
    """Standardized output returned from a tool execution."""

    tool_name: str = Field(description="The name of the executed tool")
    success: bool = Field(description="Whether tool execution succeeded")
    output: Any = Field(default=None, description="Output data or result if successful")
    error: str | None = Field(default=None, description="Error message if execution failed")


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human/LLM-readable description of what the tool does and when to use it."""
        pass

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON-schema representation of tool parameters."""
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool synchronously with provided arguments."""
        pass
