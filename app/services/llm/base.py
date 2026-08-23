from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.core.errors import AppException


class LLMResponse(BaseModel):
    """Normalized response returned by any LLM provider."""

    content: str = Field(description="Generated text content")
    model: str = Field(description="Model name identifier")
    provider: str = Field(description="Provider name identifier")
    prompt_tokens: int | None = Field(default=None, description="Number of tokens in prompt")
    completion_tokens: int | None = Field(
        default=None, description="Number of tokens in completion"
    )
    total_tokens: int | None = Field(default=None, description="Total tokens consumed")
    raw_response: Any = Field(default=None, description="Optional raw provider response")


class LLMError(AppException):
    """Raised when an LLM provider request fails."""

    def __init__(self, message: str, details: Any = None, status_code: int = 502) -> None:
        super().__init__(
            message=message,
            code="LLM_PROVIDER_ERROR",
            status_code=status_code,
            details=details,
        )


class LLMConfigurationError(AppException):
    """Raised when LLM configuration or credentials are missing or invalid."""

    def __init__(self, message: str, details: Any = None) -> None:
        super().__init__(
            message=message,
            code="LLM_CONFIGURATION_ERROR",
            status_code=500,
            details=details,
        )


class BaseLLMProvider(ABC):
    """Abstract interface for LLM provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider implementation."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate a response for a prompt."""
