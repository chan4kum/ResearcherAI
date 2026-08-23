import time

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.core.metrics import record_llm_metrics
from app.services.llm.base import BaseLLMProvider, LLMResponse
from app.services.llm.factory import create_llm_provider

logger = get_logger("app.services.llm.service")



class LLMService:
    """Core domain service for orchestrating LLM interactions."""

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or create_llm_provider(self._settings)

    @property
    def provider(self) -> BaseLLMProvider:
        return self._provider

    async def chat(
        self,
        message: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a prompt message to the underlying provider and return normalized response."""
        temp = temperature if temperature is not None else self._settings.llm_temperature
        tokens = max_tokens if max_tokens is not None else self._settings.llm_max_tokens

        start_time = time.perf_counter()
        model_name = getattr(self._provider, "model_name", "unknown")
        provider_name = self._provider.provider_name

        try:
            response = await self._provider.generate(
                prompt=message,
                system_prompt=system_prompt,
                temperature=temp,
                max_tokens=tokens,
            )
            duration_sec = time.perf_counter() - start_time
            record_llm_metrics(
                model=response.model or model_name,
                provider=response.provider or provider_name,
                status="success",
                duration_sec=duration_sec,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
            return response
        except Exception as exc:
            duration_sec = time.perf_counter() - start_time
            record_llm_metrics(
                model=model_name,
                provider=provider_name,
                status="error",
                duration_sec=duration_sec,
            )
            raise exc


    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate response for a prompt message (alias for chat)."""
        return await self.chat(
            message=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
