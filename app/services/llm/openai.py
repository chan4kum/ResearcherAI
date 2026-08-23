from typing import Any

import openai
from openai import AsyncOpenAI

from app.core.logging import get_logger
from app.services.llm.base import BaseLLMProvider, LLMError, LLMResponse

logger = get_logger("app.services.llm.openai")


class OpenAIProvider(BaseLLMProvider):
    """OpenAI and OpenAI-compatible LLM provider adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._client = client or AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
            max_retries=max_retries,
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            logger.info("llm_generate_start", provider="openai", model=self._model)
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content or ""

            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else None
            completion_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None

            logger.info(
                "llm_generate_success",
                provider="openai",
                model=response.model,
                total_tokens=total_tokens,
            )

            return LLMResponse(
                content=content,
                model=response.model or self._model,
                provider=self.provider_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )
        except openai.AuthenticationError as exc:
            logger.error("openai_auth_error", error=str(exc))
            raise LLMError(f"OpenAI authentication failed: {exc}", status_code=401) from exc
        except openai.RateLimitError as exc:
            logger.error("openai_rate_limit", error=str(exc))
            raise LLMError(f"OpenAI rate limit exceeded: {exc}", status_code=429) from exc
        except openai.APIConnectionError as exc:
            logger.error("openai_connection_error", error=str(exc))
            raise LLMError(
                f"Could not connect to OpenAI API: {exc}", status_code=503
            ) from exc
        except openai.APIError as exc:
            logger.error("openai_api_error", error=str(exc))
            raise LLMError(f"OpenAI API error: {exc}", status_code=502) from exc
        except Exception as exc:
            logger.error("openai_unhandled_error", error=str(exc))
            raise LLMError(f"Unexpected error during LLM generation: {exc}") from exc
