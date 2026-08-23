"""
app/services/llm/gemini.py — Google Gemini Provider Adapter

Connects to Google's Generative Language API using httpx async client.
"""

from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.llm.base import BaseLLMProvider, LLMError, LLMResponse

logger = get_logger("app.services.llm.gemini")


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider adapter."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-3.6-flash",
        timeout_seconds: float = 30.0,
    ) -> None:
        self._api_key = api_key
        # Default to working flash model if generic model string passed
        self._model = model if "gemini" in model else "gemini-3.6-flash"
        self._timeout_seconds = timeout_seconds
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "gemini"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"

        contents: list[dict[str, Any]] = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})

        contents.append({"role": "user", "parts": [{"text": prompt}]})

        generation_config: dict[str, Any] = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens

        payload: dict[str, Any] = {"contents": contents}
        if generation_config:
            payload["generationConfig"] = generation_config

        try:
            logger.info("llm_generate_start", provider="gemini", model=self._model)
            response = await self._client.post(url, json=payload)

            if response.status_code != 200:
                err_msg = response.text
                logger.error("gemini_api_error", status_code=response.status_code, error=err_msg)
                raise LLMError(f"Gemini API returned status {response.status_code}: {err_msg}")

            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise LLMError("Gemini API returned no candidates.")

            content = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = data.get("usageMetadata", {})
            prompt_tokens = usage.get("promptTokenCount", len(prompt.split()))
            completion_tokens = usage.get("candidatesTokenCount", len(content.split()))
            total_tokens = usage.get("totalTokenCount", prompt_tokens + completion_tokens)

            return LLMResponse(
                content=content,
                model=self._model,
                provider="gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            if isinstance(e, LLMError):
                raise
            logger.error("gemini_call_failed", error=str(e))
            raise LLMError(f"Gemini LLM request failed: {e}") from e

    async def close(self) -> None:
        await self._client.aclose()
