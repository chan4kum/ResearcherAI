from app.config import Settings
from app.services.llm.base import BaseLLMProvider, LLMConfigurationError
from app.services.llm.gemini import GeminiProvider
from app.services.llm.mock import MockLLMProvider
from app.services.llm.openai import OpenAIProvider


def create_llm_provider(settings: Settings) -> BaseLLMProvider:
    """Instantiate and return the configured LLM provider adapter."""
    provider_type = settings.llm_provider.strip().lower()

    if provider_type == "mock":
        return MockLLMProvider(model_name=settings.llm_model)

    if provider_type == "openai":
        if not settings.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY must be configured when LLM_PROVIDER is 'openai'"
            )
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.llm_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider_type in ("gemini", "google"):
        if not settings.gemini_api_key:
            raise LLMConfigurationError(
                "GEMINI_API_KEY must be configured when LLM_PROVIDER is 'gemini'"
            )
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.llm_model if "gemini" in settings.llm_model else "gemini-3.6-flash",
            timeout_seconds=settings.llm_timeout_seconds,
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{settings.llm_provider}'. Supported: 'mock', 'openai', 'gemini'."
    )
