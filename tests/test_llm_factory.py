import pytest
from app.config import Settings
from app.services.llm.base import LLMConfigurationError
from app.services.llm.factory import create_llm_provider
from app.services.llm.mock import MockLLMProvider
from app.services.llm.openai import OpenAIProvider


def test_factory_returns_mock_provider() -> None:
    """Verify factory instantiates MockLLMProvider when provider is 'mock'."""
    settings = Settings(llm_provider="mock", llm_model="test-mock-v1")
    provider = create_llm_provider(settings)
    assert isinstance(provider, MockLLMProvider)
    assert provider.provider_name == "mock"


def test_factory_returns_openai_provider() -> None:
    """Verify factory instantiates OpenAIProvider when provider is 'openai' and key is present."""
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test-key-12345",
        llm_model="gpt-4o",
    )
    provider = create_llm_provider(settings)
    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "openai"


def test_factory_raises_when_openai_key_missing() -> None:
    """Verify factory raises LLMConfigurationError when OpenAI key is missing."""
    settings = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(LLMConfigurationError) as exc_info:
        create_llm_provider(settings)
    assert "OPENAI_API_KEY must be configured" in str(exc_info.value)


def test_factory_raises_for_unsupported_provider() -> None:
    """Verify factory raises LLMConfigurationError for unknown provider string."""
    settings = Settings(llm_provider="unsupported-provider-xyz")
    with pytest.raises(LLMConfigurationError) as exc_info:
        create_llm_provider(settings)
    assert "Unsupported LLM provider" in str(exc_info.value)
