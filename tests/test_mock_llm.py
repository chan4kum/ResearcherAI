import pytest
from app.services.llm.base import LLMError
from app.services.llm.mock import MockLLMProvider


@pytest.mark.asyncio
async def test_mock_llm_generates_response() -> None:
    """Verify MockLLMProvider returns structured response and records calls."""
    provider = MockLLMProvider()
    response = await provider.generate(
        prompt="Explain quantum computing.",
        system_prompt="Be concise.",
        temperature=0.5,
    )

    assert response.provider == "mock"
    assert response.model == "mock-model-v1"
    assert "quantum computing" in response.content
    assert response.total_tokens is not None
    assert response.total_tokens > 0

    assert len(provider.calls) == 1
    assert provider.calls[0]["prompt"] == "Explain quantum computing."
    assert provider.calls[0]["system_prompt"] == "Be concise."


@pytest.mark.asyncio
async def test_mock_llm_custom_default_response() -> None:
    """Verify MockLLMProvider returns custom canned response if specified."""
    canned = "Semiconductors are essential materials for modern microchips."
    provider = MockLLMProvider(default_response=canned)
    response = await provider.generate("Tell me about chips.")

    assert response.content == canned


@pytest.mark.asyncio
async def test_mock_llm_simulated_failure() -> None:
    """Verify MockLLMProvider raises LLMError when should_fail is True."""
    provider = MockLLMProvider(should_fail=True, failure_message="Rate limit simulated")
    with pytest.raises(LLMError) as exc_info:
        await provider.generate("hello")

    assert "Rate limit simulated" in str(exc_info.value)
    assert exc_info.value.code == "LLM_PROVIDER_ERROR"
