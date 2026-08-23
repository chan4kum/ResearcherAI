from unittest.mock import AsyncMock, MagicMock

import openai
import pytest
from app.services.llm.base import LLMError
from app.services.llm.openai import OpenAIProvider


@pytest.mark.asyncio
async def test_openai_provider_success() -> None:
    """Verify OpenAIProvider parses completion response correctly."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Semiconductors are foundation elements in electronics."

    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 15
    mock_usage.total_tokens = 25

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-4o-mini"
    mock_response.usage = mock_usage
    mock_response.model_dump.return_value = {"id": "chatcmpl-123"}

    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    provider = OpenAIProvider(
        api_key="test-key",
        model="gpt-4o-mini",
        client=mock_client,
    )

    result = await provider.generate(
        prompt="Explain semiconductors.",
        system_prompt="You are a tutor.",
        temperature=0.2,
    )

    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.content == "Semiconductors are foundation elements in electronics."
    assert result.total_tokens == 25

    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "Explain semiconductors."},
        ],
        temperature=0.2,
    )


@pytest.mark.asyncio
async def test_openai_provider_handles_auth_error() -> None:
    """Verify OpenAIProvider translates AuthenticationError to LLMError."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.AuthenticationError(
            message="Invalid API Key",
            response=MagicMock(status_code=401),
            body=None,
        )
    )

    provider = OpenAIProvider(api_key="bad-key", client=mock_client)

    with pytest.raises(LLMError) as exc_info:
        await provider.generate("hello")

    assert exc_info.value.status_code == 401
    assert "authentication failed" in str(exc_info.value).lower()
