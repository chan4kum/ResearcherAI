import pytest
from app.config import Settings
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService


@pytest.mark.asyncio
async def test_llm_service_uses_provided_adapter() -> None:
    """Verify LLMService properly delegates calls to its configured provider."""
    mock_provider = MockLLMProvider(default_response="Service test response")
    service = LLMService(
        provider=mock_provider,
        settings=Settings(llm_temperature=0.3, llm_max_tokens=1000),
    )

    response = await service.chat(message="Test query")

    assert response.content == "Service test response"
    assert len(mock_provider.calls) == 1
    assert mock_provider.calls[0]["prompt"] == "Test query"
    assert mock_provider.calls[0]["temperature"] == 0.3
    assert mock_provider.calls[0]["max_tokens"] == 1000
