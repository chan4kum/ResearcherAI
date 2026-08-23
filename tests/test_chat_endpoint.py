import pytest
from app.config import Settings
from app.main import create_app
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_post_chat_success(client: AsyncClient) -> None:
    """Verify POST /api/v1/chat successfully generates an answer using mock provider."""
    payload = {"message": "Explain semiconductor manufacturing in simple terms."}
    response = await client.post("/api/v1/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "semiconductor manufacturing" in data["answer"]
    assert data["provider"] == "mock"
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_post_chat_with_custom_canned_mock() -> None:
    """Verify POST /api/v1/chat with custom canned response."""
    canned = "Semiconductors conduct electricity conditionally."
    mock_provider = MockLLMProvider(default_response=canned)
    llm_service = LLMService(provider=mock_provider)
    app = create_app(settings=Settings(llm_provider="mock"), llm_service=llm_service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        response = await test_client.post(
            "/api/v1/chat",
            json={
                "message": "What is a semiconductor?",
                "system_prompt": "You are a physicist.",
                "temperature": 0.1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Semiconductors conduct electricity conditionally."
        assert len(mock_provider.calls) == 1
        assert mock_provider.calls[0]["system_prompt"] == "You are a physicist."
        assert mock_provider.calls[0]["temperature"] == 0.1


@pytest.mark.asyncio
async def test_post_chat_validation_error_on_empty_message(client: AsyncClient) -> None:
    """Verify POST /api/v1/chat returns 422 when message is empty."""
    response = await client.post("/api/v1/chat", json={"message": ""})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_post_chat_error_propagation() -> None:
    """Verify POST /api/v1/chat returns 502 when the LLM provider fails."""
    mock_provider = MockLLMProvider(should_fail=True, failure_message="Upstream provider timeout")
    llm_service = LLMService(provider=mock_provider)
    app = create_app(settings=Settings(llm_provider="mock"), llm_service=llm_service)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        response = await test_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "LLM_PROVIDER_ERROR"
        assert "Upstream provider timeout" in data["error"]["message"]
