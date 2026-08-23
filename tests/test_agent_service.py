import pytest
from app.models.schemas import TaskStatus
from app.services.agent.service import AgentService
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService


@pytest.mark.asyncio
async def test_agent_service_delegates_task() -> None:
    """Verify AgentService coordinates task execution and returns TaskResponse."""
    mock_provider = MockLLMProvider()
    llm_service = LLMService(provider=mock_provider)
    service = AgentService(llm_service=llm_service)

    response = await service.execute_task(
        task="Summarize database indexing strategies.",
        system_instructions="Focus on B-Tree vs Hash indexes.",
    )

    assert response.status == TaskStatus.COMPLETED
    assert response.task == "Summarize database indexing strategies."
    assert len(response.plan) > 0
    assert response.answer is not None
    assert response.metadata.provider == "mock"
    assert "completed" in response.metadata.trace
