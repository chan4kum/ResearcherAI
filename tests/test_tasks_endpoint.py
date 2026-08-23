import pytest
from app.config import Settings
from app.main import create_app
from app.models.schemas import TaskStatus
from app.services.agent.service import AgentService
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_post_task_endpoint_success(client: AsyncClient) -> None:
    """Verify POST /api/v1/tasks returns structured response with plan, answer and metadata."""
    payload = {"task": "Explain how semiconductor manufacturing works."}
    response = await client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["task"] == "Explain how semiconductor manufacturing works."
    assert data["status"] == TaskStatus.COMPLETED.value
    assert data["task_id"] is not None
    assert isinstance(data["plan"], list)
    assert len(data["plan"]) >= 2
    assert data["answer"] is not None
    assert len(data["answer"]) > 0

    # Validate execution metadata
    metadata = data["metadata"]
    assert metadata["provider"] == "mock"
    assert metadata["model"] in {"gpt-4o-mini", "mock-model-v1"}
    assert metadata["duration_ms"] > 0
    assert metadata["total_tokens"] > 0
    assert "initialized" in metadata["trace"]
    assert "completed" in metadata["trace"]

    # Verify correlation header
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_post_task_with_system_instructions(client: AsyncClient) -> None:
    """Verify POST /api/v1/tasks accepts and applies optional system_instructions."""
    payload = {
        "task": "Design a high-throughput cache tier.",
        "system_instructions": "Focus on Redis cluster topologies and cache-aside patterns.",
    }
    response = await client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == TaskStatus.COMPLETED.value
    assert len(data["plan"]) > 0


@pytest.mark.asyncio
async def test_post_task_validation_error_on_empty_task(client: AsyncClient) -> None:
    """Verify POST /api/v1/tasks returns HTTP 422 when task is empty."""
    response = await client.post("/api/v1/tasks", json={"task": ""})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_post_task_failure_handling() -> None:
    """Verify POST /api/v1/tasks returns structured failed status if LLM fails."""
    failing_provider = MockLLMProvider(should_fail=True, failure_message="Upstream outage")
    llm_service = LLMService(provider=failing_provider)
    agent_service = AgentService(llm_service=llm_service)
    app = create_app(
        settings=Settings(llm_provider="mock"),
        llm_service=llm_service,
        agent_service=agent_service,
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        response = await test_client.post("/api/v1/tasks", json={"task": "Failing task"})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == TaskStatus.FAILED.value
        assert "Upstream outage" in (data["error"] or "")
        assert "failed" in data["metadata"]["trace"]


@pytest.mark.asyncio
async def test_post_task_with_tool_invocation(client: AsyncClient) -> None:
    """Verify POST /api/v1/tasks invokes calculator tool and records tools_used metadata."""
    payload = {"task": "Please calculate 25 * 4 for me."}
    response = await client.post("/api/v1/tasks", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == TaskStatus.COMPLETED.value
    assert "calculator" in data["metadata"]["tools_used"]
    assert "tool_executing" in data["metadata"]["trace"]
    assert "tool_executed" in data["metadata"]["trace"]

