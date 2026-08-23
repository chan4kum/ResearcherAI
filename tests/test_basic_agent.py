import pytest
from app.models.schemas import TaskStatus
from app.services.agent.agent import BasicAgent
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService


@pytest.mark.asyncio
async def test_basic_agent_successful_run() -> None:
    """Verify BasicAgent executes planning and execution stages to produce an answer."""
    mock_provider = MockLLMProvider()
    llm_service = LLMService(provider=mock_provider)
    agent = BasicAgent(llm_service=llm_service)

    state = await agent.run(task="Explain how semiconductor manufacturing works.")

    assert state.status == TaskStatus.COMPLETED
    assert state.task == "Explain how semiconductor manufacturing works."
    assert len(state.plan) >= 2
    assert state.answer is not None
    assert len(state.answer) > 0
    assert state.duration_ms > 0
    assert state.total_tokens > 0

    # Verify execution trace order
    assert state.trace == [
        "initialized",
        "planning",
        "planned",
        "tool_decision",
        "executing",
        "executed",
        "completed",
    ]

    # Verify 3 LLM calls were made (planning, tool decision, execution)
    assert len(mock_provider.calls) == 3
    assert "PLANNER" in (mock_provider.calls[0]["system_prompt"] or "")
    assert "TOOL_DECIDER" in (mock_provider.calls[1]["system_prompt"] or "")
    assert "EXECUTOR" in (mock_provider.calls[2]["system_prompt"] or "")


@pytest.mark.asyncio
async def test_basic_agent_plan_parsing() -> None:
    """Verify BasicAgent correctly strips numbers and bullets during plan parsing."""
    agent = BasicAgent(llm_service=LLMService(provider=MockLLMProvider()))

    raw_plan = (
        "1. Step one\n"
        "2) Step two\n"
        "- Step three\n"
        "* Step four\n"
        "Step 5: Step five"
    )
    parsed = agent._parse_plan(raw_plan, "fallback")
    assert parsed == ["Step one", "Step two", "Step three", "Step four", "Step five"]


@pytest.mark.asyncio
async def test_basic_agent_handles_llm_failure() -> None:
    """Verify BasicAgent captures failures and marks state as FAILED."""
    failing_provider = MockLLMProvider(should_fail=True, failure_message="Model provider timeout")
    agent = BasicAgent(llm_service=LLMService(provider=failing_provider))

    state = await agent.run(task="Analyze this task")

    assert state.status == TaskStatus.FAILED
    assert "Model provider timeout" in (state.error or "")
    assert state.answer is None
    assert "failed" in state.trace
    assert state.duration_ms >= 0
