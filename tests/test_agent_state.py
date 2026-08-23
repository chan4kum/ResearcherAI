from app.models.schemas import TaskStatus
from app.services.agent.state import AgentState


def test_agent_state_initialization() -> None:
    """Verify initial AgentState properties and defaults."""
    state = AgentState(task="Analyze quarterly financials")
    assert state.task == "Analyze quarterly financials"
    assert state.status == TaskStatus.PENDING
    assert state.plan == []
    assert state.answer is None
    assert state.error is None
    assert state.trace == []
    assert state.task_id


def test_agent_state_trace_and_usage() -> None:
    """Verify trace appending and token usage accumulation in AgentState."""
    state = AgentState(task="Test task")
    state.add_trace("step_1")
    state.add_trace("step_2")
    assert state.trace == ["step_1", "step_2"]

    state.record_usage(
        model="gpt-4o-mini",
        provider="mock",
        prompt_tokens=15,
        completion_tokens=25,
        total_tokens=40,
    )
    state.record_usage(
        model="gpt-4o-mini",
        provider="mock",
        prompt_tokens=20,
        completion_tokens=30,
        total_tokens=50,
    )

    assert state.prompt_tokens == 35
    assert state.completion_tokens == 55
    assert state.total_tokens == 90
    assert state.model == "gpt-4o-mini"
    assert state.provider == "mock"


def test_agent_state_to_response_conversion() -> None:
    """Verify conversion from AgentState to external TaskResponse contract."""
    state = AgentState(
        task="Explain lithography",
        status=TaskStatus.COMPLETED,
        plan=["Step 1: Prep", "Step 2: Expose", "Step 3: Etch"],
        answer="Photolithography uses light to transfer geometric patterns.",
        model="mock-model",
        provider="mock",
        duration_ms=45.2,
    )
    state.add_trace("initialized")
    state.add_trace("completed")

    response = state.to_response()
    assert response.task_id == state.task_id
    assert response.task == "Explain lithography"
    assert response.status == TaskStatus.COMPLETED
    assert len(response.plan) == 3
    assert response.answer == "Photolithography uses light to transfer geometric patterns."
    assert response.metadata.model == "mock-model"
    assert response.metadata.provider == "mock"
    assert response.metadata.duration_ms == 45.2
    assert response.metadata.trace == ["initialized", "completed"]
