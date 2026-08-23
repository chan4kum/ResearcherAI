import pytest
from app.models.schemas import TaskStatus
from app.services.agent.graph.nodes import (
    create_tool_decision_node,
    create_tool_executor_node,
    parse_plan,
)
from app.services.agent.graph.state import AgentGraphState
from app.services.agent.graph.workflow import build_agent_graph, route_tool_decision
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.mock import MockLLMProvider
from app.services.llm.service import LLMService


def test_agent_graph_construction() -> None:
    """Verify that the LangGraph workflow compiles with the expected nodes and edges."""
    llm_service = LLMService(provider=MockLLMProvider())
    graph = build_agent_graph(llm_service)

    assert graph is not None
    assert "planner" in graph.nodes
    assert "tool_decision" in graph.nodes
    assert "tool_node" in graph.nodes
    assert "answer_agent" in graph.nodes


def test_route_tool_decision() -> None:
    """Verify conditional router directs to tool_node when tool is chosen, else answer_agent."""
    state_with_tool: AgentGraphState = {
        "tool_call": {"tool_name": "calculator", "tool_args": {"expression": "25 * 4"}},
    }
    assert route_tool_decision(state_with_tool) == "tool_node"

    state_without_tool: AgentGraphState = {
        "tool_call": None,
    }
    assert route_tool_decision(state_without_tool) == "answer_agent"


@pytest.mark.asyncio
async def test_tool_decision_node_selects_tool() -> None:
    """Verify tool_decision node selects calculator for math query."""
    llm_service = LLMService(provider=MockLLMProvider())
    registry = ToolRegistry()
    node = create_tool_decision_node(llm_service, registry)

    state: AgentGraphState = {
        "task_id": "test-task-calc",
        "task": "Please calculate 25 * 4.",
        "plan": ["Calculate expression"],
        "trace": ["initialized", "planning", "planned"],
    }
    result = await node(state)
    assert result["tool_call"] is not None
    assert result["tool_call"]["tool_name"] == "calculator"
    assert "tool_decision" in result["trace"]


@pytest.mark.asyncio
async def test_tool_executor_node_executes_tool() -> None:
    """Verify tool_node executes calculator and updates tool_result."""
    registry = ToolRegistry()
    tool_node = create_tool_executor_node(registry)

    state: AgentGraphState = {
        "task_id": "test-task-exec",
        "tool_call": {"tool_name": "calculator", "tool_args": {"expression": "25 * 4"}},
        "trace": ["initialized", "planning", "planned", "tool_decision"],
    }
    result = await tool_node(state)
    assert result["tool_result"]["success"] is True
    assert result["tool_result"]["output"] == 100
    assert result["tools_used"] == ["calculator"]
    assert "tool_executing" in result["trace"]
    assert "tool_executed" in result["trace"]


@pytest.mark.asyncio
async def test_full_graph_non_tool_flow() -> None:
    """Verify full graph execution when no tool is required."""
    mock_provider = MockLLMProvider()
    llm_service = LLMService(provider=mock_provider)
    graph = build_agent_graph(llm_service)

    initial_state: AgentGraphState = {
        "task_id": "graph-non-tool-test",
        "task": "Explain how semiconductor manufacturing works.",
        "status": TaskStatus.PENDING.value,
        "plan": [],
        "tool_call": None,
        "tool_result": None,
        "tools_used": [],
        "answer": None,
        "trace": ["initialized"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    final_state: dict[str, object] = await graph.ainvoke(initial_state)

    assert final_state["status"] == TaskStatus.COMPLETED.value
    assert isinstance(final_state["plan"], list)
    assert len(final_state["plan"]) >= 2
    assert final_state["tools_used"] == []
    assert final_state["trace"] == [
        "initialized",
        "planning",
        "planned",
        "tool_decision",
        "executing",
        "executed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_full_graph_tool_invocation_flow() -> None:
    """Verify full graph execution with calculator tool invocation."""
    mock_provider = MockLLMProvider()
    llm_service = LLMService(provider=mock_provider)
    graph = build_agent_graph(llm_service)

    initial_state: AgentGraphState = {
        "task_id": "graph-tool-test",
        "task": "Please calculate 25 * 4 for me.",
        "status": TaskStatus.PENDING.value,
        "plan": [],
        "tool_call": None,
        "tool_result": None,
        "tools_used": [],
        "answer": None,
        "trace": ["initialized"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    final_state: dict[str, object] = await graph.ainvoke(initial_state)

    assert final_state["status"] == TaskStatus.COMPLETED.value
    assert final_state["tools_used"] == ["calculator"]
    assert final_state["tool_result"] is not None
    assert final_state["trace"] == [
        "initialized",
        "planning",
        "planned",
        "tool_decision",
        "tool_executing",
        "tool_executed",
        "executing",
        "executed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_full_graph_invalid_tool_handling() -> None:
    """Verify full graph handles invalid/unknown tool calls gracefully without crashing."""
    custom_responses = {
        "decide if a tool is required": '{"tool_name": "unknown_tool_xyz", "tool_args": {}}',
    }
    mock_provider = MockLLMProvider(custom_responses=custom_responses)
    llm_service = LLMService(provider=mock_provider)
    graph = build_agent_graph(llm_service)

    initial_state: AgentGraphState = {
        "task_id": "graph-invalid-tool-test",
        "task": "Run unknown operation",
        "status": TaskStatus.PENDING.value,
        "plan": [],
        "tool_call": None,
        "tool_result": None,
        "tools_used": [],
        "answer": None,
        "trace": ["initialized"],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }

    final_state: dict[str, object] = await graph.ainvoke(initial_state)
    assert final_state["status"] == TaskStatus.COMPLETED.value
    assert final_state["tool_result"] is not None
    assert final_state["tool_result"]["success"] is False  # type: ignore[index]
    assert "not found" in final_state["tool_result"]["error"]  # type: ignore[index]
    assert final_state["answer"] is not None


def test_parse_plan_utility() -> None:
    """Verify plan parser handles bullets, numbering, and empty fallbacks."""
    raw = (
        "1. Step one\n"
        "2) Step two\n"
        "- Step three\n"
        "* Step four\n"
        "• Step five\n"
        "Step 6: Step six"
    )
    steps = parse_plan(raw, "fallback")
    assert steps == ["Step one", "Step two", "Step three", "Step four", "Step five", "Step six"]

    empty_fallback = parse_plan("", "default goal")
    assert len(empty_fallback) == 1
    assert "default goal" in empty_fallback[0]
