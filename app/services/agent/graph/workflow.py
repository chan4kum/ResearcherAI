from typing import Any

from langgraph.graph import END, START, StateGraph

from app.core.versioning.prompts import PromptRegistry
from app.services.agent.graph.nodes import (
    create_answer_node,
    create_planner_node,
    create_tool_decision_node,
    create_tool_executor_node,
)
from app.services.agent.graph.state import AgentGraphState
from app.services.agent.tools.registry import ToolRegistry
from app.services.llm.service import LLMService


def route_tool_decision(state: AgentGraphState) -> str:
    """Conditional router determining whether to route to tool executor or answer agent."""
    tool_call = state.get("tool_call")
    if tool_call and tool_call.get("tool_name"):
        return "tool_node"
    return "answer_agent"


def build_agent_graph(
    llm_service: LLMService,
    tool_registry: ToolRegistry | None = None,
    prompt_registry: PromptRegistry | None = None,
) -> Any:
    """Construct and compile the LangGraph agent workflow with tool decision and execution.

    Topology:
    START
      ↓
    planner
      ↓
    tool_decision
      ↓ (conditional)
      ├─ [tool needed] ──> tool_node ──> answer_agent ──> END
      └─ [no tool needed] ────────────> answer_agent ──> END
    """
    registry = tool_registry or ToolRegistry()
    workflow = StateGraph(AgentGraphState)

    # Register workflow nodes with versioned prompts
    workflow.add_node("planner", create_planner_node(llm_service, prompt_registry=prompt_registry))
    workflow.add_node(
        "tool_decision",
        create_tool_decision_node(llm_service, registry, prompt_registry=prompt_registry),
    )
    workflow.add_node("tool_node", create_tool_executor_node(registry))
    workflow.add_node("answer_agent", create_answer_node(llm_service, prompt_registry=prompt_registry))

    # Define edges and conditional routing
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "tool_decision")
    workflow.add_conditional_edges(
        "tool_decision",
        route_tool_decision,
        {
            "tool_node": "tool_node",
            "answer_agent": "answer_agent",
        },
    )
    workflow.add_edge("tool_node", "answer_agent")
    workflow.add_edge("answer_agent", END)

    return workflow.compile()
