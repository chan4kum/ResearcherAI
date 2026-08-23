from typing import Any, TypedDict


class AgentGraphState(TypedDict, total=False):
    """Schema representing the state dictionary flowing through LangGraph nodes."""

    task_id: str
    task: str
    system_instructions: str | None
    status: str
    plan: list[str]
    tool_call: dict[str, Any] | None
    tool_result: dict[str, Any] | None
    tools_used: list[str]
    answer: str | None
    error: str | None
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    trace: list[str]
    duration_ms: float
