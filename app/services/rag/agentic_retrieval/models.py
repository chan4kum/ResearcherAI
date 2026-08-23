from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.services.rag.models import Citation
from app.services.rag.routing import SourceDestination


class RetrievalStepType(StrEnum):
    """Categorical types of actions in the agentic retrieval loop."""

    ANALYZE = "analyze"
    PLAN = "plan"
    SELECT_SOURCE = "select_source"
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    REWRITE = "rewrite"
    SYNTHESIZE = "synthesize"
    TERMINATE = "terminate"


class RetrievalPlan(BaseModel):
    """Initial strategy and subgoals produced by the Retrieval Planner."""

    needs_retrieval: bool = Field(
        description="Whether answering the query requires external knowledge retrieval"
    )
    subgoals: list[str] = Field(
        default_factory=list,
        description="Sub-tasks or subquestions required to answer the query",
    )
    target_sources: list[SourceDestination] = Field(
        default_factory=list,
        description="Target source destinations planned for lookup",
    )
    planned_queries: list[str] = Field(
        default_factory=list,
        description="Initial search queries mapped to the subgoals",
    )
    rationale: str = Field(description="Planner justification for the execution path")


class RetrievalTraceStep(BaseModel):
    """Atomic step recorded in the agentic retrieval trace."""

    step_index: int = Field(description="Sequential step index")
    step_type: RetrievalStepType = Field(description="Action classification")
    query: str = Field(description="Active query or subquery at this step")
    sources_contacted: list[str] = Field(
        default_factory=list,
        description="Names of retrieval sources invoked in this step",
    )
    documents_retrieved_count: int = Field(
        default=0,
        description="Number of document snippets returned in this step",
    )
    evaluation_summary: dict[str, Any] | None = Field(
        default=None,
        description="Evaluation metrics if step was an evaluation",
    )
    decision: str = Field(description="Agent rationale or next-action decision")
    duration_ms: float = Field(
        default=0.0,
        description="Execution duration for this step in milliseconds",
    )


class AgenticRetrievalTrace(BaseModel):
    """Telemetry log persisting the complete execution path of an agentic retrieval session."""

    session_id: str = Field(description="Unique session or trace identifier")
    original_query: str = Field(description="Original user question")
    total_iterations: int = Field(
        default=0, description="Total retrieval iteration rounds executed"
    )
    total_tool_calls: int = Field(default=0, description="Total source tool queries invoked")
    total_documents_retrieved: int = Field(
        default=0, description="Total cumulative document passages retrieved"
    )
    duration_ms: float = Field(default=0.0, description="Total elapsed time in milliseconds")
    termination_reason: str = Field(
        description="Reason the loop ended (e.g. EVIDENCE_SUFFICIENT, MAX_ITERATIONS, TIMEOUT)"
    )
    steps: list[RetrievalTraceStep] = Field(
        default_factory=list,
        description="Chronological sequence of all execution steps",
    )


class AgenticRetrievalResult(BaseModel):
    """Final output produced by the Agent-Driven Retrieval Loop."""

    query: str = Field(description="Original user query")
    answer: str = Field(description="Grounded final answer synthesized by the agent")
    is_sufficient: bool = Field(description="Whether evidence satisfied all sufficiency criteria")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source citations backing the generated answer",
    )
    trace: AgenticRetrievalTrace = Field(
        description="Complete execution trace of the retrieval process"
    )
