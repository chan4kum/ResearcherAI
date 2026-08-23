from enum import StrEnum

from pydantic import BaseModel, Field

from app.services.rag.models import Citation
from app.services.rag.routing import SourceDestination


class ResearchSubquestionType(StrEnum):
    """Categorical type classification for a planned research subquestion."""

    BACKGROUND = "background"
    FACTUAL = "factual"
    STRATEGY = "strategy"
    CHALLENGE = "challenge"
    COMPARISON = "comparison"
    IMPLICATION = "implication"
    SYNTHESIS = "synthesis"


class SubquestionExecutionStatus(StrEnum):
    """Execution status for an individual research subquestion."""

    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchSubquestion(BaseModel):
    """Structured atomic subquestion within a broader research plan."""

    id: str = Field(
        description="Unique identifier for the subquestion (e.g. subq_1)",
        examples=["subq_1"],
    )
    index: int = Field(
        description="1-based execution sequence index",
        examples=[1],
    )
    question: str = Field(
        description="Focused subquestion text",
        examples=["What is TSMC's current advanced node manufacturing strategy?"],
    )
    subquestion_type: ResearchSubquestionType = Field(
        description="Classification of the subquestion purpose",
    )
    target_entities: list[str] = Field(
        default_factory=list,
        description="Specific entities focused in this subquestion",
        examples=[["TSMC"]],
    )
    expected_output_type: str = Field(
        default="summary",
        description="Anticipated information format (overview, metrics, risks, comparison)",
    )
    suggested_sources: list[SourceDestination] = Field(
        default_factory=list,
        description="Target knowledge destinations recommended for answering this subquestion",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of subquestion IDs that must be answered before this one",
    )


class ResearchPlan(BaseModel):
    """Comprehensive structured research plan breaking down a complex multi-part inquiry."""

    plan_id: str = Field(
        description="Unique identifier for this research plan",
        examples=["plan_tsmc_intel_99"],
    )
    original_query: str = Field(
        description="Original complex user inquiry",
    )
    overall_goal: str = Field(
        description="Synthesized high-level objective of the research inquiry",
    )
    subquestions: list[ResearchSubquestion] = Field(
        default_factory=list,
        description="Chronological and dependency-ordered subquestions to resolve",
    )
    estimated_complexity: str = Field(
        default="high",
        description="Estimated inquiry complexity (low, medium, high, complex)",
    )
    suggested_synthesis_strategy: str = Field(
        description="Recommended approach for synthesizing subquestion findings into final report",
    )
    created_at: str = Field(
        description="ISO formatted timestamp of plan formulation",
    )


class SubquestionExecutionResult(BaseModel):
    """Retained execution trace, evidence, and intermediate answer for a research subquestion."""

    subquestion_id: str = Field(description="Identifier of the executed subquestion")
    index: int = Field(description="Sequential execution index")
    query: str = Field(description="Specific subquestion query executed")
    sources: list[str] = Field(
        default_factory=list,
        description="Retrieval source names utilized",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Retrieved textual evidence snippets",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Grounded citation provenance for this subquestion",
    )
    sub_answer: str = Field(
        default="",
        description="Intermediate synthesized answer for this subquestion",
    )
    status: SubquestionExecutionStatus = Field(
        default=SubquestionExecutionStatus.PENDING,
        description="Execution outcome status",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Execution duration in milliseconds",
    )
    error: str | None = Field(
        default=None,
        description="Error description if execution failed",
    )


class ResearchExecutionResult(BaseModel):
    """Complete multi-step research outcome containing evidence and final synthesis."""

    research_id: str = Field(description="Unique research execution identifier")
    original_query: str = Field(description="Original complex user research inquiry")
    plan: ResearchPlan = Field(description="Structured research plan executed")
    subquestion_results: list[SubquestionExecutionResult] = Field(
        default_factory=list,
        description="Sequential list of all subquestion execution outcomes",
    )
    final_synthesis: str = Field(
        description="Holistic final report synthesizing findings across all subquestions",
    )
    total_citations: list[Citation] = Field(
        default_factory=list,
        description="Deduplicated consolidated citations across all executed subquestions",
    )
    total_duration_ms: float = Field(
        description="Total elapsed execution time in milliseconds",
    )
    status: str = Field(
        default="completed",
        description="Overall research status (completed, partial, failed)",
    )


class ParallelResearchConfig(BaseModel):
    """Configuration guardrails for parallel multi-step research execution."""

    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent subquestion executions allowed",
    )
    subquestion_timeout_seconds: float = Field(
        default=5.0,
        ge=0.01,
        le=30.0,
        description="Timeout per individual subquestion execution in seconds",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries for transient subquestion execution errors",
    )
    retry_delay_seconds: float = Field(
        default=0.05,
        ge=0.0,
        le=5.0,
        description="Delay in seconds between retry attempts",
    )


