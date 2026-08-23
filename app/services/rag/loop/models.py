from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.services.rag.critic.models import SelfCorrectionResult
from app.services.rag.models import Citation
from app.services.rag.query_analysis import QueryAnalysis
from app.services.rag.research.models import (
    ResearchPlan,
    SubquestionExecutionResult,
)
from app.services.rag.research.store import ResearchEvidenceStore
from app.services.rag.verification.models import VerificationReport


class AgenticResearchLoopConfig(BaseModel):
    """Configuration options controlling the complete agentic research loop execution."""

    max_research_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum re-retrieval iterations when evidence is insufficient",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum concurrent research subquestions executed in parallel",
    )
    enable_hyde: bool = Field(
        default=True,
        description="Whether to enable Hypothetical Document Embeddings for subqueries",
    )
    enable_reranking: bool = Field(
        default=True,
        description="Whether to rerank candidate documents",
    )
    enable_self_correction: bool = Field(
        default=True,
        description="Whether to run the Critic Agent and bounded self-correction loop",
    )
    enable_verification: bool = Field(
        default=True,
        description="Whether to run claim-by-claim answer verification and sanitization",
    )
    timeout_seconds: float = Field(
        default=60.0,
        gt=0.0,
        description="Overall timeout for the complete research loop in seconds",
    )


class AgenticResearchLoopResult(BaseModel):
    """Full execution telemetry and results from the complete agentic research loop."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop_id: str = Field(description="Unique identifier for this research loop execution")
    question: str = Field(description="Original user question")
    query_analysis: QueryAnalysis = Field(description="Structured query analysis output")
    research_plan: ResearchPlan | None = Field(
        default=None,
        description="Topological subquestion research plan if multi-step",
    )
    subquestion_results: list[SubquestionExecutionResult] = Field(
        default_factory=list,
        description="Execution results for each research subquestion",
    )
    evidence_store: ResearchEvidenceStore = Field(
        description="Consolidated research evidence repository",
    )
    draft_answer: str = Field(description="Initial synthesized answer from evidence")
    self_correction_result: SelfCorrectionResult | None = Field(
        default=None,
        description="Telemetry from the self-correction critic loop if executed",
    )
    verification_report: VerificationReport | None = Field(
        default=None,
        description="Claim-level verification report if executed",
    )
    final_answer: str = Field(
        description="Final cited, self-corrected, and verified answer presented to user",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Consolidated citations supporting the final answer",
    )
    total_duration_ms: float = Field(
        default=0.0,
        description="Total duration of complete agentic research loop in milliseconds",
    )
    status: str = Field(
        default="completed",
        description="Overall execution status: completed, partial, or failed",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional loop telemetry",
    )
