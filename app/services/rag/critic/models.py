from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.services.rag.models import Citation


class CriticIssueType(StrEnum):
    """Categories of flaws evaluated by the Critic Agent."""

    UNSUPPORTED_CLAIM = "unsupported_claim"
    MISSING_EVIDENCE = "missing_evidence"
    CONTRADICTION = "contradiction"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    IRRELEVANT_INFORMATION = "irrelevant_information"
    CITATION_PROBLEM = "citation_problem"


class CriticIssueSeverity(StrEnum):
    """Severity levels for issues discovered by the Critic Agent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CriticIssue(BaseModel):
    """Specific defect identified in draft answer relative to evidence."""

    issue_type: CriticIssueType = Field(description="Classification of defect")
    severity: CriticIssueSeverity = Field(
        default=CriticIssueSeverity.MEDIUM,
        description="Impact severity on response validity",
    )
    claim_or_passage: str = Field(
        description="Flawed passage or claim in draft answer",
    )
    reason: str = Field(
        description="Explanation of why this constitutes an issue",
    )
    suggested_fix: str = Field(
        description="Actionable instruction to resolve the issue",
    )


class CriticEvaluation(BaseModel):
    """Complete evaluation report produced by the Critic Agent."""

    is_acceptable: bool = Field(
        description="True if draft passes quality and grounding criteria",
    )
    critique_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall quality score (0.0 = severe flaws, 1.0 = flawless)",
    )
    issues: list[CriticIssue] = Field(
        default_factory=list,
        description="List of specific flaws discovered",
    )
    feedback_summary: str = Field(
        description="High-level feedback summary for revision",
    )
    action_recommended: str = Field(
        default="accept",
        description="Recommended action: 'accept', 'revise_answer', 'retrieve_more_evidence'",
    )


class SelfCorrectionAttempt(BaseModel):
    """Record of a single self-correction iteration."""

    iteration: int = Field(description="1-indexed iteration counter")
    draft_answer: str = Field(description="Draft answer evaluated in this attempt")
    evaluation: CriticEvaluation = Field(description="Critic report for this draft")
    revised_answer: str | None = Field(
        default=None,
        description="Revised answer generated if revision was needed",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time spent on this self-correction attempt in ms",
    )


class SelfCorrectionResult(BaseModel):
    """Overall outcome of the self-correction loop."""

    question: str = Field(description="Original user question")
    original_draft: str = Field(description="Initial uncorrected draft answer")
    final_answer: str = Field(description="Final corrected output answer")
    iterations: int = Field(description="Number of correction iterations performed")
    max_iterations: int = Field(description="Configured maximum allowed iterations")
    is_corrected: bool = Field(description="True if modifications were made")
    final_evaluation: CriticEvaluation = Field(
        description="Critic evaluation of final answer",
    )
    attempts: list[SelfCorrectionAttempt] = Field(
        default_factory=list,
        description="Sequential trace of all correction attempts",
    )
    total_duration_ms: float = Field(
        description="Total elapsed time for self-correction in ms",
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Retained citations supporting the answer",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional telemetry metadata",
    )
