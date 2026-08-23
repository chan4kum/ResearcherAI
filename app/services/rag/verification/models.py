from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ClaimSupportStatus(StrEnum):
    """Verification outcomes for factual claims evaluated against evidence."""

    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class FactualClaim(BaseModel):
    """Structured representation of an atomic factual assertion evaluated against evidence."""

    claim_id: str = Field(description="Unique claim identifier")
    claim_text: str = Field(description="Atomic factual proposition evaluated")
    evidence_text: str | None = Field(
        default=None,
        description="Verbatim supporting or contradictory evidence snippet",
    )
    source: str | None = Field(
        default=None,
        description="Source document name or provenance path",
    )
    support_status: ClaimSupportStatus = Field(
        description="Verification outcome for this claim",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the verification assessment (0.0 to 1.0)",
    )
    citation_chunk_id: str | None = Field(
        default=None,
        description="Referenced chunk identifier if citation tag was present",
    )
    reason: str = Field(
        default="",
        description="Explanation justifying the support status assignment",
    )


class VerificationReport(BaseModel):
    """Consolidated verification assessment for an entire answer."""

    report_id: str = Field(description="Unique report identifier")
    question: str = Field(description="Original user question")
    original_answer: str = Field(description="Input draft answer evaluated")
    verified_answer: str = Field(
        description="Sanitized final answer ensuring unsupported claims are not presented as facts",
    )
    total_claims: int = Field(description="Total number of factual claims extracted")
    claims: list[FactualClaim] = Field(
        default_factory=list,
        description="Detailed claim-by-claim verification assessments",
    )
    supported_count: int = Field(default=0, description="Count of SUPPORTED claims")
    partially_supported_count: int = Field(
        default=0,
        description="Count of PARTIALLY_SUPPORTED claims",
    )
    unsupported_count: int = Field(default=0, description="Count of UNSUPPORTED claims")
    contradicted_count: int = Field(
        default=0,
        description="Count of CONTRADICTED claims",
    )
    verified_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Proportion of claims that are supported or partially supported",
    )
    is_verified: bool = Field(
        description="True if answer has zero unsupported and zero contradicted claims",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Time taken to execute verification in milliseconds",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional verification telemetry",
    )
