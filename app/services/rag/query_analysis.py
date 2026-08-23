from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class QueryIntent(StrEnum):
    """Semantic intent classification for user queries."""

    FACTUAL = "factual"
    COMPARISON = "comparison"
    MULTI_PART_RESEARCH = "multi_part_research"
    AMBIGUOUS = "ambiguous"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"


class ExtractedEntity(BaseModel):
    """Named entity, product, metric, or concept extracted from query."""

    text: str = Field(description="Exact entity string mentioned in query")
    label: str = Field(
        description="Entity classification label (e.g. organization, product, metric)"
    )
    category: str | None = Field(default=None, description="Broad ontological category")


class QueryAnalysis(BaseModel):
    """Structured understanding and decomposition of a user query."""

    original_query: str = Field(description="Raw user query analyzed")
    intent: QueryIntent = Field(description="Primary semantic query intent")
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted entities, organizations, products, and technical terms",
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description="Atomic subquestions necessary to answer the overall query",
    )
    required_information_types: list[str] = Field(
        default_factory=list,
        description="Types of information required (e.g. metrics, causes, timelines)",
    )
    potential_source_types: list[str] = Field(
        default_factory=list,
        description="Candidate document and source types likely to contain answers",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="Whether query lacks sufficient specificity or context",
    )
    clarification_needed: str | None = Field(
        default=None,
        description="Follow-up clarification prompt if query is ambiguous",
    )
    temporal_scope: str | None = Field(
        default=None,
        description="Timeframe or date range identified in the query",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the intent analysis [0.0 - 1.0]",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional heuristic or diagnostic metadata",
    )

    @property
    def is_complex(self) -> bool:
        """Determine if query is complex requiring multi-step planning."""
        return (
            self.intent in (QueryIntent.COMPARISON, QueryIntent.MULTI_PART_RESEARCH)
            or len(self.subquestions) > 1
        )
