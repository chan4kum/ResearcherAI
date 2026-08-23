from enum import StrEnum

from pydantic import BaseModel, Field

from app.services.rag.query_analysis import QueryIntent


class SourceDestination(StrEnum):
    """Categorical destination sources for query routing."""

    INTERNAL_DOCUMENTS = "internal_documents"
    EXTERNAL_WEB = "external_web"
    STRUCTURED_DATABASE = "structured_database"


class RoutingDecision(BaseModel):
    """Structured decision determining the target retrieval sources for a query."""

    query: str = Field(description="Query string routed")
    intent: QueryIntent = Field(description="Analyzed query intent")
    selected_sources: list[SourceDestination] = Field(
        description="List of selected knowledge sources to query"
    )
    reason: str = Field(description="Justification and rationale for the routing decision")
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score for the selected routing decision",
    )
    entities_detected: list[str] = Field(
        default_factory=list,
        description="Named entities identified and considered during routing",
    )
