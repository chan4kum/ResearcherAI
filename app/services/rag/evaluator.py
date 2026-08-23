from enum import StrEnum

from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.services.rag.bm25 import tokenize
from app.services.rag.models import Citation
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent

logger = get_logger("app.services.rag.evaluator")


class EvaluationReason(StrEnum):
    """Categorical reasons for retrieval sufficiency or deficiency."""

    SUFFICIENT = "sufficient"
    LOW_RELEVANCE = "low_relevance"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    MISSING_ENTITIES = "missing_entities"
    AMBIGUITY = "ambiguity"
    POOR_COVERAGE = "poor_coverage"


class RetrievalEvaluation(BaseModel):
    """Detailed evaluation of retrieved candidates against analyzed query intent."""

    is_sufficient: bool = Field(
        description="Whether retrieved evidence is sufficient to ground an accurate answer"
    )
    reasons: list[EvaluationReason] = Field(
        default_factory=list,
        description="List of detected retrieval evaluation deficiency flags",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Top relevance or similarity score among retrieved chunks",
    )
    entity_coverage: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Proportion of key query entities found across retrieved chunks",
    )
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Entities present in the query but absent from retrieved chunks",
    )
    feedback_prompt: str | None = Field(
        default=None,
        description="Prescriptive guidance for the query rewriter to address deficiencies",
    )


class RetrievalEvaluator:
    """Evaluates retrieved citation candidates against semantic query analysis."""

    def evaluate(
        self,
        query: str,
        analysis: QueryAnalysis,
        citations: list[Citation],
        min_relevance: float = 0.01,
        min_evidence: int = 1,
    ) -> RetrievalEvaluation:
        """Assess relevance, entity presence, and evidence coverage."""
        reasons: list[EvaluationReason] = []
        feedback_parts: list[str] = []

        # 1. Check for insufficient evidence (zero or too few citations)
        if not citations or len(citations) < min_evidence:
            reasons.append(EvaluationReason.INSUFFICIENT_EVIDENCE)
            feedback_parts.append(
                f"No or insufficient candidate chunks retrieved (found {len(citations)})."
            )
            return RetrievalEvaluation(
                is_sufficient=False,
                reasons=reasons,
                relevance_score=0.0,
                entity_coverage=0.0,
                missing_entities=[e.text for e in analysis.entities],
                feedback_prompt="Expand terminology, use synonyms, and remove restrictive filters.",
            )

        # 2. Check relevance score threshold
        top_score = max(c.similarity for c in citations)
        if top_score < min_relevance:
            reasons.append(EvaluationReason.LOW_RELEVANCE)
            feedback_parts.append(
                f"Top similarity score ({top_score:.4f}) is below threshold ({min_relevance:.4f})."
            )

        # 3. Check entity coverage
        all_chunk_text = " ".join(c.content.lower() for c in citations)
        chunk_tokens = set(tokenize(all_chunk_text))

        missing_entities: list[str] = []
        target_labels = ("organization", "product", "metric_issue")
        key_entities = [
            e.text for e in analysis.entities if e.label in target_labels
        ]
        for ent in key_entities:
            ent_tokens = tokenize(ent.lower())
            if not any(t in chunk_tokens or t in all_chunk_text for t in ent_tokens):
                missing_entities.append(ent)

        total_entities_count = len(key_entities) or 1
        found_entities_count = total_entities_count - len(missing_entities)
        entity_coverage = round(found_entities_count / total_entities_count, 2)

        if missing_entities:
            reasons.append(EvaluationReason.MISSING_ENTITIES)
            feedback_parts.append(
                f"Missing critical entities in retrieved context: {', '.join(missing_entities)}."
            )

        # 4. Check coverage for comparison queries (both subjects must be present)
        if analysis.intent == QueryIntent.COMPARISON:
            orgs = [e.text for e in analysis.entities if e.label == "organization"]
            if len(orgs) >= 2:
                missing_orgs = [
                    org for org in orgs
                    if not any(org.lower() in c.content.lower() for c in citations)
                ]
                if missing_orgs:
                    reasons.append(EvaluationReason.POOR_COVERAGE)
                    feedback_parts.append(
                        f"Comparison query missing coverage for subject: {', '.join(missing_orgs)}."
                    )

        # 5. Check ambiguity
        if analysis.is_ambiguous:
            reasons.append(EvaluationReason.AMBIGUITY)
            feedback_parts.append("Original query was flagged as ambiguous or underspecified.")

        is_sufficient = len(reasons) == 0
        if is_sufficient:
            reasons.append(EvaluationReason.SUFFICIENT)

        feedback_prompt = " ".join(feedback_parts) if feedback_parts else None

        logger.info(
            "retrieval_evaluation_completed",
            is_sufficient=is_sufficient,
            reasons=[r.value for r in reasons],
            top_score=top_score,
            entity_coverage=entity_coverage,
        )

        return RetrievalEvaluation(
            is_sufficient=is_sufficient,
            reasons=reasons,
            relevance_score=top_score,
            entity_coverage=entity_coverage,
            missing_entities=missing_entities,
            feedback_prompt=feedback_prompt,
        )
