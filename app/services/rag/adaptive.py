from enum import StrEnum

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.query_analysis import QueryAnalysis
from app.services.rag.router import RetrievalRouter
from app.services.rag.routing import RoutingDecision, SourceDestination
from app.services.rag.sources.models import SourceResult, SourceType

logger = get_logger("app.services.rag.adaptive")


class EvidenceSufficiencyStatus(StrEnum):
    """Categorical status of evidence sufficiency."""

    SUFFICIENT = "sufficient"
    NEEDS_MORE_RETRIEVAL = "needs_more_retrieval"


class EvidenceEvaluation(BaseModel):
    """Multi-dimensional evaluation of retrieved knowledge evidence."""

    status: EvidenceSufficiencyStatus = Field(
        description="Whether evidence is sufficient or needs more retrieval"
    )
    is_sufficient: bool = Field(
        description="Boolean indicating whether answer generation can proceed"
    )
    relevance_score: float = Field(
        description="Mean relevance/similarity score across retrieved results [0.0 - 1.0]"
    )
    coverage_score: float = Field(
        description="Ratio of query entities and concepts covered in evidence [0.0 - 1.0]"
    )
    source_diversity_score: float = Field(
        description="Score reflecting representation across expected knowledge sources [0.0 - 1.0]"
    )
    evidence_quantity: int = Field(
        description="Total number of evidence snippets retrieved"
    )
    confidence: float = Field(
        description="Composite confidence score for evidence sufficiency [0.0 - 1.0]"
    )
    reason: str = Field(description="Detailed explanation justifying the sufficiency evaluation")
    missing_entities: list[str] = Field(
        default_factory=list,
        description="Query entities not located in retrieved passages",
    )
    missing_sources: list[str] = Field(
        default_factory=list,
        description="Target source destinations with zero retrieved evidence",
    )


class AdaptiveRetrievalResult(BaseModel):
    """Result returned by the adaptive retrieval pipeline."""

    query: str = Field(description="Original query processed")
    status: EvidenceSufficiencyStatus = Field(description="Final sufficiency status")
    rounds_executed: int = Field(description="Total retrieval rounds executed")
    max_rounds: int = Field(description="Maximum allowed retrieval rounds")
    evaluation: EvidenceEvaluation = Field(description="Final evidence evaluation metrics")
    sources_queried: list[str] = Field(description="List of knowledge source names queried")
    results: list[SourceResult] = Field(
        default_factory=list,
        description="All retrieved source results",
    )
    answer: str | None = Field(
        default=None,
        description="Synthesized LLM answer if evidence was sufficient",
    )


class EvidenceEvaluator:
    """Evaluates multi-dimensional evidence quality against query requirements."""

    def __init__(
        self,
        min_relevance_threshold: float = 0.40,
        min_coverage_threshold: float = 0.50,
        min_confidence_threshold: float = 0.55,
    ) -> None:
        self._min_relevance = min_relevance_threshold
        self._min_coverage = min_coverage_threshold
        self._min_confidence = min_confidence_threshold

    def evaluate(
        self,
        query: str,
        analysis: QueryAnalysis,
        routing: RoutingDecision,
        results: list[SourceResult],
    ) -> EvidenceEvaluation:
        """Evaluate evidence quality across relevance, coverage, diversity, and quantity."""
        quantity = len(results)
        if quantity == 0:
            return EvidenceEvaluation(
                status=EvidenceSufficiencyStatus.NEEDS_MORE_RETRIEVAL,
                is_sufficient=False,
                relevance_score=0.0,
                coverage_score=0.0,
                source_diversity_score=0.0,
                evidence_quantity=0,
                confidence=0.0,
                reason="No evidence retrieved from any candidate knowledge source.",
                missing_entities=[e.text for e in analysis.entities],
                missing_sources=[s.value for s in routing.selected_sources],
            )

        # 1. Relevance Score
        mean_relevance = sum(r.relevance for r in results) / quantity
        max_relevance = max(r.relevance for r in results)
        effective_relevance = round((mean_relevance * 0.4) + (max_relevance * 0.6), 4)

        # 2. Entity & Concept Coverage
        combined_text = " ".join(r.content.lower() for r in results)
        missing_entities: list[str] = []
        if analysis.entities:
            for entity in analysis.entities:
                if entity.text.lower() not in combined_text:
                    missing_entities.append(entity.text)
            coverage_score = round(
                (len(analysis.entities) - len(missing_entities)) / len(analysis.entities), 4
            )
        else:
            coverage_score = 1.0

        # 3. Source Diversity
        retrieved_source_types = {r.source_type for r in results}
        missing_sources: list[str] = []
        for dest in routing.selected_sources:
            if dest == SourceDestination.INTERNAL_DOCUMENTS:
                if not (
                    SourceType.INTERNAL_VECTOR in retrieved_source_types
                    or SourceType.KEYWORD in retrieved_source_types
                ):
                    missing_sources.append(dest.value)
            elif dest == SourceDestination.EXTERNAL_WEB:
                if SourceType.WEB_SEARCH not in retrieved_source_types:
                    missing_sources.append(dest.value)
            elif dest == SourceDestination.STRUCTURED_DATABASE:
                if SourceType.STRUCTURED_DB not in retrieved_source_types:
                    missing_sources.append(dest.value)

        diversity_ratio = (
            (len(routing.selected_sources) - len(missing_sources)) / len(routing.selected_sources)
            if routing.selected_sources
            else 1.0
        )
        source_diversity_score = round(diversity_ratio, 4)

        # 4. Composite Confidence Calculation
        confidence = round(
            (effective_relevance * 0.45)
            + (coverage_score * 0.35)
            + (source_diversity_score * 0.20),
            4,
        )

        # 5. Sufficiency Decision Rules
        is_sufficient = True
        reasons: list[str] = []

        if effective_relevance < self._min_relevance:
            is_sufficient = False
            reasons.append(
                f"Low relevance score ({effective_relevance:.2f} < {self._min_relevance:.2f})"
            )

        if coverage_score < self._min_coverage and missing_entities:
            is_sufficient = False
            reasons.append(f"Missing critical entities: {', '.join(missing_entities)}")

        if missing_sources:
            is_sufficient = False
            reasons.append(f"Missing required sources: {', '.join(missing_sources)}")

        if confidence < self._min_confidence:
            is_sufficient = False
            reasons.append(f"Overall confidence too low ({confidence:.2f})")

        status = (
            EvidenceSufficiencyStatus.SUFFICIENT
            if is_sufficient
            else EvidenceSufficiencyStatus.NEEDS_MORE_RETRIEVAL
        )
        reason_str = (
            "Evidence meets all relevance, coverage, and source diversity requirements."
            if is_sufficient
            else f"Evidence insufficient: {'; '.join(reasons)}."
        )

        logger.info(
            "evidence_evaluation_completed",
            query=query[:80],
            status=status.value,
            relevance=effective_relevance,
            coverage=coverage_score,
            diversity=source_diversity_score,
            confidence=confidence,
        )

        return EvidenceEvaluation(
            status=status,
            is_sufficient=is_sufficient,
            relevance_score=effective_relevance,
            coverage_score=coverage_score,
            source_diversity_score=source_diversity_score,
            evidence_quantity=quantity,
            confidence=confidence,
            reason=reason_str,
            missing_entities=missing_entities,
            missing_sources=missing_sources,
        )


class AdaptiveRetriever:
    """Executes adaptive routing, retrieval, evaluation, and conditional answer generation."""

    def __init__(
        self,
        router: RetrievalRouter | None = None,
        evaluator: EvidenceEvaluator | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._router = router or RetrievalRouter(
            llm_service=self._llm_service, settings=self._settings
        )
        self._evaluator = evaluator or EvidenceEvaluator(
            min_relevance_threshold=getattr(
                self._settings, "min_retrieval_relevance_threshold", 0.35
            ),
            min_coverage_threshold=getattr(
                self._settings, "min_entity_coverage_threshold", 0.50
            ),
        )

    async def retrieve_adaptively(
        self,
        query: str,
        max_rounds: int = 2,
        top_k_per_source: int = 3,
        generate_answer: bool = True,
    ) -> AdaptiveRetrievalResult:
        """Run adaptive retrieval loop with bounded iterations and multi-factor evaluation."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty for adaptive retrieval.")

        analysis = await self._router._query_analyzer.analyze(clean_query)
        routing = await self._router.route(clean_query)

        rounds_executed = 0
        all_results: list[SourceResult] = []
        queried_source_names: set[str] = set()

        while rounds_executed < max_rounds:
            rounds_executed += 1
            logger.info(
                "adaptive_retrieval_round_started",
                round=rounds_executed,
                max_rounds=max_rounds,
                query=clean_query[:80],
            )

            # Retrieve from target sources
            decision, new_results = await self._router.route_and_retrieve(
                query=clean_query,
                top_k_per_source=top_k_per_source,
                min_relevance=0.0,
            )
            for r in new_results:
                queried_source_names.add(r.source)

            # Deduplicate by content & source
            existing_signatures = {f"{r.source}:{r.content[:50]}" for r in all_results}
            for r in new_results:
                sig = f"{r.source}:{r.content[:50]}"
                if sig not in existing_signatures:
                    all_results.append(r)
                    existing_signatures.add(sig)

            # Evaluate cumulative evidence
            evaluation = self._evaluator.evaluate(
                query=clean_query,
                analysis=analysis,
                routing=routing,
                results=all_results,
            )

            if evaluation.is_sufficient:
                logger.info(
                    "adaptive_evidence_sufficient",
                    round=rounds_executed,
                    confidence=evaluation.confidence,
                )
                break

            # If insufficient and more rounds allowed, attempt broader retrieval
            if rounds_executed < max_rounds and evaluation.missing_sources:
                logger.info(
                    "adaptive_expanding_sources",
                    missing_sources=evaluation.missing_sources,
                )

        # Generate answer if evidence is sufficient and requested
        answer: str | None = None
        if evaluation.is_sufficient and generate_answer:
            context_blocks = "\n\n".join(
                f"[{idx}] Source: {r.source} ({r.source_type.value})\n{r.content}"
                for idx, r in enumerate(all_results, start=1)
            )
            prompt = (
                f"Question: {clean_query}\n\n"
                f"Retrieved Evidence:\n{context_blocks}\n\n"
                f"Provide an evidence-based answer citing sources."
            )
            llm_res = await self._llm_service.generate(
                prompt=prompt,
                system_prompt="You are a helpful research assistant synthesizing verified facts.",
            )
            answer = llm_res.content

        return AdaptiveRetrievalResult(
            query=clean_query,
            status=evaluation.status,
            rounds_executed=rounds_executed,
            max_rounds=max_rounds,
            evaluation=evaluation,
            sources_queried=list(queried_source_names),
            results=all_results,
            answer=answer,
        )
