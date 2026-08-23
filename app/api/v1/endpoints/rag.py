from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.logging import get_logger
from app.models.schemas import (
    AdaptiveRetrievalRequest,
    AdaptiveRetrievalResponse,
    AgenticResearchLoopRequest,
    AgenticResearchLoopResponse,
    AgenticRetrievalRequest,
    AgenticRetrievalResponse,
    AnalyzeQueryRequest,
    AnalyzeQueryResponse,
    CitationItem,
    CreateResearchPlanRequest,
    CreateResearchPlanResponse,
    CriticEvaluationItem,
    CriticIssueItem,
    CritiqueAnswerRequest,
    CritiqueAnswerResponse,
    EvidenceEvaluationItem,
    ExecuteResearchRequest,
    ExecuteResearchResponse,
    ExtractedEntityItem,
    FactualClaimItem,
    QueryRewriteAttemptItem,
    RAGQueryRequest,
    RAGQueryResponse,
    RerankMeasurementItem,
    ResearchSubquestionItem,
    RetrievalTraceStepItem,
    RouteQueryRequest,
    RouteQueryResponse,
    SelfCorrectAnswerRequest,
    SelfCorrectAnswerResponse,
    SelfCorrectionAttemptItem,
    SubquestionResultItem,
    VerifyAnswerRequest,
    VerifyAnswerResponse,
)
from app.services.rag.adaptive import AdaptiveRetriever
from app.services.rag.agentic_retrieval.engine import AgenticRetrievalEngine
from app.services.rag.agentic_retrieval.trace_store import RetrievalTraceStore
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.critic.agent import CriticAgent
from app.services.rag.critic.engine import SelfCorrectionEngine
from app.services.rag.loop.models import AgenticResearchLoopConfig
from app.services.rag.loop.orchestrator import AgenticResearchOrchestrator
from app.services.rag.research.executor import MultiStepResearchExecutor
from app.services.rag.research.planner import MultiStepResearchPlanner
from app.services.rag.retriever import create_retriever
from app.services.rag.router import RetrievalRouter
from app.services.rag.service import RAGService
from app.services.rag.verification.verifier import AnswerVerifier

logger = get_logger("app.api.v1.rag")
router = APIRouter()


def get_rag_service(request: Request) -> RAGService:
    """Dependency resolver for RAGService attached to application state."""
    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        embedding_service = getattr(request.app.state, "embedding_service", None)
        vector_repo = getattr(request.app.state, "vector_repository", None)
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)

        if not embedding_service or not vector_repo:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Embedding service or Vector repository not initialized.",
            )

        retriever = create_retriever(
            embedding_service=embedding_service,
            vector_repository=vector_repo,
            mode=getattr(settings, "default_retrieval_mode", "hybrid") if settings else "hybrid",
            settings=settings,
        )
        service = RAGService(
            retriever=retriever,
            llm_service=llm_service,
            settings=settings,
        )
        request.app.state.rag_service = service
    return service


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question and receive a grounded answer with citation metadata",
)
async def query_rag_endpoint(
    request: RAGQueryRequest,
    raw_request: Request,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGQueryResponse:
    """Execute end-to-end RAG flow:
    Question -> Hybrid/Semantic/Keyword Search -> LLM -> Grounded Answer.
    """
    try:
        retriever_override = None
        if request.mode or request.strategy or request.hyde or request.alpha is not None:
            settings = getattr(raw_request.app.state, "settings", None)
            embedding_service = getattr(raw_request.app.state, "embedding_service", None)
            vector_repo = getattr(raw_request.app.state, "vector_repository", None)
            llm_service = getattr(raw_request.app.state, "llm_service", None)
            if embedding_service and vector_repo:
                strategy_val = "hyde" if request.hyde else (request.strategy or "normal")
                retriever_override = create_retriever(
                    embedding_service=embedding_service,
                    vector_repository=vector_repo,
                    mode=request.mode or "hybrid",
                    strategy=strategy_val,
                    settings=settings,
                    alpha=request.alpha if request.alpha is not None else 0.5,
                    llm_service=llm_service,
                )

        response = await rag_service.answer(
            question=request.question,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            system_prompt=request.system_prompt,
            filters=request.filters,
            retriever=retriever_override,
            rerank=request.rerank,
            top_n=request.top_n,
            enable_rewriting=request.enable_rewriting,
            max_attempts=request.max_attempts,
            strategy=request.strategy,
            hyde=request.hyde,
        )

        citation_items = [
            CitationItem(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                source=c.source,
                file_type=c.file_type,
                chunk_index=c.chunk_index,
                content=c.content,
                similarity=c.similarity,
                document_type=c.document_type,
                department=c.department,
                date=c.date,
                author=c.author,
                tags=c.tags,
                initial_rank=c.initial_rank,
                rerank_score=c.rerank_score,
                metadata=c.metadata,
            )
            for c in response.citations
        ]

        active_mode = str(
            request.mode
            or getattr(
                getattr(raw_request.app.state, "settings", None),
                "default_retrieval_mode",
                "hybrid",
            )
            or "hybrid"
        )

        rerank_telemetry = None
        is_reranked = False
        if any(c.rerank_score is not None for c in response.citations):
            is_reranked = True
            rerank_telemetry = [
                RerankMeasurementItem(
                    chunk_id=c.chunk_id,
                    source=c.source,
                    initial_rank=c.initial_rank or idx,
                    reranked_rank=idx,
                    initial_score=c.similarity,
                    rerank_score=c.rerank_score or c.similarity,
                    rank_delta=(c.initial_rank or idx) - idx,
                )
                for idx, c in enumerate(response.citations, start=1)
            ]

        raw_rewrites = response.metadata.get("query_rewriting", [])
        rewrite_telemetry = (
            [QueryRewriteAttemptItem(**item) for item in raw_rewrites]
            if raw_rewrites
            else None
        )

        return RAGQueryResponse(
            question=response.question,
            final_query=response.final_query,
            strategy=response.strategy,
            hypothetical_document=response.hypothetical_document,
            answer=response.answer,
            citations=citation_items,
            retrieved_chunks_count=response.retrieved_chunks_count,
            retrieval_mode=active_mode,
            reranked=is_reranked,
            rerank_metrics=rerank_telemetry,
            rewrite_history=rewrite_telemetry,
            model=response.model,
            provider=response.provider,
            metadata={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_query_analyzer(request: Request) -> QueryAnalyzer:
    """Dependency resolver for QueryAnalyzer attached to application state."""
    analyzer: QueryAnalyzer | None = getattr(request.app.state, "query_analyzer", None)
    if analyzer is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        analyzer = QueryAnalyzer(llm_service=llm_service, settings=settings)
        request.app.state.query_analyzer = analyzer
    return analyzer


@router.post(
    "/analyze-query",
    response_model=AnalyzeQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Decompose and analyze a query into structured intent, entities, and subquestions",
)
async def analyze_query_endpoint(
    request: AnalyzeQueryRequest,
    analyzer: QueryAnalyzer = Depends(get_query_analyzer),
) -> AnalyzeQueryResponse:
    """Perform pre-retrieval query understanding and semantic decomposition."""
    try:
        analysis = await analyzer.analyze(request.query)
        return AnalyzeQueryResponse(
            original_query=analysis.original_query,
            intent=analysis.intent.value,
            entities=[
                ExtractedEntityItem(
                    text=e.text,
                    label=e.label,
                    category=e.category,
                )
                for e in analysis.entities
            ],
            subquestions=analysis.subquestions,
            required_information_types=analysis.required_information_types,
            potential_source_types=analysis.potential_source_types,
            is_ambiguous=analysis.is_ambiguous,
            clarification_needed=analysis.clarification_needed,
            temporal_scope=analysis.temporal_scope,
            confidence_score=analysis.confidence_score,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_retrieval_router(request: Request) -> RetrievalRouter:
    """Dependency resolver for RetrievalRouter."""
    router_instance: RetrievalRouter | None = getattr(
        request.app.state, "retrieval_router", None
    )
    if router_instance is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        query_analyzer = QueryAnalyzer(llm_service=llm_service, settings=settings)
        router_instance = RetrievalRouter(
            query_analyzer=query_analyzer,
            llm_service=llm_service,
            settings=settings,
        )
    return router_instance


@router.post(
    "/route-query",
    response_model=RouteQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Route a query to optimal heterogeneous knowledge source destinations",
)
async def route_query_endpoint(
    request: RouteQueryRequest,
    router_instance: RetrievalRouter = Depends(get_retrieval_router),
) -> RouteQueryResponse:
    """Analyze query and determine target knowledge source destinations."""
    try:
        decision = await router_instance.route(request.query)
        return RouteQueryResponse(
            query=decision.query,
            intent=decision.intent.value,
            selected_sources=[s.value for s in decision.selected_sources],
            reason=decision.reason,
            confidence=decision.confidence,
            entities_detected=decision.entities_detected,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_adaptive_retriever(request: Request) -> AdaptiveRetriever:
    """Dependency resolver for AdaptiveRetriever."""
    adaptive_instance: AdaptiveRetriever | None = getattr(
        request.app.state, "adaptive_retriever", None
    )
    if adaptive_instance is None:
        router_instance = get_retrieval_router(request)
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        adaptive_instance = AdaptiveRetriever(
            router=router_instance,
            llm_service=llm_service,
            settings=settings,
        )
    return adaptive_instance


@router.post(
    "/adaptive-retrieve",
    response_model=AdaptiveRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Adaptively route, retrieve, evaluate evidence, and conditionally generate answers",
)
async def adaptive_retrieve_endpoint(
    request: AdaptiveRetrievalRequest,
    retriever: AdaptiveRetriever = Depends(get_adaptive_retriever),
) -> AdaptiveRetrievalResponse:
    """Run iterative adaptive retrieval loop with evidence sufficiency evaluation."""
    try:
        result = await retriever.retrieve_adaptively(
            query=request.query,
            max_rounds=request.max_rounds,
            generate_answer=request.generate_answer,
        )
        return AdaptiveRetrievalResponse(
            query=result.query,
            status=result.status.value,
            rounds_executed=result.rounds_executed,
            max_rounds=result.max_rounds,
            evaluation=EvidenceEvaluationItem(
                status=result.evaluation.status.value,
                is_sufficient=result.evaluation.is_sufficient,
                relevance_score=result.evaluation.relevance_score,
                coverage_score=result.evaluation.coverage_score,
                source_diversity_score=result.evaluation.source_diversity_score,
                evidence_quantity=result.evaluation.evidence_quantity,
                confidence=result.evaluation.confidence,
                reason=result.evaluation.reason,
                missing_entities=result.evaluation.missing_entities,
                missing_sources=result.evaluation.missing_sources,
            ),
            sources_queried=result.sources_queried,
            evidence_count=len(result.results),
            answer=result.answer,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_agentic_retrieval_engine(request: Request) -> AgenticRetrievalEngine:
    """Dependency resolver for AgenticRetrievalEngine."""
    engine_instance: AgenticRetrievalEngine | None = getattr(
        request.app.state, "agentic_retrieval_engine", None
    )
    if engine_instance is None:
        router_instance = get_retrieval_router(request)
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        trace_store = (
            getattr(request.app.state, "retrieval_trace_store", None) or RetrievalTraceStore()
        )
        engine_instance = AgenticRetrievalEngine(
            router=router_instance,
            llm_service=llm_service,
            trace_store=trace_store,
            settings=settings,
        )
    return engine_instance


@router.post(
    "/agentic-retrieve",
    response_model=AgenticRetrievalResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute autonomous agent-driven retrieval loop with hard guardrails and trace",
)
async def agentic_retrieve_endpoint(
    request: AgenticRetrievalRequest,
    engine: AgenticRetrievalEngine = Depends(get_agentic_retrieval_engine),
) -> AgenticRetrievalResponse:
    """Run autonomous retrieval loop, evaluate evidence, synthesize answer, and persist trace."""
    try:
        result = await engine.execute(
            query=request.query,
            max_iterations=request.max_iterations,
            max_tool_calls=request.max_tool_calls,
            max_retrieved_documents=request.max_retrieved_documents,
            timeout_seconds=request.timeout_seconds,
        )
        return AgenticRetrievalResponse(
            session_id=result.trace.session_id,
            query=result.query,
            answer=result.answer,
            is_sufficient=result.is_sufficient,
            total_iterations=result.trace.total_iterations,
            total_tool_calls=result.trace.total_tool_calls,
            total_documents_retrieved=result.trace.total_documents_retrieved,
            termination_reason=result.trace.termination_reason,
            duration_ms=result.trace.duration_ms,
            citations=[
                CitationItem(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    file_type=c.file_type,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    similarity=c.similarity,
                    metadata=c.metadata,
                    document_type=c.document_type,
                    department=c.department,
                    author=c.author,
                    date=c.date,
                    tags=c.tags,
                )
                for c in result.citations
            ],
            steps=[
                RetrievalTraceStepItem(
                    step_index=s.step_index,
                    step_type=s.step_type.value,
                    query=s.query,
                    sources_contacted=s.sources_contacted,
                    documents_retrieved_count=s.documents_retrieved_count,
                    decision=s.decision,
                    duration_ms=s.duration_ms,
                )
                for s in result.trace.steps
            ],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.get(
    "/traces/{session_id}",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Retrieve persistent telemetry trace by session ID",
)
async def get_retrieval_trace_endpoint(
    session_id: str,
    engine: AgenticRetrievalEngine = Depends(get_agentic_retrieval_engine),
) -> dict[str, Any]:
    """Fetch stored retrieval trace."""
    trace = engine.trace_store.get_trace(session_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with session_id '{session_id}' not found.",
        )
    return trace.model_dump()


def get_research_planner(request: Request) -> MultiStepResearchPlanner:
    """Dependency resolver for MultiStepResearchPlanner."""
    planner_instance: MultiStepResearchPlanner | None = getattr(
        request.app.state, "research_planner", None
    )
    if planner_instance is None:
        analyzer_instance = get_query_analyzer(request)
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        planner_instance = MultiStepResearchPlanner(
            query_analyzer=analyzer_instance,
            llm_service=llm_service,
            settings=settings,
        )
    return planner_instance


@router.post(
    "/research/plan",
    response_model=CreateResearchPlanResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a structured multi-step research plan for a complex inquiry",
)
async def create_research_plan_endpoint(
    request: CreateResearchPlanRequest,
    planner: MultiStepResearchPlanner = Depends(get_research_planner),
) -> CreateResearchPlanResponse:
    """Decompose complex inquiry into structured subquestions with dependency pointers."""
    try:
        plan = await planner.create_plan(request.query)
        return CreateResearchPlanResponse(
            plan_id=plan.plan_id,
            original_query=plan.original_query,
            overall_goal=plan.overall_goal,
            estimated_complexity=plan.estimated_complexity,
            suggested_synthesis_strategy=plan.suggested_synthesis_strategy,
            created_at=plan.created_at,
            subquestions=[
                ResearchSubquestionItem(
                    id=sq.id,
                    index=sq.index,
                    question=sq.question,
                    subquestion_type=sq.subquestion_type.value,
                    target_entities=sq.target_entities,
                    expected_output_type=sq.expected_output_type,
                    suggested_sources=[s.value for s in sq.suggested_sources],
                    depends_on=sq.depends_on,
                )
                for sq in plan.subquestions
            ],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_research_executor(request: Request) -> MultiStepResearchExecutor:
    """Dependency resolver for MultiStepResearchExecutor."""
    executor_instance: MultiStepResearchExecutor | None = getattr(
        request.app.state, "research_executor", None
    )
    if executor_instance is None:
        planner = get_research_planner(request)
        router = get_retrieval_router(request)
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        executor_instance = MultiStepResearchExecutor(
            planner=planner,
            retrieval_router=router,
            llm_service=llm_service,
            settings=settings,
        )
    return executor_instance


@router.post(
    "/research/execute",
    response_model=ExecuteResearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute multi-step research inquiry with parallel waves and synthesize final report",
)
async def execute_research_endpoint(
    request: ExecuteResearchRequest,
    executor: MultiStepResearchExecutor = Depends(get_research_executor),
) -> ExecuteResearchResponse:
    """Execute research plan across subquestions with failure isolation and synthesize report."""
    try:
        from app.services.rag.research.models import ParallelResearchConfig

        cfg = ParallelResearchConfig(
            max_concurrency=request.max_concurrency,
            subquestion_timeout_seconds=request.timeout_seconds,
            max_retries=request.max_retries,
        )
        result = await executor.execute_research(
            query=request.query,
            top_k_per_source=request.top_k_per_source,
            mode=request.mode,
            config=cfg,
        )
        return ExecuteResearchResponse(
            research_id=result.research_id,
            original_query=result.original_query,
            final_synthesis=result.final_synthesis,
            status=result.status,
            total_duration_ms=result.total_duration_ms,
            total_citations=[
                CitationItem(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    file_type=c.file_type,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    similarity=c.similarity,
                )
                for c in result.total_citations
            ],
            subquestion_results=[
                SubquestionResultItem(
                    subquestion_id=sq.subquestion_id,
                    index=sq.index,
                    query=sq.query,
                    sources=sq.sources,
                    evidence=sq.evidence,
                    citations=[
                        CitationItem(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            source=c.source,
                            file_type=c.file_type,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            similarity=c.similarity,
                        )
                        for c in sq.citations
                    ],
                    sub_answer=sq.sub_answer,
                    status=sq.status.value,
                    duration_ms=sq.duration_ms,
                    error=sq.error,
                )
                for sq in result.subquestion_results
            ],
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_critic_agent(request: Request) -> CriticAgent:
    """Dependency resolver for CriticAgent attached to application state."""
    agent: CriticAgent | None = getattr(request.app.state, "critic_agent", None)
    if agent is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        agent = CriticAgent(llm_service=llm_service, settings=settings)
        request.app.state.critic_agent = agent
    return agent


def get_self_correction_engine(request: Request) -> SelfCorrectionEngine:
    """Dependency resolver for SelfCorrectionEngine attached to application state."""
    engine: SelfCorrectionEngine | None = getattr(request.app.state, "self_correction_engine", None)
    if engine is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        critic = get_critic_agent(request)
        engine = SelfCorrectionEngine(
            critic_agent=critic,
            llm_service=llm_service,
            settings=settings,
        )
        request.app.state.self_correction_engine = engine
    return engine


@router.post(
    "/critic/evaluate",
    response_model=CritiqueAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate draft answer against evidence documents across 6 defect dimensions",
)
async def critique_answer_endpoint(
    request: CritiqueAnswerRequest,
    critic: CriticAgent = Depends(get_critic_agent),
) -> CritiqueAnswerResponse:
    """Evaluate draft answer for hallucinations, missing evidence, contradictions, and flaws."""
    try:
        evaluation = await critic.evaluate(
            question=request.question,
            evidence=request.evidence,
            draft_answer=request.draft_answer,
        )
        return CritiqueAnswerResponse(
            is_acceptable=evaluation.is_acceptable,
            critique_score=evaluation.critique_score,
            issues=[
                CriticIssueItem(
                    issue_type=i.issue_type.value,
                    severity=i.severity.value,
                    claim_or_passage=i.claim_or_passage,
                    reason=i.reason,
                    suggested_fix=i.suggested_fix,
                )
                for i in evaluation.issues
            ],
            feedback_summary=evaluation.feedback_summary,
            action_recommended=evaluation.action_recommended,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


@router.post(
    "/critic/correct",
    response_model=SelfCorrectAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Run bounded self-correction loop on draft answer until accepted or max iterations",
)
async def self_correct_answer_endpoint(
    request: SelfCorrectAnswerRequest,
    engine: SelfCorrectionEngine = Depends(get_self_correction_engine),
) -> SelfCorrectAnswerResponse:
    """Iteratively critique and revise draft answer within strict loop guardrails."""
    try:
        result = await engine.correct_answer(
            question=request.question,
            evidence=request.evidence,
            draft_answer=request.draft_answer,
            max_corrections=request.max_corrections,
        )
        return SelfCorrectAnswerResponse(
            question=result.question,
            original_draft=result.original_draft,
            final_answer=result.final_answer,
            iterations=result.iterations,
            max_iterations=result.max_iterations,
            is_corrected=result.is_corrected,
            final_evaluation=CriticEvaluationItem(
                is_acceptable=result.final_evaluation.is_acceptable,
                critique_score=result.final_evaluation.critique_score,
                issues=[
                    CriticIssueItem(
                        issue_type=i.issue_type.value,
                        severity=i.severity.value,
                        claim_or_passage=i.claim_or_passage,
                        reason=i.reason,
                        suggested_fix=i.suggested_fix,
                    )
                    for i in result.final_evaluation.issues
                ],
                feedback_summary=result.final_evaluation.feedback_summary,
                action_recommended=result.final_evaluation.action_recommended,
            ),
            attempts=[
                SelfCorrectionAttemptItem(
                    iteration=att.iteration,
                    draft_answer=att.draft_answer,
                    evaluation=CriticEvaluationItem(
                        is_acceptable=att.evaluation.is_acceptable,
                        critique_score=att.evaluation.critique_score,
                        issues=[
                            CriticIssueItem(
                                issue_type=i.issue_type.value,
                                severity=i.severity.value,
                                claim_or_passage=i.claim_or_passage,
                                reason=i.reason,
                                suggested_fix=i.suggested_fix,
                            )
                            for i in att.evaluation.issues
                        ],
                        feedback_summary=att.evaluation.feedback_summary,
                        action_recommended=att.evaluation.action_recommended,
                    ),
                    revised_answer=att.revised_answer,
                    duration_ms=att.duration_ms,
                )
                for att in result.attempts
            ],
            total_duration_ms=result.total_duration_ms,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_answer_verifier(request: Request) -> AnswerVerifier:
    """Dependency resolver for AnswerVerifier attached to application state."""
    verifier: AnswerVerifier | None = getattr(request.app.state, "answer_verifier", None)
    if verifier is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        verifier = AnswerVerifier(llm_service=llm_service, settings=settings)
        request.app.state.answer_verifier = verifier
    return verifier


@router.post(
    "/verify",
    response_model=VerifyAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Decompose answer into atomic factual claims and verify against evidence documents",
)
async def verify_answer_endpoint(
    request: VerifyAnswerRequest,
    verifier: AnswerVerifier = Depends(get_answer_verifier),
) -> VerifyAnswerResponse:
    """Verify each claim in an answer for groundedness and return sanitized response."""
    try:
        report = await verifier.verify_answer(
            question=request.question,
            answer=request.answer,
            evidence=request.evidence,
        )
        return VerifyAnswerResponse(
            report_id=report.report_id,
            question=report.question,
            original_answer=report.original_answer,
            verified_answer=report.verified_answer,
            total_claims=report.total_claims,
            claims=[
                FactualClaimItem(
                    claim_id=c.claim_id,
                    claim_text=c.claim_text,
                    evidence_text=c.evidence_text,
                    source=c.source,
                    support_status=c.support_status.value,
                    confidence=c.confidence,
                    citation_chunk_id=c.citation_chunk_id,
                    reason=c.reason,
                )
                for c in report.claims
            ],
            supported_count=report.supported_count,
            partially_supported_count=report.partially_supported_count,
            unsupported_count=report.unsupported_count,
            contradicted_count=report.contradicted_count,
            verified_ratio=report.verified_ratio,
            is_verified=report.is_verified,
            duration_ms=report.duration_ms,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


def get_agentic_research_orchestrator(request: Request) -> AgenticResearchOrchestrator:
    """Dependency resolver for AgenticResearchOrchestrator attached to application state."""
    orchestrator: AgenticResearchOrchestrator | None = getattr(
        request.app.state, "agentic_research_orchestrator", None
    )
    if orchestrator is None:
        llm_service = getattr(request.app.state, "llm_service", None)
        settings = getattr(request.app.state, "settings", None)
        query_analyzer = getattr(request.app.state, "query_analyzer", None)
        research_planner = getattr(request.app.state, "research_planner", None)
        research_executor = getattr(request.app.state, "research_executor", None)
        retrieval_router = getattr(request.app.state, "retrieval_router", None)
        self_correction_engine = getattr(request.app.state, "self_correction_engine", None)
        answer_verifier = getattr(request.app.state, "answer_verifier", None)

        orchestrator = AgenticResearchOrchestrator(
            query_analyzer=query_analyzer,
            research_planner=research_planner,
            research_executor=research_executor,
            retrieval_router=retrieval_router,
            self_correction_engine=self_correction_engine,
            answer_verifier=answer_verifier,
            llm_service=llm_service,
            settings=settings,
        )
        request.app.state.agentic_research_orchestrator = orchestrator
    return orchestrator


@router.post(
    "/research/loop",
    response_model=AgenticResearchLoopResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute end-to-end agentic research loop with self-correction and verification",
)
async def execute_agentic_research_loop_endpoint(
    request: AgenticResearchLoopRequest,
    orchestrator: AgenticResearchOrchestrator = Depends(get_agentic_research_orchestrator),
) -> AgenticResearchLoopResponse:
    """Run full agentic loop: analysis, planning, parallel execution, critic, and verification."""
    try:
        config = AgenticResearchLoopConfig(
            max_research_iterations=request.max_research_iterations,
            max_concurrency=request.max_concurrency,
            enable_self_correction=request.enable_self_correction,
            enable_verification=request.enable_verification,
            timeout_seconds=request.timeout_seconds,
        )
        result = await orchestrator.run(
            question=request.question,
            config=config,
        )
        return AgenticResearchLoopResponse(
            loop_id=result.loop_id,
            question=result.question,
            intent=result.query_analysis.intent.value,
            is_complex=result.query_analysis.is_complex,
            subquestions=[
                ResearchSubquestionItem(
                    id=sub.id,
                    index=sub.index,
                    question=sub.question,
                    subquestion_type=sub.subquestion_type.value,
                    target_entities=sub.target_entities,
                    expected_output_type=sub.expected_output_type,
                    suggested_sources=[s.value for s in sub.suggested_sources],
                    depends_on=sub.depends_on,
                )
                for sub in (result.research_plan.subquestions if result.research_plan else [])
            ],
            subquestion_results=[
                SubquestionResultItem(
                    subquestion_id=res.subquestion_id,
                    index=res.index,
                    query=res.query,
                    sources=res.sources,
                    evidence=res.evidence,
                    citations=[
                        CitationItem(
                            chunk_id=c.chunk_id,
                            doc_id=c.doc_id,
                            source=c.source,
                            file_type=c.file_type,
                            chunk_index=c.chunk_index,
                            content=c.content,
                            similarity=c.similarity,
                        )
                        for c in res.citations
                    ],
                    sub_answer=res.sub_answer,
                    status=res.status.value,
                    duration_ms=res.duration_ms,
                    error=res.error,
                )
                for res in result.subquestion_results
            ],
            draft_answer=result.draft_answer,
            final_answer=result.final_answer,
            citations=[
                CitationItem(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    source=c.source,
                    file_type=c.file_type,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    similarity=c.similarity,
                )
                for c in result.citations
            ],
            is_verified=(
                result.verification_report.is_verified
                if result.verification_report
                else True
            ),
            verified_ratio=(
                result.verification_report.verified_ratio
                if result.verification_report
                else 1.0
            ),
            total_duration_ms=result.total_duration_ms,
            status=result.status,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc





