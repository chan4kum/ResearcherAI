import time
import uuid

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.adaptive import EvidenceEvaluator
from app.services.rag.agentic_retrieval.models import (
    AgenticRetrievalResult,
    AgenticRetrievalTrace,
    RetrievalStepType,
    RetrievalTraceStep,
)
from app.services.rag.agentic_retrieval.planner import RetrievalPlanner
from app.services.rag.agentic_retrieval.trace_store import RetrievalTraceStore
from app.services.rag.evaluator import EvaluationReason, RetrievalEvaluation
from app.services.rag.models import Citation
from app.services.rag.rewriter import QueryRewriter
from app.services.rag.router import RetrievalRouter
from app.services.rag.sources.models import SourceResult

logger = get_logger("app.services.rag.agentic.engine")


class AgenticRetrievalEngine:
    """Autonomous agent-driven retrieval loop with hard guardrails and trace persistence."""

    def __init__(
        self,
        planner: RetrievalPlanner | None = None,
        router: RetrievalRouter | None = None,
        evaluator: EvidenceEvaluator | None = None,
        rewriter: QueryRewriter | None = None,
        llm_service: LLMService | None = None,
        trace_store: RetrievalTraceStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._router = router or RetrievalRouter(
            llm_service=self._llm_service, settings=self._settings
        )
        self._planner = planner or RetrievalPlanner(
            router=self._router,
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._evaluator = evaluator or EvidenceEvaluator()
        self._rewriter = rewriter or QueryRewriter(
            llm_service=self._llm_service, settings=self._settings
        )
        self._trace_store = trace_store or RetrievalTraceStore()

    @property
    def trace_store(self) -> RetrievalTraceStore:
        return self._trace_store

    async def execute(
        self,
        query: str,
        max_iterations: int = 3,
        max_tool_calls: int = 6,
        max_retrieved_documents: int = 20,
        timeout_seconds: float = 10.0,
    ) -> AgenticRetrievalResult:
        """Execute the agentic retrieval loop within strict bounds."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty for agentic retrieval.")

        start_time = time.monotonic()
        session_id = f"trace_{uuid.uuid4().hex[:12]}"
        steps: list[RetrievalTraceStep] = []
        all_results: list[SourceResult] = []
        seen_signatures: set[str] = set()

        total_tool_calls = 0
        iterations = 0
        termination_reason = "UNKNOWN"
        is_sufficient = False
        active_query = clean_query

        # Step 1: Analyze & Plan
        step_t0 = time.monotonic()
        analysis, plan = await self._planner.create_plan(clean_query)
        step_plan_duration = round((time.monotonic() - step_t0) * 1000, 2)

        steps.append(
            RetrievalTraceStep(
                step_index=len(steps) + 1,
                step_type=RetrievalStepType.PLAN,
                query=clean_query,
                sources_contacted=[s.value for s in plan.target_sources],
                decision=f"Needs retrieval: {plan.needs_retrieval}. Rationale: {plan.rationale}",
                duration_ms=step_plan_duration,
            )
        )

        # Early exit if conversational or no retrieval required
        if not plan.needs_retrieval:
            llm_res = await self._llm_service.generate(
                prompt=clean_query,
                system_prompt="You are a helpful conversational assistant.",
            )
            total_duration = round((time.monotonic() - start_time) * 1000, 2)
            trace = AgenticRetrievalTrace(
                session_id=session_id,
                original_query=clean_query,
                total_iterations=0,
                total_tool_calls=0,
                total_documents_retrieved=0,
                duration_ms=total_duration,
                termination_reason="NO_RETRIEVAL_NEEDED",
                steps=steps,
            )
            self._trace_store.save_trace(trace)
            return AgenticRetrievalResult(
                query=clean_query,
                answer=llm_res.content,
                is_sufficient=True,
                citations=[],
                trace=trace,
            )

        # Retrieval Loop
        routing = await self._router.route(clean_query)

        while iterations < max_iterations:
            iterations += 1

            # Guard 1: Timeout
            if (time.monotonic() - start_time) > timeout_seconds:
                termination_reason = "TIMEOUT"
                logger.warning("agentic_retrieval_timeout_reached", query=clean_query[:80])
                break

            # Guard 2: Max Tool Calls
            if total_tool_calls >= max_tool_calls:
                termination_reason = "MAX_TOOL_CALLS_EXCEEDED"
                logger.info("agentic_max_tool_calls_reached", count=total_tool_calls)
                break

            # Guard 3: Max Documents
            if len(all_results) >= max_retrieved_documents:
                termination_reason = "MAX_DOCUMENTS_REACHED"
                logger.info("agentic_max_documents_reached", count=len(all_results))
                break

            # Step: RETRIEVE
            step_ret_t0 = time.monotonic()
            decision, new_results = await self._router.route_and_retrieve(
                query=active_query,
                top_k_per_source=3,
                min_relevance=0.0,
            )
            total_tool_calls += len(decision.selected_sources)
            step_ret_duration = round((time.monotonic() - step_ret_t0) * 1000, 2)

            # Accumulate results with deduplication up to max_retrieved_documents
            added_this_step = 0
            for r in new_results:
                sig = f"{r.source}:{r.content[:50]}"
                if sig not in seen_signatures and len(all_results) < max_retrieved_documents:
                    all_results.append(r)
                    seen_signatures.add(sig)
                    added_this_step += 1

            steps.append(
                RetrievalTraceStep(
                    step_index=len(steps) + 1,
                    step_type=RetrievalStepType.RETRIEVE,
                    query=active_query,
                    sources_contacted=[s.value for s in decision.selected_sources],
                    documents_retrieved_count=added_this_step,
                    decision=(
                        f"Retrieved {added_this_step} documents for query '{active_query[:50]}'."
                    ),
                    duration_ms=step_ret_duration,
                )
            )

            # Step: EVALUATE
            step_eval_t0 = time.monotonic()
            evaluation = self._evaluator.evaluate(
                query=clean_query,
                analysis=analysis,
                routing=routing,
                results=all_results,
            )
            step_eval_duration = round((time.monotonic() - step_eval_t0) * 1000, 2)

            steps.append(
                RetrievalTraceStep(
                    step_index=len(steps) + 1,
                    step_type=RetrievalStepType.EVALUATE,
                    query=active_query,
                    evaluation_summary={
                        "is_sufficient": evaluation.is_sufficient,
                        "confidence": evaluation.confidence,
                        "relevance": evaluation.relevance_score,
                        "coverage": evaluation.coverage_score,
                        "missing_entities": evaluation.missing_entities,
                    },
                    decision=evaluation.reason,
                    duration_ms=step_eval_duration,
                )
            )

            if evaluation.is_sufficient:
                is_sufficient = True
                termination_reason = "EVIDENCE_SUFFICIENT"
                logger.info(
                    "agentic_retrieval_satisfied",
                    iteration=iterations,
                    confidence=evaluation.confidence,
                )
                break

            # If insufficient and more iterations allowed, REWRITE
            if iterations < max_iterations:
                step_rw_t0 = time.monotonic()
                reasons: list[EvaluationReason] = []
                if evaluation.relevance_score < 0.5:
                    reasons.append(EvaluationReason.LOW_RELEVANCE)
                if evaluation.missing_entities:
                    reasons.append(EvaluationReason.MISSING_ENTITIES)
                if evaluation.coverage_score < 0.5:
                    reasons.append(EvaluationReason.POOR_COVERAGE)
                if not reasons:
                    reasons.append(EvaluationReason.INSUFFICIENT_EVIDENCE)

                ret_eval = RetrievalEvaluation(
                    is_sufficient=False,
                    reasons=reasons,
                    relevance_score=evaluation.relevance_score,
                    entity_coverage=evaluation.coverage_score,
                    missing_entities=evaluation.missing_entities,
                    feedback_prompt=evaluation.reason,
                )

                attempted_queries = [
                    s.query for s in steps if s.step_type == RetrievalStepType.RETRIEVE
                ]
                active_query = await self._rewriter.rewrite(
                    original_query=clean_query,
                    analysis=analysis,
                    evaluation=ret_eval,
                    attempt_index=iterations + 1,
                    previous_queries=attempted_queries,
                )
                step_rw_duration = round((time.monotonic() - step_rw_t0) * 1000, 2)

                steps.append(
                    RetrievalTraceStep(
                        step_index=len(steps) + 1,
                        step_type=RetrievalStepType.REWRITE,
                        query=active_query,
                        decision=f"Rewrote query to '{active_query}' for next retrieval round.",
                        duration_ms=step_rw_duration,
                    )
                )

        if termination_reason == "UNKNOWN":
            termination_reason = "MAX_ITERATIONS_REACHED"

        # Step: SYNTHESIZE Answer
        step_syn_t0 = time.monotonic()
        citations: list[Citation] = [r.citation for r in all_results]

        if all_results:
            context_blocks = "\n\n".join(
                f"[{idx}] Source: {r.source} ({r.source_type.value})\n{r.content}"
                for idx, r in enumerate(all_results, start=1)
            )
            prompt = (
                f"Question: {clean_query}\n\n"
                f"Retrieved Evidence:\n{context_blocks}\n\n"
                f"Synthesize an accurate, grounded answer citing evidence numbers."
            )
            llm_res = await self._llm_service.generate(
                prompt=prompt,
                system_prompt="You are a research agent synthesizing ground-truth facts.",
            )
            answer_text = llm_res.content
        else:
            answer_text = "No relevant knowledge or records could be located for this query."

        step_syn_duration = round((time.monotonic() - step_syn_t0) * 1000, 2)
        steps.append(
            RetrievalTraceStep(
                step_index=len(steps) + 1,
                step_type=RetrievalStepType.SYNTHESIZE,
                query=clean_query,
                documents_retrieved_count=len(all_results),
                decision=(
                    f"Synthesized answer with {len(citations)} citations. "
                    f"Termination: {termination_reason}."
                ),
                duration_ms=step_syn_duration,
            )
        )

        total_duration = round((time.monotonic() - start_time) * 1000, 2)
        trace = AgenticRetrievalTrace(
            session_id=session_id,
            original_query=clean_query,
            total_iterations=iterations,
            total_tool_calls=total_tool_calls,
            total_documents_retrieved=len(all_results),
            duration_ms=total_duration,
            termination_reason=termination_reason,
            steps=steps,
        )

        # Persist telemetry trace
        self._trace_store.save_trace(trace)

        return AgenticRetrievalResult(
            query=clean_query,
            answer=answer_text,
            is_sufficient=is_sufficient,
            citations=citations,
            trace=trace,
        )
