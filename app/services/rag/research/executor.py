import asyncio
import time
import uuid

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.models import Citation
from app.services.rag.research.models import (
    ParallelResearchConfig,
    ResearchExecutionResult,
    ResearchPlan,
    ResearchSubquestion,
    SubquestionExecutionResult,
    SubquestionExecutionStatus,
)
from app.services.rag.research.planner import MultiStepResearchPlanner
from app.services.rag.research.store import ResearchEvidenceStore
from app.services.rag.router import RetrievalRouter

logger = get_logger("app.services.rag.research.executor")

SUBQUESTION_ANSWER_PROMPT = """You are an Expert Research Analyst.
Given the subquestion and evidence, generate a concise, grounded summary answering the question.

Subquestion: {subquestion}

Retrieved Evidence:
{evidence}

Provide a direct, factual 2-4 paragraph analysis citing specific facts from the evidence.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a Principal Research Director and Industry Strategist.
Synthesize the structured evidence collected across research subquestions into a final report.

Structure your report with:
1. Executive Summary
2. Core Strategy & Operational Comparative Analysis
3. Critical Technology Challenges & Execution Bottlenecks
4. Broader Industry & Market Implications
5. Strategic Outlook & Key Takeaways

Ensure every section is grounded in the provided evidence.
"""


class MultiStepResearchExecutor:
    """Executes multi-step research plans with parallel wave execution and failure isolation."""

    def __init__(
        self,
        planner: MultiStepResearchPlanner | None = None,
        retrieval_router: RetrievalRouter | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._planner = planner or MultiStepResearchPlanner(
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._router = retrieval_router or RetrievalRouter(
            llm_service=self._llm_service,
            settings=self._settings,
        )

    def _compute_execution_waves(
        self,
        subquestions: list[ResearchSubquestion],
    ) -> list[list[ResearchSubquestion]]:
        """Group subquestions into topological waves of mutually independent tasks."""
        completed_ids: set[str] = set()
        remaining = list(subquestions)
        waves: list[list[ResearchSubquestion]] = []

        while remaining:
            ready = [
                sq for sq in remaining
                if all(dep in completed_ids for dep in sq.depends_on)
            ]
            if not ready:
                # Cycle fallback or unresolvable dependencies: advance head
                ready = [remaining[0]]

            waves.append(ready)
            for sq in ready:
                completed_ids.add(sq.id)
                remaining.remove(sq)

        return waves

    async def _execute_single_subquestion(
        self,
        subquestion: ResearchSubquestion,
        store: ResearchEvidenceStore,
        top_k_per_source: int = 3,
    ) -> SubquestionExecutionResult:
        """Execute retrieval and intermediate analysis for a single subquestion."""
        start_time = time.perf_counter()
        logger.info(
            "subquestion_execution_start",
            subquestion_id=subquestion.id,
            index=subquestion.index,
            question=subquestion.question[:80],
        )

        try:
            decision, source_results = await self._router.route_and_retrieve(
                query=subquestion.question,
                top_k_per_source=top_k_per_source,
                min_relevance=0.1,
            )

            used_sources = [s.value for s in decision.selected_sources]
            evidence_snippets: list[str] = [
                f"[{res.source}] {res.content}" for res in source_results
            ]

            citations: list[Citation] = [
                Citation(
                    chunk_id=str(res.metadata.get("chunk_id", f"{res.source}_{idx}")),
                    doc_id=str(res.metadata.get("doc_id", res.source)),
                    source=res.source,
                    file_type=str(res.metadata.get("file_type", "txt")),
                    chunk_index=int(res.metadata.get("index", idx)),
                    content=res.content,
                    similarity=res.relevance,
                )
                for idx, res in enumerate(source_results)
            ]

            # Generate intermediate sub-answer
            evidence_text = (
                "\n\n".join(evidence_snippets)
                if evidence_snippets
                else "No specific documents retrieved from sources."
            )
            prompt = SUBQUESTION_ANSWER_PROMPT.format(
                subquestion=subquestion.question,
                evidence=evidence_text,
            )

            llm_resp = await self._llm_service.generate(
                prompt=prompt,
                system_prompt="You are a precise, grounded factual analyst.",
                temperature=0.2,
            )
            sub_answer = llm_resp.content.strip()

            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            result = SubquestionExecutionResult(
                subquestion_id=subquestion.id,
                index=subquestion.index,
                query=subquestion.question,
                sources=used_sources,
                evidence=evidence_snippets,
                citations=citations,
                sub_answer=sub_answer,
                status=SubquestionExecutionStatus.COMPLETED,
                duration_ms=duration_ms,
                error=None,
            )
            store.add_result(result)
            return result

        except Exception as exc:
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error(
                "subquestion_execution_failed",
                subquestion_id=subquestion.id,
                error=str(exc),
            )
            result = SubquestionExecutionResult(
                subquestion_id=subquestion.id,
                index=subquestion.index,
                query=subquestion.question,
                sources=[],
                evidence=[],
                citations=[],
                sub_answer=f"Execution failed for subquestion: {exc}",
                status=SubquestionExecutionStatus.FAILED,
                duration_ms=duration_ms,
                error=str(exc),
            )
            store.add_result(result)
            return result

    async def _execute_subquestion_with_guardrails(
        self,
        subquestion: ResearchSubquestion,
        store: ResearchEvidenceStore,
        semaphore: asyncio.Semaphore,
        config: ParallelResearchConfig,
        top_k_per_source: int = 3,
    ) -> SubquestionExecutionResult:
        """Execute a subquestion with concurrency limiter, timeout, retry, and failure isolation."""
        async with semaphore:
            last_error: Exception | None = None
            for attempt in range(config.max_retries + 1):
                try:
                    return await asyncio.wait_for(
                        self._execute_single_subquestion(
                            subquestion=subquestion,
                            store=store,
                            top_k_per_source=top_k_per_source,
                        ),
                        timeout=config.subquestion_timeout_seconds,
                    )
                except TimeoutError as exc:
                    last_error = exc
                    logger.warning(
                        "subquestion_execution_timeout",
                        subquestion_id=subquestion.id,
                        attempt=attempt + 1,
                        timeout=config.subquestion_timeout_seconds,
                    )
                except Exception as exc:
                    last_error = exc
                    logger.warning(
                        "subquestion_execution_attempt_failed",
                        subquestion_id=subquestion.id,
                        attempt=attempt + 1,
                        error=str(exc),
                    )

                if attempt < config.max_retries and config.retry_delay_seconds > 0:
                    await asyncio.sleep(config.retry_delay_seconds)

            err_msg = (
                f"Timed out after {config.subquestion_timeout_seconds}s"
                if isinstance(last_error, TimeoutError)
                else str(last_error or "Execution failed")
            )
            failed_res = SubquestionExecutionResult(
                subquestion_id=subquestion.id,
                index=subquestion.index,
                query=subquestion.question,
                sources=[],
                evidence=[],
                citations=[],
                sub_answer=f"Subquestion execution failed after retries: {err_msg}",
                status=SubquestionExecutionStatus.FAILED,
                duration_ms=0.0,
                error=err_msg,
            )
            store.add_result(failed_res)
            return failed_res

    async def execute_research(
        self,
        query: str,
        plan: ResearchPlan | None = None,
        top_k_per_source: int = 3,
        mode: str = "parallel",
        config: ParallelResearchConfig | None = None,
    ) -> ResearchExecutionResult:
        """Execute multi-step research workflow across subquestions and synthesize final report."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty for research execution.")

        active_config = config or ParallelResearchConfig()
        start_time = time.perf_counter()
        research_id = f"research_{uuid.uuid4().hex[:10]}"

        # Step 1: Formulate or adopt Research Plan
        active_plan = plan or await self._planner.create_plan(clean_query)
        logger.info(
            "research_execution_start",
            research_id=research_id,
            mode=mode,
            subquestions_count=len(active_plan.subquestions),
            max_concurrency=active_config.max_concurrency,
        )

        store = ResearchEvidenceStore()
        subquestion_results: list[SubquestionExecutionResult] = []

        if mode == "parallel":
            # Step 2 (Parallel): Execute topological waves of independent subquestions
            semaphore = asyncio.Semaphore(active_config.max_concurrency)
            waves = self._compute_execution_waves(active_plan.subquestions)

            for wave_idx, wave in enumerate(waves):
                logger.info(
                    "executing_research_wave",
                    wave_index=wave_idx,
                    wave_size=len(wave),
                    subquestion_ids=[sq.id for sq in wave],
                )
                tasks = [
                    self._execute_subquestion_with_guardrails(
                        subquestion=sq,
                        store=store,
                        semaphore=semaphore,
                        config=active_config,
                        top_k_per_source=top_k_per_source,
                    )
                    for sq in wave
                ]
                wave_results = await asyncio.gather(*tasks)
                subquestion_results.extend(wave_results)
        else:
            # Step 2 (Sequential): Execute subquestions one by one
            for subq in active_plan.subquestions:
                res = await self._execute_single_subquestion(
                    subquestion=subq,
                    store=store,
                    top_k_per_source=top_k_per_source,
                )
                subquestion_results.append(res)

        # Step 3: Final Synthesis
        synthesis_context = store.format_synthesis_context()
        synthesis_prompt = (
            f"Original Complex Research Inquiry: {clean_query}\n\n"
            f"Overall Goal: {active_plan.overall_goal}\n\n"
            f"Aggregated Subquestion Findings and Evidence:\n\n"
            f"{synthesis_context}\n\n"
            f"Synthesize the comprehensive final report."
        )

        synthesis_resp = await self._llm_service.generate(
            prompt=synthesis_prompt,
            system_prompt=SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.3,
        )
        final_synthesis = synthesis_resp.content.strip()

        all_citations = store.get_all_citations()
        total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        has_failures = any(
            r.status == SubquestionExecutionStatus.FAILED for r in subquestion_results
        )
        overall_status = "partial" if has_failures else "completed"

        logger.info(
            "research_execution_completed",
            research_id=research_id,
            total_duration_ms=total_duration_ms,
            total_citations=len(all_citations),
            status=overall_status,
        )

        return ResearchExecutionResult(
            research_id=research_id,
            original_query=clean_query,
            plan=active_plan,
            subquestion_results=subquestion_results,
            final_synthesis=final_synthesis,
            total_citations=all_citations,
            total_duration_ms=total_duration_ms,
            status=overall_status,
        )
