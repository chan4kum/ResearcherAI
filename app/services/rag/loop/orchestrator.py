import time
import uuid

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.critic.engine import SelfCorrectionEngine
from app.services.rag.loop.models import (
    AgenticResearchLoopConfig,
    AgenticResearchLoopResult,
)
from app.services.rag.research.executor import MultiStepResearchExecutor
from app.services.rag.research.models import ParallelResearchConfig
from app.services.rag.research.planner import MultiStepResearchPlanner
from app.services.rag.research.store import ResearchEvidenceStore
from app.services.rag.router import RetrievalRouter
from app.services.rag.verification.verifier import AnswerVerifier

logger = get_logger("app.services.rag.loop.orchestrator")


class AgenticResearchOrchestrator:
    """End-to-end orchestrator executing the full agentic research loop."""

    def __init__(
        self,
        query_analyzer: QueryAnalyzer | None = None,
        research_planner: MultiStepResearchPlanner | None = None,
        research_executor: MultiStepResearchExecutor | None = None,
        retrieval_router: RetrievalRouter | None = None,
        self_correction_engine: SelfCorrectionEngine | None = None,
        answer_verifier: AnswerVerifier | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._query_analyzer = query_analyzer or QueryAnalyzer(llm_service=self._llm_service)
        self._research_planner = research_planner or MultiStepResearchPlanner(
            query_analyzer=self._query_analyzer,
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._retrieval_router = retrieval_router or RetrievalRouter(
            query_analyzer=self._query_analyzer,
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._research_executor = research_executor or MultiStepResearchExecutor(
            planner=self._research_planner,
            retrieval_router=self._retrieval_router,
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._self_correction_engine = self_correction_engine or SelfCorrectionEngine(
            llm_service=self._llm_service,
            settings=self._settings,
        )
        self._answer_verifier = answer_verifier or AnswerVerifier(
            llm_service=self._llm_service,
            settings=self._settings,
        )

    async def run(
        self,
        question: str,
        config: AgenticResearchLoopConfig | None = None,
    ) -> AgenticResearchLoopResult:
        """Execute complete agentic research loop from analysis to verified answer."""
        start_time = time.perf_counter()
        loop_id = f"loop_{uuid.uuid4().hex[:10]}"
        cfg = config or AgenticResearchLoopConfig()
        clean_question = question.strip()

        if not clean_question:
            raise ValueError("Research question cannot be empty.")

        logger.info("agentic_research_loop_started", loop_id=loop_id, question=clean_question)

        # 1. Query Analysis
        analysis = await self._query_analyzer.analyze(clean_question)
        logger.info(
            "loop_query_analyzed",
            loop_id=loop_id,
            intent=analysis.intent.value,
            is_complex=analysis.is_complex,
        )

        # 2. Research Planning
        plan = await self._research_planner.create_plan(query=clean_question)
        logger.info(
            "loop_plan_created",
            loop_id=loop_id,
            plan_id=plan.plan_id,
            subquestion_count=len(plan.subquestions),
        )

        # 3. Parallel Multi-Step Research Execution
        sub_timeout = min(30.0, max(1.0, cfg.timeout_seconds / max(1, len(plan.subquestions))))
        parallel_cfg = ParallelResearchConfig(
            max_concurrency=cfg.max_concurrency,
            subquestion_timeout_seconds=sub_timeout,
        )
        exec_result = await self._research_executor.execute_research(
            query=clean_question,
            plan=plan,
            config=parallel_cfg,
            mode="parallel",
        )

        subquestion_results = exec_result.subquestion_results
        evidence_store = ResearchEvidenceStore()
        for sub_res in subquestion_results:
            evidence_store.add_result(sub_res)
        draft_answer = exec_result.final_synthesis
        all_citations = exec_result.total_citations

        # 4. Critic & Self-Correction
        self_correction_result = None
        if cfg.enable_self_correction:
            evidence_text = "\n\n".join(evidence_store.get_all_evidence())
            self_correction_result = await self._self_correction_engine.correct_answer(
                question=clean_question,
                draft_answer=draft_answer,
                evidence=evidence_text,
                citations=all_citations,
                max_corrections=cfg.max_research_iterations,
            )
            draft_answer = self_correction_result.final_answer
            logger.info(
                "loop_self_correction_completed",
                loop_id=loop_id,
                iterations=self_correction_result.iterations,
                is_corrected=self_correction_result.is_corrected,
            )

        # 5. Claim-by-Claim Answer Verification
        verification_report = None
        final_answer = draft_answer
        if cfg.enable_verification:
            verification_report = await self._answer_verifier.verify_answer(
                question=clean_question,
                answer=draft_answer,
                evidence=all_citations,
            )
            final_answer = verification_report.verified_answer
            logger.info(
                "loop_verification_completed",
                loop_id=loop_id,
                is_verified=verification_report.is_verified,
                verified_ratio=verification_report.verified_ratio,
            )

        total_duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        overall_status = "completed" if exec_result.status != "failed" else "failed"

        logger.info(
            "agentic_research_loop_finished",
            loop_id=loop_id,
            total_duration_ms=total_duration_ms,
            status=overall_status,
        )

        return AgenticResearchLoopResult(
            loop_id=loop_id,
            question=clean_question,
            query_analysis=analysis,
            research_plan=plan,
            subquestion_results=subquestion_results,
            evidence_store=evidence_store,
            draft_answer=exec_result.final_synthesis,
            self_correction_result=self_correction_result,
            verification_report=verification_report,
            final_answer=final_answer,
            citations=all_citations,
            total_duration_ms=total_duration_ms,
            status=overall_status,
            metadata={
                "subquestion_count": len(plan.subquestions),
                "total_evidence_items": evidence_store.total_evidence_items,
            },
        )
