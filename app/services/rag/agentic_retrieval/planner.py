from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.agentic_retrieval.models import RetrievalPlan
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent
from app.services.rag.router import RetrievalRouter

logger = get_logger("app.services.rag.agentic.planner")

CONVERSATIONAL_GREETINGS = {
    "hello", "hi", "hey", "good morning", "good evening", "how are you",
    "who are you", "what can you do", "thanks", "thank you",
}


class RetrievalPlanner:
    """Decides if retrieval is necessary, formulates subgoals, and drafts search queries."""

    def __init__(
        self,
        query_analyzer: QueryAnalyzer | None = None,
        router: RetrievalRouter | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._query_analyzer = query_analyzer or QueryAnalyzer(
            llm_service=self._llm_service, settings=self._settings
        )
        self._router = router or RetrievalRouter(
            query_analyzer=self._query_analyzer,
            llm_service=self._llm_service,
            settings=self._settings,
        )

    async def create_plan(self, query: str) -> tuple[QueryAnalysis, RetrievalPlan]:
        """Analyze query and formulate structured retrieval plan."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty for retrieval planning.")

        # 1. Check if conversational or non-retrieval
        q_lower = clean_query.lower()
        domain_indicators = {
            "boeing", "airbus", "policy", "earnings", "contract", "system",
            "spec", "damper", "revenue", "fiscal", "report", "database",
            "sql", "table", "flight", "inspection", "hardware", "nvidia",
        }
        has_domain_term = any(d in q_lower for d in domain_indicators)
        has_greeting_term = any(g in q_lower for g in CONVERSATIONAL_GREETINGS)

        if has_greeting_term and not has_domain_term:
            analysis = QueryAnalysis(
                original_query=clean_query,
                intent=QueryIntent.FACTUAL,
                confidence_score=0.95,
            )
            plan = RetrievalPlan(
                needs_retrieval=False,
                subgoals=[],
                target_sources=[],
                planned_queries=[],
                rationale="Query is conversational; direct LLM response is sufficient.",
            )
            logger.info("retrieval_plan_conversational", query=clean_query)
            return analysis, plan

        # 2. Decompose query and route
        analysis = await self._query_analyzer.analyze(clean_query)
        routing = await self._router.route(clean_query)

        subgoals = analysis.subquestions if analysis.subquestions else [clean_query]
        planned_queries = [sq for sq in subgoals if sq.strip()]

        plan = RetrievalPlan(
            needs_retrieval=True,
            subgoals=subgoals,
            target_sources=routing.selected_sources,
            planned_queries=planned_queries,
            rationale=routing.reason,
        )

        logger.info(
            "retrieval_plan_formulated",
            query=clean_query[:80],
            needs_retrieval=True,
            subgoals_count=len(subgoals),
            target_sources=[s.value for s in routing.selected_sources],
        )
        return analysis, plan
