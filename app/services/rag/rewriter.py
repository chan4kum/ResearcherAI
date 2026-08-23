import re

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.evaluator import EvaluationReason, RetrievalEvaluation
from app.services.rag.models import Citation
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent

logger = get_logger("app.services.rag.rewriter")

REWRITE_SYSTEM_PROMPT = """You are an expert Query Reformulation & Retrieval Optimization engine.
Your task is to rewrite a search query that produced insufficient retrieval results.

Given:
1. Original user query
2. Query analysis (intent, entities, subquestions)
3. Evaluator feedback (e.g. missing entities, low relevance, poor coverage)
4. Previously attempted queries

Rules:
- Produce a single, focused search query designed to maximize recall and precision.
- Incorporate missing entities, resolve acronyms, or expand technical synonyms as needed.
- Do NOT output explanations or preamble. Output the rewritten query string ONLY.
"""


class QueryRewriteAttempt(BaseModel):
    """Telemetry item recording each iterative retrieval and evaluation step."""

    attempt: int = Field(description="1-based iteration index")
    query: str = Field(description="Query string executed in this attempt")
    retrieved_count: int = Field(description="Number of chunks retrieved")
    top_score: float = Field(description="Highest similarity or fusion score in this attempt")
    average_score: float = Field(description="Average similarity score across retrieved chunks")
    is_sufficient: bool = Field(description="Whether retrieval passed evaluation criteria")
    reasons: list[str] = Field(
        default_factory=list,
        description="Evaluation deficiency flags identified in this attempt",
    )
    feedback: str | None = Field(
        default=None,
        description="Prescriptive evaluator feedback",
    )


class IterativeRetrievalResult(BaseModel):
    """Aggregated outcome of the multi-attempt retrieval and rewriting pipeline."""

    original_query: str = Field(description="Original user question")
    final_query: str = Field(description="Final query executed")
    attempts: list[QueryRewriteAttempt] = Field(
        default_factory=list,
        description="Chronological log of all query rewrite attempts",
    )
    total_attempts: int = Field(description="Total number of retrieval attempts performed")
    is_sufficient: bool = Field(description="Whether final attempt satisfied sufficiency criteria")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Final set of retrieved citations",
    )


class QueryRewriter:
    """Generates targeted query reformulations based on retrieval evaluation feedback."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)

    def _heuristic_rewrite(
        self,
        original_query: str,
        analysis: QueryAnalysis,
        evaluation: RetrievalEvaluation,
        attempt_index: int,
        previous_queries: list[str],
    ) -> str:
        """Deterministic rule-based query rewriting for offline test stability."""
        # 1. Missing Entities Priority
        if evaluation.missing_entities:
            missing_str = " ".join(evaluation.missing_entities)
            candidate = f"{missing_str} {original_query}".strip()
            if candidate not in previous_queries:
                return candidate

        # 2. Comparison / Multi-part Subquestion Expansion
        if (
            EvaluationReason.POOR_COVERAGE in evaluation.reasons
            or analysis.intent in (QueryIntent.COMPARISON, QueryIntent.MULTI_PART_RESEARCH)
        ):
            if attempt_index - 1 < len(analysis.subquestions):
                sq = analysis.subquestions[attempt_index - 1]
                if sq not in previous_queries:
                    return sq

        # 3. Keyword / Concept Stripping (remove conversational fluff)
        fluff_words = {
            "what", "were", "the", "main", "reasons", "for", "and", "how",
            "did", "those", "compare", "with", "tell", "me", "about", "is",
            "are", "can", "you", "explain", "please", "in", "to", "of",
        }
        tokens = [
            w for w in re.findall(r"\b\w+\b", original_query)
            if w.lower() not in fluff_words
        ]
        if tokens:
            candidate = " ".join(tokens)
            if candidate not in previous_queries:
                return candidate

        # 4. Synonym / Source Type expansion fallback
        if analysis.potential_source_types:
            src_hint = analysis.potential_source_types[0].replace("_", " ")
            candidate = f"{original_query} {src_hint}"
            if candidate not in previous_queries:
                return candidate

        return f"{original_query} attempt_{attempt_index}"

    async def rewrite(
        self,
        original_query: str,
        analysis: QueryAnalysis,
        evaluation: RetrievalEvaluation,
        attempt_index: int,
        previous_queries: list[str],
    ) -> str:
        """Synthesize a new query variant addressing identified retrieval deficiencies."""
        logger.info(
            "query_rewrite_started",
            attempt=attempt_index,
            reasons=[r.value for r in evaluation.reasons],
        )

        # In mock LLM environment, use heuristic rewriter
        if getattr(self._settings, "llm_provider", "mock") == "mock":
            rewritten = self._heuristic_rewrite(
                original_query=original_query,
                analysis=analysis,
                evaluation=evaluation,
                attempt_index=attempt_index,
                previous_queries=previous_queries,
            )
            logger.info("query_rewritten_heuristic", new_query=rewritten)
            return rewritten

        # In real LLM environment, prompt LLM
        try:
            user_prompt = (
                f"Original Query: {original_query}\n"
                f"Identified Intent: {analysis.intent.value}\n"
                f"Missing Entities: {', '.join(evaluation.missing_entities) or 'None'}\n"
                f"Evaluator Feedback: {evaluation.feedback_prompt or 'Improve relevance'}\n"
                f"Previously Attempted Queries:\n"
                + "\n".join(f"- {q}" for q in previous_queries)
                + "\n\nGenerate the next reformulated query:"
            )
            response = await self._llm_service.generate(
                prompt=user_prompt,
                system_prompt=REWRITE_SYSTEM_PROMPT,
                temperature=0.3,
            )
            clean_res = response.content.strip().strip('"').strip("'")
            if clean_res and clean_res not in previous_queries:
                logger.info("query_rewritten_llm", new_query=clean_res)
                return clean_res
            return self._heuristic_rewrite(
                original_query, analysis, evaluation, attempt_index, previous_queries
            )
        except Exception as exc:
            logger.warning("query_rewrite_llm_failed_fallback", error=str(exc))
            return self._heuristic_rewrite(
                original_query, analysis, evaluation, attempt_index, previous_queries
            )
