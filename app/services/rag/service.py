import time
from typing import Any

from app.config import Settings, get_settings
from app.core.guardrails.injection import PromptInjectionGuard
from app.core.guardrails.secrets_filter import SecretsScrubber
from app.core.logging import get_logger
from app.core.metrics import (
    RETRIEVAL_DURATION_SECONDS,
    RETRIEVAL_EXECUTIONS_TOTAL,
    RETRIEVAL_ITERATIONS_TOTAL,
)
from app.core.tracing import agent_span
from app.core.versioning.manager import get_version_manager
from app.core.versioning.prompts import get_prompt_registry
from app.services.llm.service import LLMService

from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.evaluator import RetrievalEvaluator
from app.services.rag.hyde import HyDEGenerator
from app.services.rag.models import Citation, RAGResponse
from app.services.rag.reranker import BaseReranker, RerankSummary, create_reranker
from app.services.rag.retriever import (
    BaseRetriever,
    HybridRetriever,
    HyDERetriever,
    VectorRetriever,
)
from app.services.rag.rewriter import QueryRewriteAttempt, QueryRewriter

logger = get_logger("app.services.rag.service")

DEFAULT_RAG_SYSTEM_PROMPT = (
    get_prompt_registry().get("rag_synthesizer").template
)


class RAGService:
    """Domain service orchestrating Question -> Rewriting -> Retrieval -> Rerank -> LLM."""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: BaseReranker | None = None,
        query_analyzer: QueryAnalyzer | None = None,
        evaluator: RetrievalEvaluator | None = None,
        rewriter: QueryRewriter | None = None,
        hyde_generator: HyDEGenerator | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._retriever = retriever
        self._settings = settings or get_settings()
        self._reranker = reranker
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._query_analyzer = query_analyzer or QueryAnalyzer(
            llm_service=self._llm_service, settings=self._settings
        )
        self._evaluator = evaluator or RetrievalEvaluator()
        self._rewriter = rewriter or QueryRewriter(
            llm_service=self._llm_service, settings=self._settings
        )
        self._hyde_generator = hyde_generator or HyDEGenerator(
            llm_service=self._llm_service, settings=self._settings
        )

    @property
    def retriever(self) -> BaseRetriever:
        return self._retriever

    @property
    def reranker(self) -> BaseReranker | None:
        return self._reranker

    @property
    def query_analyzer(self) -> QueryAnalyzer:
        return self._query_analyzer

    @property
    def evaluator(self) -> RetrievalEvaluator:
        return self._evaluator

    @property
    def rewriter(self) -> QueryRewriter:
        return self._rewriter

    @property
    def hyde_generator(self) -> HyDEGenerator:
        return self._hyde_generator

    def format_context(self, citations: list[Citation]) -> str:
        """Format retrieved citations into a structured, numbered context block."""
        if not citations:
            return "No relevant context found in the knowledge base."

        formatted_blocks: list[str] = []
        for i, cite in enumerate(citations, start=1):
            sanitized_content = PromptInjectionGuard.sanitize_retrieved_context(cite.content.strip())
            header = (
                f"[Citation {i}] (Source: {cite.source}, Chunk: {cite.chunk_index}, "
                f"Similarity: {cite.similarity:.4f})"
            )
            block = f"{header}\n{sanitized_content}"
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)

    def build_prompts(
        self,
        question: str,
        citations: list[Citation],
        custom_system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Construct the system prompt and augmented user prompt containing cited context."""
        system_prompt = custom_system_prompt or DEFAULT_RAG_SYSTEM_PROMPT
        context_str = self.format_context(citations)

        user_prompt = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n\n"
            f"Question: {question.strip()}\n\n"
            f"Answer based on the context above, citing [Citation X] where appropriate:"
        )
        return system_prompt, user_prompt

    async def answer(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        system_prompt: str | None = None,
        filters: Any = None,
        retriever: BaseRetriever | None = None,
        rerank: bool | None = None,
        top_n: int | None = None,
        reranker: BaseReranker | None = None,
        enable_rewriting: bool | None = None,
        max_attempts: int | None = None,
        strategy: str | None = None,
        hyde: bool | None = None,
    ) -> RAGResponse:
        """Execute complete RAG flow: Strategy -> Analyze -> Retrieve -> Rerank -> LLM."""
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty or whitespace.")

        # Determine retrieval strategy: "normal" or "hyde"
        strat_candidate = (
            strategy or getattr(self._settings, "default_retrieval_strategy", "normal") or "normal"
        )
        resolved_strategy: str = (
            "hyde"
            if hyde is True or (strategy and strategy.lower().strip() == "hyde")
            else str(strat_candidate).lower().strip()
        )

        active_retriever = retriever or self._retriever

        # If HyDE strategy requested, adapt retriever if needed
        if resolved_strategy == "hyde" and not isinstance(active_retriever, HyDERetriever):
            if isinstance(active_retriever, VectorRetriever):
                active_retriever = HyDERetriever(
                    embedding_service=active_retriever._embedding_service,
                    vector_repository=active_retriever._vector_repository,
                    hyde_generator=self._hyde_generator,
                    settings=self._settings,
                )
            elif isinstance(active_retriever, HybridRetriever):
                active_retriever = HyDERetriever(
                    embedding_service=active_retriever.vector_retriever._embedding_service,
                    vector_repository=active_retriever.vector_retriever._vector_repository,
                    hyde_generator=self._hyde_generator,
                    settings=self._settings,
                )

        should_rerank = (
            rerank
            if rerank is not None
            else getattr(self._settings, "enable_reranking", False)
        )
        active_reranker = (
            reranker
            or self._reranker
            or (
                create_reranker(getattr(self._settings, "reranker_provider", "mock"))
                if should_rerank
                else None
            )
        )

        should_rewrite = (
            enable_rewriting
            if enable_rewriting is not None
            else getattr(self._settings, "enable_query_rewriting", False)
        )
        max_retries: int = int(
            max_attempts
            if max_attempts is not None
            else getattr(self._settings, "max_retrieval_attempts", 3) or 3
        )

        initial_k: int = int(
            max(top_n or getattr(self._settings, "reranker_top_n", 10), top_k)
            if should_rerank
            else top_k
        )

        logger.info(
            "rag_query_started",
            question=clean_question[:80],
            strategy=resolved_strategy,
            top_k=top_k,
            initial_k=initial_k,
            rerank_enabled=should_rerank,
            rewriting_enabled=should_rewrite,
            max_attempts=max_retries,
            filters=filters,
        )

        current_query = clean_question
        initial_citations: list[Citation] = []
        rewrite_history: list[QueryRewriteAttempt] = []
        seen_queries: set[str] = {clean_question}

        rag_start_time = time.perf_counter()

        with agent_span(
            "retrieval.query",
            strategy=resolved_strategy,
            extra={"retrieval.top_k": top_k, "retrieval.rerank_enabled": should_rerank},
        ) as retrieval_span:
            if should_rewrite:
                # Analyze query structure and semantics
                analysis = await self._query_analyzer.analyze(clean_question)

                for attempt_idx in range(1, max_retries + 1):
                    RETRIEVAL_ITERATIONS_TOTAL.labels(
                        strategy=resolved_strategy,
                        iteration=str(attempt_idx),
                    ).inc()

                    # Retrieve candidate chunks
                    initial_citations = await active_retriever.retrieve(
                        query=current_query,
                        top_k=initial_k,
                        min_similarity=min_similarity,
                        filters=filters,
                    )

                    # Evaluate retrieval quality
                    min_rel_thresh = getattr(
                        self._settings, "min_retrieval_relevance_threshold", 0.01
                    )
                    eval_res = self._evaluator.evaluate(
                        query=current_query,
                        analysis=analysis,
                        citations=initial_citations,
                        min_relevance=min_rel_thresh,
                    )

                    top_score = eval_res.relevance_score
                    avg_score = (
                        round(
                            sum(c.similarity for c in initial_citations)
                            / (len(initial_citations) or 1),
                            4,
                        )
                        if initial_citations
                        else 0.0
                    )

                    attempt_record = QueryRewriteAttempt(
                        attempt=attempt_idx,
                        query=current_query,
                        retrieved_count=len(initial_citations),
                        top_score=top_score,
                        average_score=avg_score,
                        is_sufficient=eval_res.is_sufficient,
                        reasons=[r.value for r in eval_res.reasons],
                        feedback=eval_res.feedback_prompt,
                    )
                    rewrite_history.append(attempt_record)

                    if eval_res.is_sufficient:
                        logger.info(
                            "retrieval_sufficient_reached",
                            attempt=attempt_idx,
                            query_len=len(current_query),
                        )
                        break

                    # If insufficient and attempts remain, rewrite query
                    if attempt_idx < max_retries:
                        next_query = await self._rewriter.rewrite(
                            original_query=clean_question,
                            analysis=analysis,
                            evaluation=eval_res,
                            attempt_index=attempt_idx + 1,
                            previous_queries=list(seen_queries),
                        )
                        if next_query in seen_queries:
                            logger.warning("duplicate_rewrite_detected_halting", query_len=len(next_query))
                            break
                        current_query = next_query
                        seen_queries.add(current_query)

            else:
                initial_citations = await active_retriever.retrieve(
                    query=current_query,
                    top_k=initial_k,
                    min_similarity=min_similarity,
                    filters=filters,
                )

            # 2. Optional Reranking stage (Top N -> Top K)
            rerank_summary: RerankSummary | None = None
            if should_rerank and active_reranker and initial_citations:
                citations, rerank_summary = await active_reranker.rerank(
                    query=current_query,
                    citations=initial_citations,
                    top_k=top_k,
                )
            else:
                citations = initial_citations[:top_k]

            duration_sec = time.perf_counter() - rag_start_time
            RETRIEVAL_EXECUTIONS_TOTAL.labels(strategy=resolved_strategy, status="success").inc()
            RETRIEVAL_DURATION_SECONDS.labels(strategy=resolved_strategy).observe(duration_sec)

            # Annotate span with retrieval outcome metadata (no document content)
            retrieval_span.set_attribute("retrieval.citations_returned", len(citations))
            retrieval_span.set_attribute("retrieval.rewrite_attempts", len(rewrite_history))
            retrieval_span.set_attribute("retrieval.duration_ms", round(duration_sec * 1000, 2))
            if rewrite_history:
                retrieval_span.set_attribute("retrieval.final_iteration", len(rewrite_history))

        # 3. Build augmented prompt
        sys_prompt, user_prompt = self.build_prompts(
            question=clean_question,
            citations=citations,
            custom_system_prompt=system_prompt,
        )

        # 4. Invoke LLM for grounded synthesis
        with agent_span(
            "retrieval.llm_synthesis",
            strategy=resolved_strategy,
            extra={"retrieval.citations_used": len(citations)},
        ) as llm_span:
            llm_response = await self._llm_service.generate(
                prompt=user_prompt,
                system_prompt=sys_prompt,
                temperature=0.2,  # Lower temperature for grounded factual generation
            )
            llm_span.set_attribute("llm.model", llm_response.model or "unknown")
            llm_span.set_attribute("llm.provider", llm_response.provider or "unknown")
            llm_span.set_attribute("llm.total_tokens", llm_response.total_tokens or 0)

        logger.info(
            "rag_query_completed",
            citations_count=len(citations),
            tokens=llm_response.total_tokens,
            reranked=bool(rerank_summary is not None),
            rewritten_attempts=len(rewrite_history),
        )

        metadata_dict: dict[str, Any] = {}
        if rerank_summary:
            metadata_dict["reranking"] = rerank_summary.model_dump()
        if rewrite_history:
            metadata_dict["query_rewriting"] = [a.model_dump() for a in rewrite_history]

        hypothetical_doc: str | None = None
        if isinstance(active_retriever, HyDERetriever) and active_retriever.last_result:
            hypothetical_doc = active_retriever.last_result.hypothetical_document
            metadata_dict["hyde"] = active_retriever.last_result.model_dump()

        # Milestone 50: Configuration and prompt version provenance
        version_mgr = get_version_manager(self._settings)
        metadata_dict["version_provenance"] = {
            "prompt_version": version_mgr.prompt_registry.get("rag_synthesizer").version,
            "retrieval_config_version": version_mgr.retrieval_config.version,
            "routing_config_version": version_mgr.routing_config.version,
            "model_config_version": version_mgr.model_config.version,
            "config_hash": version_mgr.retrieval_config.config_hash,
        }

        return RAGResponse(
            question=clean_question,
            final_query=current_query if should_rewrite else None,
            strategy=resolved_strategy,
            hypothetical_document=hypothetical_doc,
            answer=SecretsScrubber.scrub_text(llm_response.content),
            citations=citations,
            retrieved_chunks_count=len(citations),
            model=llm_response.model,
            provider=llm_response.provider,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            metadata=metadata_dict,
        )
