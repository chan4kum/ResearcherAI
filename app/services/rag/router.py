import asyncio
import json
import re

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent
from app.services.rag.routing import RoutingDecision, SourceDestination
from app.services.rag.sources.models import SourceResult, SourceType
from app.services.rag.sources.registry import RetrievalSourceRegistry

logger = get_logger("app.services.rag.router")

ROUTER_SYSTEM_PROMPT = """You are an expert Retrieval Router.
Analyze the user's question and select the appropriate information source(s).

Available Source Destinations:
- internal_documents: Company policies, employee handbook, proprietary technical documentation.
- external_web: Public company news, current events, stock earnings, market trends.
- structured_database: Tabular logs, structured metrics, SQL records, maintenance logs.

Rules:
- If a query compares internal metrics with public data, select BOTH internal and external sources.
- Output JSON strictly conforming to:
{
  "selected_sources": ["internal_documents"],
  "reason": "Brief justification for source selection.",
  "confidence": 0.95
}
"""


class RetrievalRouter:
    """Intelligent router directing queries to optimal heterogeneous knowledge sources."""

    def __init__(
        self,
        query_analyzer: QueryAnalyzer | None = None,
        llm_service: LLMService | None = None,
        registry: RetrievalSourceRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._query_analyzer = query_analyzer or QueryAnalyzer(
            llm_service=self._llm_service, settings=self._settings
        )
        self._registry = registry or RetrievalSourceRegistry()

    @property
    def registry(self) -> RetrievalSourceRegistry:
        return self._registry

    def _heuristic_route(self, query: str, analysis: QueryAnalysis) -> RoutingDecision:
        """Deterministic rule-based routing for offline test execution and high performance."""
        q_lower = query.lower()

        internal_keywords = {
            "internal", "our policy", "our numbers", "our sales", "our team",
            "employee handbook", "handbook", "sdr", "sop", "proprietary",
            "company policy", "internal documents", "our quality", "our codebase",
        }
        external_keywords = {
            "latest earnings", "earnings", "market information", "public news",
            "stock price", "nvidia", "apple", "google", "quarterly results",
            "what happened in", "recent news", "public market", "industry trends",
            "external",
        }
        structured_keywords = {
            "maintenance log", "inspection records", "tail number", "tabular",
            "database records", "sql", "audit score", "aircraft registry",
            "flight log", "table", "fleet status",
        }

        has_internal = any(k in q_lower for k in internal_keywords)
        has_external = any(k in q_lower for k in external_keywords)
        has_structured = any(k in q_lower for k in structured_keywords)

        selected: list[SourceDestination] = []
        reason = ""

        # 1. Comparative / Compound (Internal + External)
        if (has_internal and has_external) or (
            analysis.intent == QueryIntent.COMPARISON and (has_internal or has_external)
        ):
            selected = [SourceDestination.INTERNAL_DOCUMENTS, SourceDestination.EXTERNAL_WEB]
            reason = (
                "Comparative query requires cross-referencing internal company data with "
                "public external market information."
            )
        # 2. Structured Relational Data
        elif has_structured:
            selected = [SourceDestination.STRUCTURED_DATABASE]
            reason = "Query requests tabular metrics, maintenance logs, or structured SQL records."
        # 3. External Web / Market
        elif has_external:
            selected = [SourceDestination.EXTERNAL_WEB]
            reason = (
                "Query asks about external corporate earnings, market developments, "
                "or public announcements."
            )
        # 4. Internal Documents (Default for enterprise knowledge queries)
        else:
            selected = [SourceDestination.INTERNAL_DOCUMENTS]
            reason = (
                "Query targets internal company policies, standard operating procedures, "
                "or proprietary documentation."
            )

        entities_detected = [e.text for e in analysis.entities]

        logger.info(
            "retrieval_routing_decision",
            query=query[:80],
            intent=analysis.intent.value,
            selected_sources=[s.value for s in selected],
            reason=reason,
        )

        return RoutingDecision(
            query=query,
            intent=analysis.intent,
            selected_sources=selected,
            reason=reason,
            confidence=0.95,
            entities_detected=entities_detected,
        )

    async def route(self, query: str) -> RoutingDecision:
        """Analyze query semantics and select target knowledge source destinations."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query cannot be empty for routing.")

        # 1. Deconstruct query structure
        analysis = await self._query_analyzer.analyze(clean_query)

        # 2. In mock LLM mode, execute deterministic routing logic
        if getattr(self._settings, "llm_provider", "mock") == "mock":
            return self._heuristic_route(clean_query, analysis)

        # 3. In real LLM mode, prompt router model with JSON constraint
        try:
            user_prompt = (
                f"Query: {clean_query}\n"
                f"Detected Intent: {analysis.intent.value}\n"
                f"Detected Entities: {', '.join(e.text for e in analysis.entities) or 'None'}\n\n"
                f"Select routing destinations:"
            )
            response = await self._llm_service.generate(
                prompt=user_prompt,
                system_prompt=ROUTER_SYSTEM_PROMPT,
                temperature=0.0,
            )
            # Parse JSON block
            raw = response.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                payload = json.loads(match.group(0))
                sources = [
                    SourceDestination(s)
                    for s in payload.get("selected_sources", [])
                    if s in SourceDestination._value2member_map_
                ]
                if sources:
                    reason = payload.get("reason", "LLM-guided multi-source routing decision.")
                    confidence = float(payload.get("confidence", 0.9))
                    logger.info(
                        "retrieval_routing_decision_llm",
                        query=clean_query[:80],
                        intent=analysis.intent.value,
                        selected_sources=[s.value for s in sources],
                        reason=reason,
                    )
                    return RoutingDecision(
                        query=clean_query,
                        intent=analysis.intent,
                        selected_sources=sources,
                        reason=reason,
                        confidence=confidence,
                        entities_detected=[e.text for e in analysis.entities],
                    )
            return self._heuristic_route(clean_query, analysis)
        except Exception as exc:
            logger.warning("llm_routing_fallback_to_heuristic", error=str(exc))
            return self._heuristic_route(clean_query, analysis)

    async def route_and_retrieve(
        self,
        query: str,
        top_k_per_source: int = 3,
        min_relevance: float = 0.0,
    ) -> tuple[RoutingDecision, list[SourceResult]]:
        """Determine routing and execute search across mapped source instances."""
        decision = await self.route(query)

        # Map SourceDestination to registered SourceType instances
        target_source_types: list[SourceType] = []
        for dest in decision.selected_sources:
            if dest == SourceDestination.INTERNAL_DOCUMENTS:
                target_source_types.extend([SourceType.INTERNAL_VECTOR, SourceType.KEYWORD])
            elif dest == SourceDestination.EXTERNAL_WEB:
                target_source_types.append(SourceType.WEB_SEARCH)
            elif dest == SourceDestination.STRUCTURED_DATABASE:
                target_source_types.append(SourceType.STRUCTURED_DB)

        # Fetch matching registered source instances
        active_sources = []
        for st in target_source_types:
            active_sources.extend(self._registry.get_sources_by_type(st))

        # Deduplicate sources
        unique_sources = list({s.source_name: s for s in active_sources}.values())

        if not unique_sources:
            # Fallback to all registered sources if none match explicitly
            unique_sources = self._registry.list_sources()

        tasks = [
            source.search(
                query=query,
                top_k=top_k_per_source,
                min_relevance=min_relevance,
            )
            for source in unique_sources
        ]
        nested_results = await asyncio.gather(*tasks) if tasks else []
        flattened: list[SourceResult] = [
            res for sublist in nested_results for res in sublist
        ]
        flattened.sort(key=lambda r: r.relevance, reverse=True)

        return decision, flattened
