from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.document.models import MetadataFilter
from app.services.rag.bm25 import tokenize
from app.services.rag.models import Citation
from app.services.rag.sources.base import BaseRetrievalSource
from app.services.rag.sources.models import SourceResult, SourceType

logger = get_logger("app.services.rag.sources.web")

# Preloaded mock web repository entries for offline testing
DEFAULT_MOCK_WEB_ENTRIES: list[dict[str, Any]] = [
    {
        "url": "https://aviation-safety.org/bulletins/2026/boeing-777x-flutter-investigation",
        "title": (
            "Aviation Safety Bureau: Boeing 777X Flutter Damper Ultrasonic Inspection Guidelines"
        ),
        "domain": "aviation-safety.org",
        "published_date": "2026-03-15",
        "content": (
            "The Global Aviation Safety Bureau issued an advisory bulletin detailing mandatory "
            "ultrasonic testing schedules for titanium hydraulic fittings on Boeing 777X wing "
            "flutter dampers following recent flight test vibration anomalies."
        ),
    },
    {
        "url": "https://aerospace-daily.com/news/airbus-engine-delivery-delays-2026",
        "title": "Aerospace Daily: Airbus A321neo Delivery Schedules Impacted by Turbofan Supply",
        "domain": "aerospace-daily.com",
        "published_date": "2026-02-28",
        "content": (
            "Airbus commercial aircraft division reported revised quarterly delivery guidance "
            "for the A321neo narrowbody line, citing component casting backlogs and Pratt & "
            "Whitney PW1100G engine delivery timelines."
        ),
    },
    {
        "url": "https://global-trade-monitor.org/reports/aerospace-supply-chain-2026",
        "title": "Global Trade Monitor: International Aerospace Titanium Supply Chain Bottlenecks",
        "domain": "global-trade-monitor.org",
        "published_date": "2026-01-10",
        "content": (
            "Global aerospace manufacturing supply chains continue to experience raw material "
            "constraints for aerospace-grade titanium forgings and carbon composite polymers."
        ),
    },
]


class WebSearchPlaceholderSource(BaseRetrievalSource):
    """Retrieval source providing real Tavily AI web search with deterministic offline fallback."""

    def __init__(
        self,
        source_name: str = "web_search_engine",
        api_key: str | None = None,
        mock_entries: list[dict[str, Any]] | None = None,
    ) -> None:
        self._source_name = source_name
        self._api_key = api_key
        self._entries = mock_entries or list(DEFAULT_MOCK_WEB_ENTRIES)

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def source_name(self) -> str:
        return self._source_name

    def add_mock_entry(
        self,
        url: str,
        title: str,
        content: str,
        domain: str = "example.com",
    ) -> None:
        """Add a custom web entry for test isolation."""
        self._entries.append({
            "url": url,
            "title": title,
            "domain": domain,
            "content": content,
        })

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> list[SourceResult]:
        """Search via Tavily Live API or fall back to offline token index."""
        clean_query = query.strip()
        if not clean_query:
            return []

        # If Tavily API Key is configured, attempt live web search
        if self._api_key:
            try:
                results = await self._search_tavily(clean_query, top_k, min_relevance)
                if results:
                    return results
            except Exception as e:
                logger.warning("tavily_search_failed_fallback_to_mock", error=str(e), query=clean_query[:80])

        # Offline fallback search
        return self._search_offline_mock(clean_query, top_k, min_relevance)

    async def _search_tavily(self, query: str, top_k: int, min_relevance: float) -> list[SourceResult]:
        """Execute search using Tavily AI Search API."""
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": top_k,
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                logger.warning("tavily_api_error_status", status=response.status_code, body=response.text[:200])
                return []

            data = response.json()
            raw_results = data.get("results", [])

            results: list[SourceResult] = []
            for idx, r in enumerate(raw_results, start=1):
                raw_score = r.get("score", 0.95 - (idx * 0.05))
                score = round(float(raw_score), 4)
                if score < min_relevance:
                    continue

                url_val = r.get("url", f"https://web.source/{idx}")
                title = r.get("title", f"Web Result #{idx}")
                content = r.get("content", "")

                citation = Citation(
                    chunk_id=f"tavily_{idx}_{abs(hash(url_val)) % 10000}",
                    doc_id=f"web_{abs(hash(url_val)) % 10000}",
                    source=url_val,
                    file_type="web",
                    chunk_index=0,
                    content=content,
                    similarity=score,
                    metadata={
                        "url": url_val,
                        "title": title,
                        "score": score,
                    },
                )

                result = SourceResult(
                    source=self.source_name,
                    source_type=self.source_type,
                    content=content,
                    relevance=score,
                    metadata={
                        "url": url_val,
                        "title": title,
                    },
                    citation=citation,
                )
                results.append(result)

            logger.info("tavily_live_search_success", query=query[:80], count=len(results))
            return results

    def _search_offline_mock(self, clean_query: str, top_k: int, min_relevance: float) -> list[SourceResult]:
        """Search simulated web index via token overlap and rank results."""
        query_tokens = set(tokenize(clean_query.lower()))
        scored_entries: list[tuple[dict[str, Any], float]] = []

        for entry in self._entries:
            entry_text = f"{entry.get('title', '')} {entry.get('content', '')}".lower()
            entry_tokens = set(tokenize(entry_text))

            overlap = len(query_tokens.intersection(entry_tokens))
            if overlap > 0 or not query_tokens:
                relevance = round(overlap / (len(query_tokens) or 1), 4)
                if relevance >= min_relevance:
                    scored_entries.append((entry, relevance))

        # Sort by relevance descending
        scored_entries.sort(key=lambda x: x[1], reverse=True)
        top_entries = scored_entries[:top_k]

        results: list[SourceResult] = []
        for idx, (entry, score) in enumerate(top_entries, start=1):
            url = entry.get("url", f"https://mocksearch.local/result/{idx}")
            title = entry.get("title", f"Web Search Result #{idx}")
            content = entry.get("content", "")

            citation = Citation(
                chunk_id=f"web_chunk_{idx}_{abs(hash(url)) % 10000}",
                doc_id=f"web_doc_{abs(hash(url)) % 10000}",
                source=url,
                file_type="html",
                chunk_index=0,
                content=content,
                similarity=score,
                metadata={
                    "url": url,
                    "title": title,
                    "domain": entry.get("domain", "web"),
                    "published_date": entry.get("published_date"),
                },
            )

            result = SourceResult(
                source=self.source_name,
                source_type=self.source_type,
                content=content,
                relevance=score,
                metadata={
                    "url": url,
                    "title": title,
                    "domain": entry.get("domain", "web"),
                    "published_date": entry.get("published_date"),
                },
                citation=citation,
            )
            results.append(result)

        logger.info(
            "web_source_search_completed",
            source=self.source_name,
            query=clean_query[:80],
            results_found=len(results),
        )
        return results
