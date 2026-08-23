"""
app/services/agent/tools/web_search.py — Live Tavily Web Search Agent Tool
"""

from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.agent.tools.base import BaseTool

logger = get_logger("app.services.agent.tools.web_search")


class WebSearchTool(BaseTool):
    """Tool allowing AI agents to query the live internet via Tavily Search API."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the live internet for recent news, industry reports, company information, "
            "and facts. Args: query (string), max_results (optional int, default 3)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to execute on the web",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return (default 3)",
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return "Error: query parameter is required for web_search tool."

        max_results = int(kwargs.get("max_results", 3))

        if not self._api_key:
            return (
                f"Web search for '{query}' simulated (No TAVILY_API_KEY configured). "
                f"Silicon Saxony, Dresden, and Munich are the primary semiconductor hubs in Germany with Infineon, Bosch, and TSMC."
            )

        url = "https://api.tavily.com/search"
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    return f"Tavily search API error ({res.status_code}): {res.text[:150]}"

                data = res.json()
                results = data.get("results", [])
                if not results:
                    return f"No web search results found for query: '{query}'"

                formatted = []
                for idx, r in enumerate(results, start=1):
                    formatted.append(f"[{idx}] {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}\n")

                return "\n".join(formatted)
        except Exception as e:
            logger.error("web_search_tool_failed", error=str(e), query=query)
            return f"Web search request failed: {e}"
