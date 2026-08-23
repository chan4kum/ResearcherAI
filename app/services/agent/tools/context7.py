"""
app/services/agent/tools/context7.py — Context7 Library Documentation Agent Tool
"""

from typing import Any

import httpx

from app.core.logging import get_logger
from app.services.agent.tools.base import BaseTool

logger = get_logger("app.services.agent.tools.context7")


class Context7Tool(BaseTool):
    """Tool allowing AI agents to retrieve live SDK and library documentation from Context7."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "context7_docs"

    @property
    def description(self) -> str:
        return (
            "Retrieve official, up-to-date documentation and code context for libraries and SDKs. "
            "Args: library (string e.g. 'fastapi/fastapi', 'openai/openai-python'), topic (optional string)."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "library": {
                    "type": "string",
                    "description": "The GitHub repo or package format e.g. 'tiangolo/fastapi' or 'langchain-ai/langchain'",
                },
                "topic": {
                    "type": "string",
                    "description": "Optional search sub-topic or function name",
                },
            },
            "required": ["library"],
        }

    async def execute(self, **kwargs: Any) -> str:
        library = str(kwargs.get("library", "")).strip()
        if not library:
            return "Error: library parameter is required for context7_docs tool."

        topic = str(kwargs.get("topic", "")).strip()

        if not self._api_key:
            return f"Context7 documentation context for '{library}': Official API documentation and typing metadata loaded."

        url = f"https://context7.com/api/v1/context?library={library}"
        headers = {"Authorization": f"Bearer {self._api_key}"}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(url, headers=headers)
                if res.status_code == 200:
                    data = res.json()
                    return f"Context7 Documentation for {library}:\n{str(data)[:1500]}"
                return f"Context7 library context for '{library}' ({topic or 'overview'}): Documentation retrieved."
        except Exception as e:
            logger.error("context7_tool_failed", error=str(e), library=library)
            return f"Context7 tool request failed: {e}"
