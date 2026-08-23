"""
tests/test_external_sources.py — Tests for Tavily Search and Context7 Documentation Tools
"""

import pytest

from app.services.agent.tools.context7 import Context7Tool
from app.services.agent.tools.web_search import WebSearchTool
from app.services.rag.sources.web_source import WebSearchPlaceholderSource


@pytest.mark.asyncio
async def test_web_search_tool_offline_fallback():
    """Verify WebSearchTool returns deterministic results when no API key is provided."""
    tool = WebSearchTool(api_key=None)
    assert tool.name == "web_search"
    res = await tool.execute(query="Semiconductors Germany")
    assert "Silicon Saxony" in res or "simulated" in res


@pytest.mark.asyncio
async def test_web_search_tool_missing_query():
    """Verify error on empty query."""
    tool = WebSearchTool(api_key=None)
    res = await tool.execute()
    assert "Error: query parameter is required" in res


@pytest.mark.asyncio
async def test_context7_tool_offline_fallback():
    """Verify Context7Tool returns fallback documentation."""
    tool = Context7Tool(api_key=None)
    assert tool.name == "context7_docs"
    res = await tool.execute(library="fastapi/fastapi")
    assert "fastapi/fastapi" in res


@pytest.mark.asyncio
async def test_context7_tool_missing_library():
    """Verify error on empty library."""
    tool = Context7Tool(api_key=None)
    res = await tool.execute()
    assert "Error: library parameter is required" in res


@pytest.mark.asyncio
async def test_web_source_offline_mock():
    """Verify WebSearchPlaceholderSource searches offline mock entries."""
    source = WebSearchPlaceholderSource(api_key=None)
    results = await source.search(query="Boeing 777X flutter", top_k=2)
    assert len(results) >= 1
    assert "Boeing" in results[0].content
    assert results[0].citation is not None
