import pytest
from app.config import Settings
from app.main import create_app
from app.services.rag.analyzer import QueryAnalyzer
from app.services.rag.query_analysis import QueryAnalysis, QueryIntent
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_analyze_factual_query(settings: Settings) -> None:
    """Verify factual query extraction: entities, technical specs, and single subquestion."""
    analyzer = QueryAnalyzer(settings=settings)
    query = "What is the titanium wing spar thickness on Boeing 777X?"
    analysis: QueryAnalysis = await analyzer.analyze(query)

    assert analysis.original_query == query
    assert analysis.intent in (QueryIntent.FACTUAL, QueryIntent.ANALYTICAL)
    assert not analysis.is_ambiguous
    assert len(analysis.entities) > 0

    entity_texts = [e.text.lower() for e in analysis.entities]
    assert any("boeing" in t for t in entity_texts)
    assert any("titanium" in t for t in entity_texts)
    assert any("777x" in t.lower() for t in entity_texts)

    assert "technical_specifications" in analysis.required_information_types
    assert len(analysis.subquestions) == 1


@pytest.mark.asyncio
async def test_analyze_comparison_query(settings: Settings) -> None:
    """Verify comparison query decomposition across two organizations with subquestions."""
    analyzer = QueryAnalyzer(settings=settings)
    query = (
        "What were the main reasons for Boeing's production delays "
        "and how did those compare with Airbus?"
    )
    analysis: QueryAnalysis = await analyzer.analyze(query)

    assert analysis.original_query == query
    assert analysis.intent == QueryIntent.COMPARISON
    assert not analysis.is_ambiguous
    assert len(analysis.entities) >= 2

    entity_texts = [e.text.lower() for e in analysis.entities]
    assert "boeing" in entity_texts
    assert "airbus" in entity_texts

    # Must contain subquestions splitting the comparison
    assert len(analysis.subquestions) >= 2
    assert any("boeing" in sq.lower() for sq in analysis.subquestions)
    assert any("airbus" in sq.lower() for sq in analysis.subquestions)

    assert "delay_causes" in analysis.required_information_types
    assert "comparative_benchmarks" in analysis.required_information_types
    assert any("annual_reports" in s for s in analysis.potential_source_types)


@pytest.mark.asyncio
async def test_analyze_multi_part_research_query(settings: Settings) -> None:
    """Verify multi-part research query extracts temporal scope and multi-part subquestions."""
    analyzer = QueryAnalyzer(settings=settings)
    query = (
        "Analyze the impact of EU carbon regulations on European budget airlines "
        "from 2020 to 2025, assess fleet renewal costs, and evaluate passenger surcharge trends."
    )
    analysis: QueryAnalysis = await analyzer.analyze(query)

    assert analysis.original_query == query
    assert analysis.intent in (QueryIntent.MULTI_PART_RESEARCH, QueryIntent.ANALYTICAL)
    assert not analysis.is_ambiguous
    assert analysis.temporal_scope == "2020 - 2025"

    entity_texts = [e.text.lower() for e in analysis.entities]
    assert any("eu" in t for t in entity_texts)
    assert any("carbon regulations" in t for t in entity_texts)

    assert len(analysis.subquestions) >= 2
    assert "financial_metrics" in analysis.required_information_types


@pytest.mark.asyncio
async def test_analyze_ambiguous_query(settings: Settings) -> None:
    """Verify ambiguous or underspecified queries trigger ambiguity flags and clarification."""
    analyzer = QueryAnalyzer(settings=settings)
    query = "Tell me about delays"
    analysis: QueryAnalysis = await analyzer.analyze(query)

    assert analysis.original_query == query
    assert analysis.intent == QueryIntent.AMBIGUOUS
    assert analysis.is_ambiguous is True
    assert analysis.clarification_needed is not None
    assert "specify" in analysis.clarification_needed.lower()


@pytest.mark.asyncio
async def test_analyze_procedural_query(settings: Settings) -> None:
    """Verify procedural intent classification for step-by-step guides."""
    analyzer = QueryAnalyzer(settings=settings)
    query = "How to inspect the flutter damper titanium fittings?"
    analysis: QueryAnalysis = await analyzer.analyze(query)

    assert analysis.intent == QueryIntent.PROCEDURAL
    assert not analysis.is_ambiguous
    assert any("flutter damper" in e.text.lower() for e in analysis.entities)


@pytest.mark.asyncio
async def test_analyze_empty_query_raises_value_error(settings: Settings) -> None:
    """Empty or whitespace-only query raises ValueError."""
    analyzer = QueryAnalyzer(settings=settings)
    with pytest.raises(ValueError, match="Query string cannot be empty"):
        await analyzer.analyze("   ")


@pytest.mark.asyncio
async def test_analyze_query_endpoint(settings: Settings) -> None:
    """Verify POST /api/v1/rag/analyze-query endpoint returns structured analysis."""
    app = create_app(settings=settings)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        payload = {
            "query": (
                "What were the main reasons for Boeing's production delays "
                "and how did those compare with Airbus?"
            )
        }
        resp = await client.post("/api/v1/rag/analyze-query", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["original_query"] == payload["query"]
        assert data["intent"] == "comparison"
        assert data["is_ambiguous"] is False
        assert len(data["entities"]) >= 2
        assert len(data["subquestions"]) >= 2
        assert "delay_causes" in data["required_information_types"]
        assert len(data["potential_source_types"]) > 0


@pytest.mark.asyncio
async def test_analyze_query_endpoint_validation_error(settings: Settings) -> None:
    """Verify POST /api/v1/rag/analyze-query returns 422 for empty query."""
    app = create_app(settings=settings)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        resp = await client.post("/api/v1/rag/analyze-query", json={"query": ""})
        assert resp.status_code == 422
