import pytest
from app.config import Settings
from app.main import app
from app.services.rag.models import Citation
from app.services.rag.research import (
    MultiStepResearchExecutor,
    MultiStepResearchPlanner,
    ResearchEvidenceStore,
    ResearchExecutionResult,
    ResearchPlan,
    ResearchSubquestion,
    ResearchSubquestionType,
    SubquestionExecutionResult,
    SubquestionExecutionStatus,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_sequential_research_execution_flow(settings: Settings) -> None:
    """Verify sequential execution of a complex multi-step research plan."""
    executor = MultiStepResearchExecutor(settings=settings)
    query = (
        "Compare TSMC and Intel manufacturing strategies, identify their major technology "
        "challenges, and assess implications for the semiconductor industry."
    )
    result = await executor.execute_research(query=query, top_k_per_source=2)

    assert isinstance(result, ResearchExecutionResult)
    assert result.original_query == query
    assert result.status == "completed"
    assert len(result.subquestion_results) == 6
    assert len(result.final_synthesis) > 50
    assert result.total_duration_ms > 0

    # Verify each subquestion retained required fields
    for sq_res in result.subquestion_results:
        assert sq_res.subquestion_id.startswith("subq_")
        assert len(sq_res.query) > 0
        assert sq_res.status == SubquestionExecutionStatus.COMPLETED
        assert isinstance(sq_res.sources, list)
        assert isinstance(sq_res.evidence, list)
        assert isinstance(sq_res.citations, list)
        assert len(sq_res.sub_answer) > 0


@pytest.mark.asyncio
async def test_research_evidence_store_operations() -> None:
    """Verify in-memory evidence store aggregation, citation deduplication, and formatting."""
    store = ResearchEvidenceStore()

    cit1 = Citation(
        chunk_id="chunk_1",
        doc_id="doc_tsmc",
        source="tsmc_strategy.pdf",
        file_type="pdf",
        chunk_index=0,
        content="TSMC 2nm N2 process enters mass production.",
        similarity=0.95,
    )
    cit2 = Citation(
        chunk_id="chunk_2",
        doc_id="doc_intel",
        source="intel_ifs.pdf",
        file_type="pdf",
        chunk_index=1,
        content="Intel 18A RibbonFET technology starts manufacturing.",
        similarity=0.92,
    )

    res1 = SubquestionExecutionResult(
        subquestion_id="subq_1",
        index=1,
        query="What is TSMC's manufacturing strategy?",
        sources=["internal_vector_db"],
        evidence=["TSMC 2nm N2 process enters mass production."],
        citations=[cit1],
        sub_answer="TSMC is leading advanced node production with N2.",
        status=SubquestionExecutionStatus.COMPLETED,
        duration_ms=45.2,
    )
    res2 = SubquestionExecutionResult(
        subquestion_id="subq_2",
        index=2,
        query="What is Intel's manufacturing strategy?",
        sources=["internal_vector_db"],
        evidence=["Intel 18A RibbonFET technology starts manufacturing."],
        citations=[cit2, cit1],  # Contains duplicate cit1
        sub_answer="Intel IFS focus on 18A and packaging.",
        status=SubquestionExecutionStatus.COMPLETED,
        duration_ms=48.1,
    )

    store.add_result(res1)
    store.add_result(res2)

    assert len(store.get_all_results()) == 2
    assert len(store.get_all_evidence()) == 2
    # Deduped citations should be 2 (cit1 and cit2)
    assert len(store.get_all_citations()) == 2

    answers = store.get_intermediate_answers()
    assert len(answers) == 2
    assert "TSMC" in answers[0][1]

    formatted_context = store.format_synthesis_context()
    assert "### Subquestion 1:" in formatted_context
    assert "### Subquestion 2:" in formatted_context
    assert "TSMC 2nm" in formatted_context

    store.clear()
    assert len(store.get_all_results()) == 0


@pytest.mark.asyncio
async def test_research_executor_with_predefined_plan(settings: Settings) -> None:
    """Verify research executor accepts a custom pre-computed ResearchPlan."""
    planner = MultiStepResearchPlanner(settings=settings)
    custom_plan = ResearchPlan(
        plan_id="plan_custom_boeing",
        original_query="Compare Boeing and Airbus supply chain resilience.",
        overall_goal="Evaluate commercial aviation supply chain stability",
        subquestions=[
            ResearchSubquestion(
                id="subq_1",
                index=1,
                question="What are Boeing's main supply chain bottlenecks?",
                subquestion_type=ResearchSubquestionType.CHALLENGE,
                target_entities=["Boeing"],
                expected_output_type="summary",
                suggested_sources=[],
                depends_on=[],
            ),
            ResearchSubquestion(
                id="subq_2",
                index=2,
                question="What are Airbus's main supply chain bottlenecks?",
                subquestion_type=ResearchSubquestionType.CHALLENGE,
                target_entities=["Airbus"],
                expected_output_type="summary",
                suggested_sources=[],
                depends_on=[],
            ),
        ],
        estimated_complexity="medium",
        suggested_synthesis_strategy="Direct comparative matrix",
        created_at="2026-08-23T12:00:00Z",
    )

    executor = MultiStepResearchExecutor(planner=planner, settings=settings)
    result = await executor.execute_research(
        query="Compare Boeing and Airbus supply chain resilience.",
        plan=custom_plan,
        top_k_per_source=2,
    )

    assert result.plan.plan_id == "plan_custom_boeing"
    assert len(result.subquestion_results) == 2
    assert result.status == "completed"


@pytest.mark.asyncio
async def test_execute_research_endpoint() -> None:
    """Verify POST /api/v1/rag/research/execute HTTP 200 response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "query": (
                "Compare TSMC and Intel manufacturing strategies, identify their major "
                "technology challenges, and assess implications for the semiconductor industry."
            ),
            "top_k_per_source": 2,
        }
        res = await client.post("/api/v1/rag/research/execute", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "research_id" in data
        assert "final_synthesis" in data
        assert data["status"] in ("completed", "partial")
        assert "subquestion_results" in data
        assert len(data["subquestion_results"]) == 6


@pytest.mark.asyncio
async def test_execute_research_endpoint_validation_error() -> None:
    """Verify empty query returns 422 Unprocessable Entity."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post("/api/v1/rag/research/execute", json={"query": ""})
        assert res.status_code == 422
