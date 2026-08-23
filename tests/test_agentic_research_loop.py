import pytest
from app.config import Settings
from app.main import app
from app.services.rag.loop import (
    AgenticResearchLoopConfig,
    AgenticResearchLoopResult,
    AgenticResearchOrchestrator,
)
from app.services.rag.query_analysis import QueryIntent
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_end_to_end_research_loop_flow(settings: Settings) -> None:
    """Verify complete multi-step agentic research loop execution end-to-end."""
    orchestrator = AgenticResearchOrchestrator(settings=settings)
    question = (
        "Compare TSMC and Intel manufacturing strategies, identify major "
        "technology challenges, and assess implications for the semiconductor industry."
    )

    config = AgenticResearchLoopConfig(
        max_research_iterations=2,
        max_concurrency=4,
        enable_self_correction=True,
        enable_verification=True,
        timeout_seconds=30.0,
    )

    result = await orchestrator.run(question=question, config=config)

    # 1. Output Type & ID
    assert isinstance(result, AgenticResearchLoopResult)
    assert result.loop_id.startswith("loop_")
    assert result.status == "completed"

    # 2. Query Analysis
    assert result.query_analysis.intent in (
        QueryIntent.MULTI_PART_RESEARCH,
        QueryIntent.COMPARISON,
    )
    assert result.query_analysis.is_complex is True

    # 3. Research Plan
    assert result.research_plan is not None
    assert len(result.research_plan.subquestions) >= 4

    # 4. Multi-Step Execution & Evidence Store
    assert len(result.subquestion_results) == len(result.research_plan.subquestions)
    assert result.evidence_store.total_evidence_items >= 0

    # 5. Draft Answer & Final Answer
    assert len(result.draft_answer) > 0
    assert len(result.final_answer) > 0

    # 6. Self-Correction Critic
    assert result.self_correction_result is not None
    assert result.self_correction_result.iterations >= 1

    # 7. Answer Verification
    assert result.verification_report is not None
    assert result.verification_report.total_claims >= 1


@pytest.mark.asyncio
async def test_single_step_research_loop(settings: Settings) -> None:
    """Verify research loop handling for single-entity factual queries."""
    orchestrator = AgenticResearchOrchestrator(settings=settings)
    question = "What is TSMC's manufacturing roadmap in Arizona?"

    config = AgenticResearchLoopConfig(
        enable_self_correction=False,
        enable_verification=True,
    )

    result = await orchestrator.run(question=question, config=config)

    assert result.status == "completed"
    assert len(result.final_answer) > 0
    assert result.self_correction_result is None
    assert result.verification_report is not None


@pytest.mark.asyncio
async def test_agentic_research_loop_endpoint() -> None:
    """Verify POST /api/v1/rag/research/loop REST API endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "question": (
                "Compare TSMC and Intel manufacturing strategies, "
                "identify key challenges, and assess industry impact."
            ),
            "max_research_iterations": 2,
            "max_concurrency": 4,
            "enable_self_correction": True,
            "enable_verification": True,
            "timeout_seconds": 30.0,
        }
        res = await client.post("/api/v1/rag/research/loop", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "loop_id" in data
        assert data["is_complex"] is True
        assert len(data["subquestions"]) >= 4
        assert len(data["subquestion_results"]) >= 4
        assert "final_answer" in data
        assert data["status"] == "completed"
        assert data["total_duration_ms"] > 0


@pytest.mark.asyncio
async def test_agentic_research_loop_empty_query_validation() -> None:
    """Verify 422 validation error when research question is empty."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "question": "   ",
        }
        res = await client.post("/api/v1/rag/research/loop", json=payload)
        assert res.status_code == 422
