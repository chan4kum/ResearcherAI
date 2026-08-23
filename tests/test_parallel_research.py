import asyncio
from unittest.mock import patch

import pytest
from app.config import Settings
from app.main import app
from app.services.rag.research import (
    MultiStepResearchExecutor,
    MultiStepResearchPlanner,
    ParallelResearchConfig,
    ResearchExecutionResult,
    SubquestionExecutionStatus,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_parallel_research_topological_waves(settings: Settings) -> None:
    """Verify topological wave partitioning separates independent and dependent subquestions."""
    planner = MultiStepResearchPlanner(settings=settings)
    executor = MultiStepResearchExecutor(planner=planner, settings=settings)

    query = (
        "Compare TSMC and Intel manufacturing strategies, identify their major technology "
        "challenges, and assess implications for the semiconductor industry."
    )
    plan = await planner.create_plan(query)
    waves = executor._compute_execution_waves(plan.subquestions)

    assert len(waves) == 3
    # Wave 0 has the 4 independent subquestions
    wave0_ids = [sq.id for sq in waves[0]]
    assert set(wave0_ids) == {"subq_1", "subq_2", "subq_3", "subq_4"}

    # Wave 1 has subq_5
    assert [sq.id for sq in waves[1]] == ["subq_5"]

    # Wave 2 has subq_6
    assert [sq.id for sq in waves[2]] == ["subq_6"]


@pytest.mark.asyncio
async def test_parallel_research_execution_flow(settings: Settings) -> None:
    """Verify full parallel research execution completes all subquestions and final synthesis."""
    executor = MultiStepResearchExecutor(settings=settings)
    query = (
        "Compare TSMC and Intel manufacturing strategies, identify their major technology "
        "challenges, and assess implications for the semiconductor industry."
    )
    config = ParallelResearchConfig(
        max_concurrency=4,
        subquestion_timeout_seconds=5.0,
        max_retries=2,
    )
    result = await executor.execute_research(
        query=query,
        top_k_per_source=2,
        mode="parallel",
        config=config,
    )

    assert isinstance(result, ResearchExecutionResult)
    assert result.status == "completed"
    assert len(result.subquestion_results) == 6
    for sq in result.subquestion_results:
        assert sq.status == SubquestionExecutionStatus.COMPLETED
        assert len(sq.sub_answer) > 0
    assert len(result.final_synthesis) > 50


@pytest.mark.asyncio
async def test_concurrency_limit_enforced(settings: Settings) -> None:
    """Verify that concurrency limiter never exceeds max_concurrency."""
    executor = MultiStepResearchExecutor(settings=settings)
    max_concurrency = 2
    active_concurrency = 0
    peak_concurrency = 0

    original_execute = executor._execute_single_subquestion

    async def tracked_execute(*args, **kwargs):
        nonlocal active_concurrency, peak_concurrency
        active_concurrency += 1
        if active_concurrency > peak_concurrency:
            peak_concurrency = active_concurrency
        await asyncio.sleep(0.05)
        res = await original_execute(*args, **kwargs)
        active_concurrency -= 1
        return res

    with patch.object(executor, "_execute_single_subquestion", side_effect=tracked_execute):
        config = ParallelResearchConfig(max_concurrency=max_concurrency)
        query = (
            "Compare TSMC and Intel manufacturing strategies, identify their major technology "
            "challenges, and assess implications for the semiconductor industry."
        )
        result = await executor.execute_research(query=query, mode="parallel", config=config)
        assert result.status == "completed"
        assert peak_concurrency <= max_concurrency


@pytest.mark.asyncio
async def test_timeout_handling_and_failure_isolation(settings: Settings) -> None:
    """Verify timeout on one subquestion fails gracefully without terminating entire research."""
    executor = MultiStepResearchExecutor(settings=settings)
    original_execute = executor._execute_single_subquestion

    async def slow_execute(subquestion, *args, **kwargs):
        if subquestion.id == "subq_3":
            await asyncio.sleep(1.0)
        return await original_execute(subquestion, *args, **kwargs)

    with patch.object(executor, "_execute_single_subquestion", side_effect=slow_execute):
        config = ParallelResearchConfig(
            max_concurrency=4,
            subquestion_timeout_seconds=0.1,
            max_retries=1,
            retry_delay_seconds=0.01,
        )
        query = (
            "Compare TSMC and Intel manufacturing strategies, identify their major technology "
            "challenges, and assess implications for the semiconductor industry."
        )
        result = await executor.execute_research(query=query, mode="parallel", config=config)

        # Research must complete with 'partial' status due to subq_3 timeout
        assert result.status == "partial"
        subq3_res = next(r for r in result.subquestion_results if r.subquestion_id == "subq_3")
        assert subq3_res.status == SubquestionExecutionStatus.FAILED
        assert "Timed out" in str(subq3_res.error)

        # Other subquestions must have succeeded
        succeeded = [
            r for r in result.subquestion_results
            if r.status == SubquestionExecutionStatus.COMPLETED
        ]
        assert len(succeeded) == 5

        # Final synthesis must still be generated from remaining evidence
        assert len(result.final_synthesis) > 50


@pytest.mark.asyncio
async def test_retry_mechanism_on_transient_failure(settings: Settings) -> None:
    """Verify transient subquestion failure recovers on second attempt."""
    executor = MultiStepResearchExecutor(settings=settings)
    original_execute = executor._execute_single_subquestion
    subq2_attempts = 0

    async def flaky_execute(subquestion, *args, **kwargs):
        nonlocal subq2_attempts
        if subquestion.id == "subq_2":
            subq2_attempts += 1
            if subq2_attempts == 1:
                raise RuntimeError("Transient connection reset")
        return await original_execute(subquestion, *args, **kwargs)

    with patch.object(executor, "_execute_single_subquestion", side_effect=flaky_execute):
        config = ParallelResearchConfig(
            max_concurrency=4,
            subquestion_timeout_seconds=2.0,
            max_retries=2,
            retry_delay_seconds=0.01,
        )
        query = (
            "Compare TSMC and Intel manufacturing strategies, identify their major technology "
            "challenges, and assess implications for the semiconductor industry."
        )
        result = await executor.execute_research(query=query, mode="parallel", config=config)

        assert result.status == "completed"
        assert subq2_attempts == 2
        subq2_res = next(r for r in result.subquestion_results if r.subquestion_id == "subq_2")
        assert subq2_res.status == SubquestionExecutionStatus.COMPLETED


@pytest.mark.asyncio
async def test_parallel_research_endpoint() -> None:
    """Verify POST /api/v1/rag/research/execute with parallel mode parameters."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "query": (
                "Compare TSMC and Intel manufacturing strategies, identify their major "
                "technology challenges, and assess implications for the semiconductor industry."
            ),
            "top_k_per_source": 2,
            "mode": "parallel",
            "max_concurrency": 4,
            "timeout_seconds": 5.0,
            "max_retries": 2,
        }
        res = await client.post("/api/v1/rag/research/execute", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "research_id" in data
        assert "final_synthesis" in data
        assert data["status"] in ("completed", "partial")
        assert len(data["subquestion_results"]) == 6
