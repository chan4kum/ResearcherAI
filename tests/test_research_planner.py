import pytest
from app.config import Settings
from app.main import app
from app.services.rag.research import (
    MultiStepResearchPlanner,
    ResearchPlan,
    ResearchSubquestionType,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_tsmc_intel_complex_research_plan(settings: Settings) -> None:
    """Verify complex multi-entity semiconductor inquiry generates structured subquestions."""
    planner = MultiStepResearchPlanner(settings=settings)
    query = (
        "Compare TSMC and Intel manufacturing strategies, identify their major technology "
        "challenges, and assess implications for the semiconductor industry."
    )
    plan = await planner.create_plan(query)

    assert isinstance(plan, ResearchPlan)
    assert plan.original_query == query
    assert len(plan.subquestions) == 6
    assert plan.estimated_complexity in ("high", "complex")

    # 1. TSMC Strategy
    sq1 = plan.subquestions[0]
    assert sq1.index == 1
    assert "TSMC" in sq1.target_entities
    assert sq1.subquestion_type == ResearchSubquestionType.STRATEGY
    assert len(sq1.depends_on) == 0

    # 2. Intel Strategy
    sq2 = plan.subquestions[1]
    assert sq2.index == 2
    assert "Intel" in sq2.target_entities
    assert sq2.subquestion_type == ResearchSubquestionType.STRATEGY
    assert len(sq2.depends_on) == 0

    # 3. TSMC Challenges
    sq3 = plan.subquestions[2]
    assert sq3.index == 3
    assert "TSMC" in sq3.target_entities
    assert sq3.subquestion_type == ResearchSubquestionType.CHALLENGE

    # 4. Intel Challenges
    sq4 = plan.subquestions[3]
    assert sq4.index == 4
    assert "Intel" in sq4.target_entities
    assert sq4.subquestion_type == ResearchSubquestionType.CHALLENGE

    # 5. Industry Implications
    sq5 = plan.subquestions[4]
    assert sq5.index == 5
    assert sq5.subquestion_type == ResearchSubquestionType.IMPLICATION

    # 6. Comparison
    sq6 = plan.subquestions[5]
    assert sq6.index == 6
    assert sq6.subquestion_type == ResearchSubquestionType.COMPARISON
    assert "subq_1" in sq6.depends_on
    assert "subq_2" in sq6.depends_on


@pytest.mark.asyncio
async def test_boeing_airbus_comparison_research_plan(settings: Settings) -> None:
    """Verify dual-entity comparison generates entity questions and synthesis."""
    planner = MultiStepResearchPlanner(settings=settings)
    query = "Compare Boeing and Airbus supply chain resilience and delivery backlogs."
    plan = await planner.create_plan(query)

    assert len(plan.subquestions) >= 3
    subq_types = [sq.subquestion_type for sq in plan.subquestions]
    assert ResearchSubquestionType.COMPARISON in subq_types

    comparison_sq = next(
        sq for sq in plan.subquestions if sq.subquestion_type == ResearchSubquestionType.COMPARISON
    )
    assert len(comparison_sq.depends_on) >= 2


@pytest.mark.asyncio
async def test_single_entity_research_plan(settings: Settings) -> None:
    """Verify factual query creates concise single subquestion plan."""
    planner = MultiStepResearchPlanner(settings=settings)
    query = "What is the battery chemistry of the Tesla 4680 cell?"
    plan = await planner.create_plan(query)

    assert len(plan.subquestions) >= 1
    assert plan.subquestions[0].index == 1
    has_target = (
        "Tesla 4680" in plan.subquestions[0].question
        or "Tesla" in str(plan.subquestions[0].target_entities)
    )
    assert has_target


@pytest.mark.asyncio
async def test_dependency_dag_integrity(settings: Settings) -> None:
    """Verify all subquestion dependencies reference existing preceding IDs."""
    planner = MultiStepResearchPlanner(settings=settings)
    query = (
        "Compare TSMC and Intel manufacturing strategies, identify their major technology "
        "challenges, and assess implications for the semiconductor industry."
    )
    plan = await planner.create_plan(query)

    all_ids = {sq.id for sq in plan.subquestions}
    for sq in plan.subquestions:
        for dep in sq.depends_on:
            assert dep in all_ids, f"Dependency {dep} in {sq.id} is not in plan subquestions"


@pytest.mark.asyncio
async def test_create_research_plan_endpoint() -> None:
    """Verify POST /api/v1/rag/research/plan HTTP 200 response."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "query": (
                "Compare TSMC and Intel manufacturing strategies, identify their major "
                "technology challenges, and assess implications for the semiconductor industry."
            )
        }
        res = await client.post("/api/v1/rag/research/plan", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "plan_id" in data
        assert "overall_goal" in data
        assert "subquestions" in data
        assert len(data["subquestions"]) == 6


@pytest.mark.asyncio
async def test_create_research_plan_endpoint_validation_error() -> None:
    """Verify empty query returns 422 Unprocessable Entity."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        res = await client.post("/api/v1/rag/research/plan", json={"query": ""})
        assert res.status_code == 422
