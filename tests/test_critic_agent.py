from unittest.mock import AsyncMock, patch

import pytest
from app.config import Settings
from app.main import app
from app.services.llm import LLMResponse
from app.services.rag.critic import (
    CriticAgent,
    CriticIssueType,
    SelfCorrectionEngine,
    SelfCorrectionResult,
)
from app.services.rag.models import Citation
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_critic_detects_unsupported_claims(settings: Settings) -> None:
    """Verify critic detects hallucinated or unsupported claims not in evidence."""
    critic = CriticAgent(settings=settings)
    question = "What is TSMC's manufacturing roadmap?"
    evidence = ["TSMC is ramping 2nm N2 GAA nodes in Hsinchu with high volume in 2025."]
    draft_answer = (
        "TSMC is ramping 2nm N2 GAA nodes in 2025. Additionally, Steve Jobs founded TSMC "
        "and announced secret classified technology in the year 2099."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
    )

    assert not evaluation.is_acceptable
    assert evaluation.critique_score < 1.0
    issue_types = [i.issue_type for i in evaluation.issues]
    assert CriticIssueType.UNSUPPORTED_CLAIM in issue_types
    assert evaluation.action_recommended == "revise_answer"


@pytest.mark.asyncio
async def test_critic_detects_contradictions(settings: Settings) -> None:
    """Verify critic detects direct internal contradictions."""
    critic = CriticAgent(settings=settings)
    question = "What is TSMC's market position?"
    evidence = ["TSMC holds approximately 60% of worldwide foundry manufacturing share."]
    draft_answer = (
        "TSMC is leading the global foundry market worldwide. However, TSMC has zero market share "
        "and produces no semiconductor wafers."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
    )

    assert not evaluation.is_acceptable
    issue_types = [i.issue_type for i in evaluation.issues]
    assert CriticIssueType.CONTRADICTION in issue_types


@pytest.mark.asyncio
async def test_critic_detects_missing_evidence(settings: Settings) -> None:
    """Verify critic detects omissions of required comparative dimensions."""
    critic = CriticAgent(settings=settings)
    question = "Compare TSMC and Intel 2nm semiconductor manufacturing strategies."
    evidence = [
        "TSMC N2 relies on GAA nanosheet architecture.",
        "Intel 18A utilizes RibbonFET GAA and PowerVia backside power delivery.",
    ]
    draft_answer = (
        "TSMC N2 relies on GAA nanosheet architecture for power efficiency and high density."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
    )

    assert not evaluation.is_acceptable
    issue_types = [i.issue_type for i in evaluation.issues]
    assert CriticIssueType.MISSING_EVIDENCE in issue_types


@pytest.mark.asyncio
async def test_critic_detects_irrelevant_information(settings: Settings) -> None:
    """Verify critic detects off-topic or tangential information."""
    critic = CriticAgent(settings=settings)
    question = "What is TSMC's manufacturing strategy?"
    evidence = ["TSMC is constructing advanced semiconductor fabrication facilities in Arizona."]
    draft_answer = (
        "TSMC is constructing advanced fabs in Arizona. Here is a recipe for chocolate cake: "
        "mix flour, cocoa, eggs, and bake for 30 minutes at 350 degrees."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
    )

    assert not evaluation.is_acceptable
    issue_types = [i.issue_type for i in evaluation.issues]
    assert CriticIssueType.IRRELEVANT_INFORMATION in issue_types


@pytest.mark.asyncio
async def test_critic_detects_citation_problems(settings: Settings) -> None:
    """Verify critic detects fabricated chunk citations."""
    critic = CriticAgent(settings=settings)
    question = "What node is TSMC launching?"
    evidence = ["TSMC N2 enters volume production in late 2025."]
    citations = [
        Citation(
            chunk_id="chunk_valid_1",
            doc_id="doc_1",
            source="tsmc.pdf",
            file_type="pdf",
            chunk_index=0,
            content="TSMC N2 enters volume production in late 2025.",
            similarity=0.95,
        )
    ]
    draft_answer = (
        "TSMC N2 enters volume production in late 2025 [chunk_99999]."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
        citations=citations,
    )

    assert not evaluation.is_acceptable
    issue_types = [i.issue_type for i in evaluation.issues]
    assert CriticIssueType.CITATION_PROBLEM in issue_types


@pytest.mark.asyncio
async def test_critic_approves_grounded_answer(settings: Settings) -> None:
    """Verify clean, well-grounded answer receives full approval."""
    critic = CriticAgent(settings=settings)
    question = "What is TSMC's Arizona fab progress?"
    evidence = ["TSMC Arizona Fab 21 is completing equipment move-in for 4nm production in 2025."]
    draft_answer = (
        "TSMC Arizona Fab 21 is completing equipment move-in for 4nm mass production in 2025."
    )

    evaluation = await critic.evaluate(
        question=question,
        evidence=evidence,
        draft_answer=draft_answer,
    )

    assert evaluation.is_acceptable
    assert evaluation.critique_score == 1.0
    assert len(evaluation.issues) == 0
    assert evaluation.action_recommended == "accept"


@pytest.mark.asyncio
async def test_self_correction_multi_iteration_loop(settings: Settings) -> None:
    """Verify self-correction engine refines flawed answer and tracks iterations."""
    engine = SelfCorrectionEngine(settings=settings)
    question = "What is TSMC's manufacturing strategy?"
    evidence = ["TSMC Arizona Fab 21 is completing equipment move-in for 4nm production in 2025."]
    flawed_draft = (
        "TSMC is constructing fabs in Arizona. Here is a recipe for chocolate cake."
    )

    result = await engine.correct_answer(
        question=question,
        evidence=evidence,
        draft_answer=flawed_draft,
        max_corrections=2,
    )

    assert isinstance(result, SelfCorrectionResult)
    assert result.is_corrected
    assert len(result.attempts) >= 1
    assert result.iterations <= 2
    assert result.final_answer != flawed_draft


@pytest.mark.asyncio
async def test_self_correction_strict_loop_termination(settings: Settings) -> None:
    """Verify self-correction strictly terminates at max_corrections even if critic rejects."""
    engine = SelfCorrectionEngine(settings=settings)
    question = "Compare TSMC and Intel strategies."
    evidence = ["TSMC and Intel produce semiconductors."]
    flawed_draft = "Unverified rumor: Steve Jobs founded TSMC in 2099."

    # Force LLM generation to always return another flawed string
    with patch.object(
        engine._llm_service,
        "generate",
        new=AsyncMock(
            return_value=LLMResponse(
                content="Still contains unverified rumor: Steve Jobs founded TSMC in 2099.",
                model="mock",
                provider="mock",
            )
        ),
    ):
        result = await engine.correct_answer(
            question=question,
            evidence=evidence,
            draft_answer=flawed_draft,
            max_corrections=2,
        )

        assert result.iterations == 2
        assert result.max_iterations == 2


@pytest.mark.asyncio
async def test_critic_and_self_correction_endpoints() -> None:
    """Verify REST API endpoints for critic evaluation and self-correction."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        # 1. Critique endpoint
        critique_payload = {
            "question": "What is TSMC roadmap?",
            "draft_answer": "TSMC 2nm N2 enters production in 2025.",
            "evidence": ["TSMC 2nm N2 enters production in 2025."],
        }
        res = await client.post("/api/v1/rag/critic/evaluate", json=critique_payload)
        assert res.status_code == 200
        data = res.json()
        assert "is_acceptable" in data
        assert "critique_score" in data
        assert "issues" in data

        # 2. Self-correction endpoint
        correct_payload = {
            "question": "What is TSMC roadmap?",
            "draft_answer": "TSMC 2nm N2 enters production in 2025. Recipe for chocolate cake.",
            "evidence": ["TSMC 2nm N2 enters production in 2025."],
            "max_corrections": 2,
        }
        res = await client.post("/api/v1/rag/critic/correct", json=correct_payload)
        assert res.status_code == 200
        data = res.json()
        assert "final_answer" in data
        assert "iterations" in data
        assert "attempts" in data
        assert data["iterations"] <= 2
