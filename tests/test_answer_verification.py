import pytest
from app.config import Settings
from app.main import app
from app.services.rag.models import Citation
from app.services.rag.verification import (
    AnswerVerifier,
    ClaimSupportStatus,
    FactualClaim,
    VerificationReport,
)
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_supported_claim(settings: Settings) -> None:
    """Verify that a claim backed by evidence is correctly marked as SUPPORTED."""
    verifier = AnswerVerifier(settings=settings)
    claim = "TSMC is constructing advanced semiconductor fabs in Phoenix, Arizona."
    evidence = [
        "[tsmc_annual.pdf] TSMC is constructing advanced semiconductor fabs in Phoenix, Arizona."
    ]

    result = await verifier.verify_claim(claim_text=claim, evidence=evidence)

    assert isinstance(result, FactualClaim)
    assert result.support_status == ClaimSupportStatus.SUPPORTED
    assert result.confidence >= 0.8
    assert result.evidence_text is not None
    assert "Phoenix, Arizona" in result.evidence_text
    assert result.source == "tsmc_annual.pdf"


@pytest.mark.asyncio
async def test_unsupported_claim(settings: Settings) -> None:
    """Verify that an unsubstantiated hallucination is marked as UNSUPPORTED."""
    verifier = AnswerVerifier(settings=settings)
    claim = "Steve Jobs founded TSMC in the year 2099 based on an unverified rumor."
    evidence = [
        "[tsmc_doc.pdf] TSMC was founded in 1987 in Hsinchu Science Park, Taiwan."
    ]

    result = await verifier.verify_claim(claim_text=claim, evidence=evidence)

    assert isinstance(result, FactualClaim)
    assert result.support_status == ClaimSupportStatus.UNSUPPORTED
    assert result.confidence <= 0.2
    assert result.evidence_text is None


@pytest.mark.asyncio
async def test_contradictory_evidence(settings: Settings) -> None:
    """Verify that an assertion conflicting with verified data is marked as CONTRADICTED."""
    verifier = AnswerVerifier(settings=settings)
    claim = "TSMC has zero market share worldwide and is not manufacturing wafers."
    evidence = [
        "[market_share.pdf] TSMC holds 60% of the worldwide foundry manufacturing market share."
    ]

    result = await verifier.verify_claim(claim_text=claim, evidence=evidence)

    assert isinstance(result, FactualClaim)
    assert result.support_status == ClaimSupportStatus.CONTRADICTED
    assert result.confidence >= 0.8
    assert result.evidence_text is not None
    assert "60%" in result.evidence_text


@pytest.mark.asyncio
async def test_missing_or_broken_citation(settings: Settings) -> None:
    """Verify that a claim referencing a non-existent citation chunk is marked as UNSUPPORTED."""
    verifier = AnswerVerifier(settings=settings)
    claim = "TSMC 2nm GAA technology enters volume production in 2025 [chunk_missing_999]."
    citations = [
        Citation(
            chunk_id="chunk_valid_1",
            doc_id="doc_1",
            source="roadmap.pdf",
            file_type="pdf",
            chunk_index=0,
            content="TSMC 2nm GAA technology enters volume production in 2025.",
            similarity=0.95,
        )
    ]

    result = await verifier.verify_claim(claim_text=claim, evidence=citations)

    assert isinstance(result, FactualClaim)
    assert result.support_status == ClaimSupportStatus.UNSUPPORTED
    assert result.citation_chunk_id == "chunk_missing_999"
    assert "not found" in result.reason


@pytest.mark.asyncio
async def test_partially_supported_claim(settings: Settings) -> None:
    """Verify that a claim with estimated or qualified details is PARTIALLY_SUPPORTED."""
    verifier = AnswerVerifier(settings=settings)
    claim = "TSMC 2nm N2 mass production is estimated around 2026."
    evidence = [
        "[tsmc_pr.txt] TSMC 2nm N2 process enters mass production in late 2025."
    ]

    result = await verifier.verify_claim(claim_text=claim, evidence=evidence)

    assert isinstance(result, FactualClaim)
    assert result.support_status == ClaimSupportStatus.PARTIALLY_SUPPORTED
    assert result.confidence >= 0.5


@pytest.mark.asyncio
async def test_full_answer_verification_report(settings: Settings) -> None:
    """Verify end-to-end answer verification and sanitized verified answer synthesis."""
    verifier = AnswerVerifier(settings=settings)
    question = "What is TSMC's manufacturing status and roadmap?"
    evidence = [
        "[tsmc_fact.pdf] TSMC is constructing advanced semiconductor fabs in Phoenix, Arizona. "
        "TSMC holds 60% of the worldwide foundry manufacturing market share."
    ]
    answer = (
        "TSMC is constructing advanced semiconductor fabs in Phoenix, Arizona. "
        "Steve Jobs founded TSMC in the year 2099 based on an unverified rumor. "
        "TSMC has zero market share worldwide and is not manufacturing wafers."
    )

    report = await verifier.verify_answer(
        question=question,
        answer=answer,
        evidence=evidence,
    )

    assert isinstance(report, VerificationReport)
    assert report.total_claims == 3
    assert report.supported_count == 1
    assert report.unsupported_count == 1
    assert report.contradicted_count == 1
    assert not report.is_verified
    assert report.verified_ratio < 1.0

    # The verified answer must explicitly flag/sanitize unsupported and contradicted claims
    assert "Unverified Claim:" in report.verified_answer
    assert "Refuted Assertion:" in report.verified_answer
    assert "constructing advanced semiconductor fabs in Phoenix" in report.verified_answer


@pytest.mark.asyncio
async def test_verify_answer_endpoint() -> None:
    """Verify POST /api/v1/rag/verify REST API endpoint."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        payload = {
            "question": "What is TSMC's roadmap?",
            "answer": "TSMC is constructing advanced semiconductor fabs in Phoenix, Arizona.",
            "evidence": [
                "[tsmc_doc.pdf] TSMC is constructing advanced semiconductor fabs in Phoenix."
            ],
        }
        res = await client.post("/api/v1/rag/verify", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "report_id" in data
        assert data["is_verified"] is True
        assert data["supported_count"] == 1
        assert data["unsupported_count"] == 0
        assert len(data["claims"]) == 1
        assert data["claims"][0]["support_status"] == "SUPPORTED"
