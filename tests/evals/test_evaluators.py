"""
tests/evals/test_evaluators.py — Unit tests for all six evaluators

All tests are fully deterministic (no LLM, no network) using mock response
objects and pre-configured EvalCases.
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any

from evals.dataset.cases import (
    AnswerQualityExpectation,
    EvalCase,
    EvalType,
    GroundednessLevel,
    RetrievalExpectation,
    ToolExpectation,
)
from evals.evaluators import (
    AgentSuccessEvaluator,
    AnswerQualityEvaluator,
    CitationCorrectnessEvaluator,
    GroundednessEvaluator,
    RetrievalRelevanceEvaluator,
    ToolSelectionEvaluator,
)
from evals.evaluators.base import EvalVerdict


# ---------------------------------------------------------------------------
# Mock response objects
# ---------------------------------------------------------------------------


@dataclass
class MockCitation:
    chunk_id: str = "chunk-1"
    doc_id: str = "doc-1"
    source: str = "platform_docs.md"
    file_type: str = "md"
    chunk_index: int = 0
    content: str = "The agentic platform processes user tasks via LLM and tools."
    similarity: float = 0.85
    document_type: str | None = None
    department: str | None = None
    date: str | None = None
    author: str | None = None
    tags: list = field(default_factory=list)
    initial_rank: int | None = None
    rerank_score: float | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class MockRAGResponse:
    question: str = "What is the platform?"
    answer: str = "The agentic platform processes user tasks via LLM and tools."
    citations: list = field(default_factory=list)
    retrieved_chunks_count: int = 1
    model: str = "mock"
    provider: str = "mock"
    strategy: str = "normal"
    error: str | None = None


@dataclass
class MockAgentState:
    status: Any = None
    answer: str = "The answer to your question is 4."
    error: str | None = None
    plan: list = field(default_factory=lambda: ["step 1", "step 2"])
    trace: list = field(default_factory=lambda: ["initialized", "planning", "planned", "completed"])
    tools_used: list = field(default_factory=list)
    duration_ms: float = 123.4
    task: str = "test task"
    task_id: str = "test-id"

    def __post_init__(self):
        if self.status is None:
            from app.models.schemas import TaskStatus
            self.status = TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# Helper to build minimal EvalCases
# ---------------------------------------------------------------------------


def _rag_case(
    case_id: str = "TEST-RAG",
    groundedness: GroundednessLevel = GroundednessLevel.HIGH,
    retrieval_exp: RetrievalExpectation | None = None,
    answer_quality: AnswerQualityExpectation | None = None,
) -> EvalCase:
    return EvalCase(
        id=case_id,
        name="test_rag",
        eval_type=EvalType.RAG,
        description="test",
        task_or_question="What is the platform?",
        retrieval_expectation=retrieval_exp or RetrievalExpectation(min_citations=1),
        answer_quality=answer_quality or AnswerQualityExpectation(must_succeed=True),
        groundedness=groundedness,
    )


def _agent_case(
    case_id: str = "TEST-AGENT",
    tool_exp: ToolExpectation | None = None,
    answer_quality: AnswerQualityExpectation | None = None,
) -> EvalCase:
    return EvalCase(
        id=case_id,
        name="test_agent",
        eval_type=EvalType.AGENT,
        description="test",
        task_or_question="What is 2 + 2?",
        tool_expectation=tool_exp or ToolExpectation(expected_tool=None),
        answer_quality=answer_quality or AnswerQualityExpectation(must_succeed=True),
        groundedness=GroundednessLevel.LOW,
    )


# ===========================================================================
# 1. Retrieval Relevance Evaluator
# ===========================================================================


class TestRetrievalRelevanceEvaluator:
    ev = RetrievalRelevanceEvaluator()

    def test_skip_for_agent_case(self):
        result = self.ev.evaluate(_agent_case(), MockAgentState())
        assert result.verdict == EvalVerdict.SKIP

    def test_pass_with_sufficient_citations(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=1))
        resp = MockRAGResponse(citations=[MockCitation()])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.PASS
        assert result.score > 0.5

    def test_fail_when_insufficient_citations(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=5))
        resp = MockRAGResponse(citations=[MockCitation()])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_when_strategy_mismatch(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=0, strategy="hyde"))
        resp = MockRAGResponse(strategy="normal")
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_pass_with_zero_min_citations(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=0))
        resp = MockRAGResponse(citations=[])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.PASS

    def test_schema_score_penalises_missing_source(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=0))
        bad_citation = MockCitation(source="")  # empty source
        resp = MockRAGResponse(citations=[bad_citation])
        result = self.ev.evaluate(case, resp)
        # schema score < 1.0 → total score < 1.0
        assert result.score < 1.0


# ===========================================================================
# 2. Citation Correctness Evaluator
# ===========================================================================


class TestCitationCorrectnessEvaluator:
    ev = CitationCorrectnessEvaluator()

    def test_skip_for_agent_case(self):
        result = self.ev.evaluate(_agent_case(), MockAgentState())
        assert result.verdict == EvalVerdict.SKIP

    def test_pass_with_well_formed_citation(self):
        case = _rag_case()
        resp = MockRAGResponse(citations=[MockCitation()])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.PASS
        assert result.score >= 0.6

    def test_fail_with_empty_chunk_id(self):
        case = _rag_case()
        bad = MockCitation(chunk_id="")
        resp = MockRAGResponse(citations=[bad])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_with_out_of_range_similarity(self):
        case = _rag_case()
        bad = MockCitation(similarity=1.5)
        resp = MockRAGResponse(citations=[bad])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_expected_source_missing_reduces_score(self):
        exp = RetrievalExpectation(min_citations=0, expected_sources=["important_doc.pdf"])
        case = _rag_case(retrieval_exp=exp)
        resp = MockRAGResponse(citations=[MockCitation(source="other_doc.md")])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL
        assert result.score < 1.0

    def test_pass_with_no_citations_no_expectation(self):
        case = _rag_case(retrieval_exp=RetrievalExpectation(min_citations=0))
        resp = MockRAGResponse(citations=[])
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.PASS


# ===========================================================================
# 3. Groundedness Evaluator
# ===========================================================================


class TestGroundednessEvaluator:
    ev = GroundednessEvaluator()

    def test_skip_for_low_groundedness(self):
        case = _rag_case(groundedness=GroundednessLevel.LOW)
        result = self.ev.evaluate(case, MockRAGResponse())
        assert result.verdict == EvalVerdict.SKIP

    def test_pass_with_good_context_overlap(self):
        case = _rag_case(groundedness=GroundednessLevel.HIGH)
        citation = MockCitation(
            content="The agentic platform processes user tasks via LLM and tools."
        )
        resp = MockRAGResponse(
            answer="The agentic platform processes user tasks via LLM and tools.",
            citations=[citation],
        )
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.PASS

    def test_fail_on_hallucination_signal(self):
        case = _rag_case(groundedness=GroundednessLevel.MEDIUM)
        resp = MockRAGResponse(
            answer="I don't know what this platform does.",
            citations=[MockCitation()],
        )
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_on_exception_disqualifier(self):
        case = _rag_case(groundedness=GroundednessLevel.MEDIUM)
        resp = MockRAGResponse(answer="traceback from exception in module xyz")
        result = self.ev.evaluate(case, resp)
        assert result.verdict == EvalVerdict.FAIL

    def test_pass_with_no_citations_medium(self):
        """No citations = no context to overlap; should not hard-fail."""
        case = _rag_case(groundedness=GroundednessLevel.MEDIUM)
        resp = MockRAGResponse(answer="A useful answer about the platform.", citations=[])
        result = self.ev.evaluate(case, resp)
        # overlap_score = 1.0 when no citations (cannot penalise)
        assert result.score >= 0.5


# ===========================================================================
# 4. Answer Quality Evaluator
# ===========================================================================


class TestAnswerQualityEvaluator:
    ev = AnswerQualityEvaluator()

    def test_pass_with_all_criteria_met(self):
        case = _agent_case(
            answer_quality=AnswerQualityExpectation(
                must_contain_any=["answer", "result"],
                must_not_contain=["error", "traceback"],
                min_word_count=3,
                max_word_count=100,
                must_succeed=True,
            )
        )
        state = MockAgentState(answer="The answer is 4 and the result is correct.")
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.PASS
        assert result.score == 1.0

    def test_fail_keyword_not_present(self):
        case = _agent_case(
            answer_quality=AnswerQualityExpectation(
                must_contain_any=["kubernetes", "helm"],
                must_succeed=True,
            )
        )
        state = MockAgentState(answer="The answer is 4.")
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_forbidden_string_present(self):
        case = _agent_case(
            answer_quality=AnswerQualityExpectation(
                must_not_contain=["traceback"],
                must_succeed=True,
            )
        )
        state = MockAgentState(answer="Traceback (most recent call last)...")
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_word_count_too_low(self):
        case = _agent_case(
            answer_quality=AnswerQualityExpectation(
                min_word_count=20,
                must_succeed=True,
            )
        )
        state = MockAgentState(answer="Short answer.")
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_on_error_when_must_succeed(self):
        case = _agent_case(
            answer_quality=AnswerQualityExpectation(must_succeed=True)
        )
        state = MockAgentState(error="Something went wrong", answer="")
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL


# ===========================================================================
# 5. Agent Success Evaluator
# ===========================================================================


class TestAgentSuccessEvaluator:
    ev = AgentSuccessEvaluator()

    def test_skip_for_rag_case(self):
        result = self.ev.evaluate(_rag_case(), MockRAGResponse())
        assert result.verdict == EvalVerdict.SKIP

    def test_pass_for_completed_agent(self):
        case = _agent_case()
        state = MockAgentState()
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.PASS
        assert result.score >= 0.8

    def test_fail_when_status_is_failed(self):
        from app.models.schemas import TaskStatus

        case = _agent_case()
        state = MockAgentState(
            status=TaskStatus.FAILED,
            error="Something went wrong",
            answer=None,
        )
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_when_plan_is_empty(self):
        case = _agent_case()
        state = MockAgentState(plan=[])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_when_trace_missing_planning(self):
        case = _agent_case()
        state = MockAgentState(trace=["initialized", "completed"])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_score_degrades_proportionally(self):
        from app.models.schemas import TaskStatus

        case = _agent_case()
        state = MockAgentState(plan=[], trace=["initialized"], status=TaskStatus.FAILED)
        result = self.ev.evaluate(case, state)
        # Multiple checks fail → score well below 1.0
        assert result.score < 0.7


# ===========================================================================
# 6. Tool Selection Evaluator
# ===========================================================================


class TestToolSelectionEvaluator:
    ev = ToolSelectionEvaluator()

    def test_skip_for_rag_case(self):
        result = self.ev.evaluate(_rag_case(), MockRAGResponse())
        assert result.verdict == EvalVerdict.SKIP

    def test_skip_when_no_tool_expectation(self):
        # tool_expectation must be None at the EvalCase level (not just expected_tool=None)
        case = EvalCase(
            id="TEST-SKIP",
            name="skip_test",
            eval_type=EvalType.AGENT,
            description="test",
            task_or_question="dummy",
            tool_expectation=None,  # evaluator should SKIP
            groundedness=GroundednessLevel.LOW,
        )
        result = self.ev.evaluate(case, MockAgentState())
        assert result.verdict == EvalVerdict.SKIP

    def test_pass_correct_tool_selected(self):
        case = _agent_case(
            tool_exp=ToolExpectation(expected_tool="calculator")
        )
        state = MockAgentState(tools_used=["calculator"])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.PASS
        assert result.score >= 0.6

    def test_fail_expected_tool_not_used(self):
        case = _agent_case(
            tool_exp=ToolExpectation(expected_tool="calculator")
        )
        state = MockAgentState(tools_used=[])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_pass_no_tool_expected_and_none_used(self):
        case = _agent_case(
            tool_exp=ToolExpectation(expected_tool=None)
        )
        state = MockAgentState(tools_used=[])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.PASS

    def test_fail_no_tool_expected_but_tool_used(self):
        case = _agent_case(
            tool_exp=ToolExpectation(expected_tool=None)
        )
        state = MockAgentState(tools_used=["calculator"])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_fail_forbidden_tool_used(self):
        case = _agent_case(
            tool_exp=ToolExpectation(
                expected_tool=None,
                must_not_use_tool=["calculator"],
            )
        )
        state = MockAgentState(tools_used=["calculator"])
        result = self.ev.evaluate(case, state)
        assert result.verdict == EvalVerdict.FAIL

    def test_partial_score_correct_tool_but_forbidden_also_used(self):
        case = _agent_case(
            tool_exp=ToolExpectation(
                expected_tool="calculator",
                must_not_use_tool=["app_info"],
            )
        )
        state = MockAgentState(tools_used=["calculator", "app_info"])
        result = self.ev.evaluate(case, state)
        # correct tool (0.6) but forbidden used (0.4 penalty) → 0.6
        assert result.verdict == EvalVerdict.FAIL
        assert result.score == pytest.approx(0.6, abs=0.01)
