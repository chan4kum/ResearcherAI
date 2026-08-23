"""
tests/evals/test_regression.py — Unit tests for LLM regression testing and CI quality gate
"""

import pytest

from evals.regression import (
    check_paths_trigger_eval,
    compare_with_baseline,
    load_baseline_report,
)
from evals.runner import EvalSuiteReport
from evals.thresholds import QualityThresholds


# ---------------------------------------------------------------------------
# Path Trigger Detection Tests
# ---------------------------------------------------------------------------


class TestPathTriggerDetection:
    def test_matches_prompt_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/agent/prompts/planner.py",
            "app/services/rag/service.py",
        ])
        assert matched is True
        assert len(triggers) >= 1

    def test_matches_model_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/llm/factory.py",
            "app/services/llm/openai.py",
        ])
        assert matched is True
        assert len(triggers) == 2

    def test_matches_retriever_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/rag/retriever.py",
            "app/services/rag/bm25.py",
        ])
        assert matched is True
        assert len(triggers) == 2

    def test_matches_agent_logic_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/agent/graph/nodes.py",
            "app/services/agent/service.py",
        ])
        assert matched is True
        assert len(triggers) == 2

    def test_matches_routing_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/rag/router.py",
            "app/services/rag/adaptive.py",
        ])
        assert matched is True
        assert len(triggers) == 2

    def test_matches_hyde_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/rag/hyde.py",
        ])
        assert matched is True
        assert len(triggers) == 1

    def test_matches_query_rewriting_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "app/services/rag/rewriter.py",
            "app/services/rag/analyzer.py",
        ])
        assert matched is True
        assert len(triggers) == 2

    def test_ignores_unrelated_documentation_or_helm_changes(self):
        matched, triggers = check_paths_trigger_eval([
            "README.md",
            "helm/agentic-platform/Chart.yaml",
            "docker-compose.yml",
        ])
        assert matched is False
        assert len(triggers) == 0


# ---------------------------------------------------------------------------
# Baseline Comparison & Threshold Tests
# ---------------------------------------------------------------------------


class TestBaselineComparison:
    @pytest.fixture
    def baseline_data(self):
        return {
            "overall_score": 0.95,
            "per_dimension_scores": {
                "retrieval_relevance": 0.90,
                "citation_correctness": 0.95,
                "groundedness": 0.92,
                "answer_quality": 0.90,
                "agent_success": 1.00,
                "tool_selection": 0.95,
            },
        }

    def _make_report(self, overall: float, per_dim: dict[str, float]) -> EvalSuiteReport:
        return EvalSuiteReport(
            timestamp="2026-08-23T18:00:00Z",
            total_cases=10,
            passed=10,
            failed=0,
            skipped_dimensions=0,
            overall_score=overall,
            per_dimension_scores=per_dim,
            cases=[],
        )

    def test_passes_when_scores_meet_thresholds_and_baseline(self, baseline_data):
        current = self._make_report(
            overall=0.96,
            per_dim={
                "retrieval_relevance": 0.92,
                "citation_correctness": 0.95,
                "groundedness": 0.92,
                "answer_quality": 0.90,
                "agent_success": 1.00,
                "tool_selection": 0.95,
            },
        )
        report = compare_with_baseline(current, baseline_data)
        assert report.passed is True
        assert len(report.threshold_failures) == 0
        assert len(report.regression_failures) == 0
        assert report.overall_delta == pytest.approx(0.01)

    def test_fails_when_overall_score_below_minimum_threshold(self, baseline_data):
        # 0.80 is below min_overall_score 0.85
        current = self._make_report(
            overall=0.80,
            per_dim={
                "retrieval_relevance": 0.80,
                "citation_correctness": 0.85,
                "groundedness": 0.80,
                "answer_quality": 0.75,
                "agent_success": 0.85,
                "tool_selection": 0.80,
            },
        )
        report = compare_with_baseline(current, baseline_data)
        assert report.passed is False
        assert any("below minimum threshold" in f for f in report.threshold_failures)

    def test_fails_when_overall_drop_exceeds_max_allowed(self, baseline_data):
        # Baseline is 0.95, current is 0.88 -> delta -0.07 (exceeds max drop 0.05)
        # Even though 0.88 > min_overall_score (0.85), regression drop trips the gate
        current = self._make_report(
            overall=0.88,
            per_dim={
                "retrieval_relevance": 0.85,
                "citation_correctness": 0.90,
                "groundedness": 0.85,
                "answer_quality": 0.85,
                "agent_success": 0.90,
                "tool_selection": 0.85,
            },
        )
        thresholds = QualityThresholds(min_overall_score=0.80, max_allowed_drop=0.05)
        report = compare_with_baseline(current, baseline_data, thresholds)
        assert report.passed is False
        assert any("exceeds maximum allowed drop" in f for f in report.regression_failures)

    def test_fails_when_individual_dimension_below_threshold(self, baseline_data):
        current = self._make_report(
            overall=0.90,
            per_dim={
                "retrieval_relevance": 0.50,  # Below min 0.80
                "citation_correctness": 0.95,
                "groundedness": 0.92,
                "answer_quality": 0.90,
                "agent_success": 1.00,
                "tool_selection": 0.95,
            },
        )
        report = compare_with_baseline(current, baseline_data)
        assert report.passed is False
        assert any("retrieval_relevance" in f for f in report.threshold_failures)

    def test_generates_markdown_table(self, baseline_data):
        current = self._make_report(
            overall=0.95,
            per_dim={
                "retrieval_relevance": 0.90,
                "citation_correctness": 0.95,
                "groundedness": 0.92,
                "answer_quality": 0.90,
                "agent_success": 1.00,
                "tool_selection": 0.95,
            },
        )
        report = compare_with_baseline(current, baseline_data)
        md = report.to_markdown()
        assert "# LLM Quality & Regression Evaluation Report" in md
        assert "| Dimension | Current | Baseline | Delta | Threshold | Status |" in md
        assert "retrieval_relevance" in md
        assert "✅ PASS" in md


# ---------------------------------------------------------------------------
# Golden Baseline File Test
# ---------------------------------------------------------------------------


def test_load_golden_baseline_file():
    data = load_baseline_report()
    assert "overall_score" in data
    assert "per_dimension_scores" in data
    assert float(data["overall_score"]) >= 0.85
