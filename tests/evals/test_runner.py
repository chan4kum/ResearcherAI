"""
tests/evals/test_runner.py — Integration test for evals runner and report generation
"""

import tempfile
from pathlib import Path

import pytest

from evals.dataset.cases import get_case_by_id
from evals.runner import run_evaluation


@pytest.mark.asyncio
async def test_run_evaluation_subset():
    with tempfile.TemporaryDirectory() as tmpdir:
        cases = [
            get_case_by_id("AGENT-001"),
            get_case_by_id("AGENT-002"),
            get_case_by_id("RAG-001"),
            get_case_by_id("TOOL-001"),
        ]
        cases = [c for c in cases if c is not None]

        report = await run_evaluation(
            cases=cases,
            verbose=False,
            report_dir=tmpdir,
        )

        assert report.total_cases == len(cases)
        assert report.passed == len(cases)
        assert report.failed == 0
        assert report.overall_score >= 0.95

        # Verify report file was written
        report_files = list(Path(tmpdir).glob("eval_report_*.json"))
        assert len(report_files) == 1
        content = report_files[0].read_text()
        assert "overall_score" in content
        assert "cases" in content
