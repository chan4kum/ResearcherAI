"""
evals/evaluators/tool_selection.py

Dimension: Tool Selection
--------------------------
Operational question: Did the agent select the right tool (or correctly abstain)?

Two sub-checks:
  1. Correct tool selected (or None when no tool expected)
  2. Forbidden tools NOT used

Score:
  = (correct_tool * 0.6) + (no_forbidden_tools * 0.4)

Note on TaskResponse structure:
  - tools_used lives inside response.metadata.tools_used
"""

from __future__ import annotations

from evals.dataset.cases import EvalCase, EvalType
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict


def _get_tools_used(response: object) -> list[str]:
    """Extract tools_used from response or response.metadata."""
    # Direct attribute (AgentState / mock)
    val = getattr(response, "tools_used", None)
    if val is not None:
        return list(val)
    # Nested in ExecutionMetadata (TaskResponse)
    metadata = getattr(response, "metadata", None)
    if metadata is not None:
        val = getattr(metadata, "tools_used", None)
        if val is not None:
            return list(val)
    return []


class ToolSelectionEvaluator(BaseEvaluator):
    """Evaluates whether the agent made the correct tool selection decision."""

    @property
    def dimension(self) -> str:
        return "tool_selection"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        if case.eval_type == EvalType.RAG:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="RAG case — tool selection not applicable.",
            )

        te = case.tool_expectation
        if te is None:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="No tool expectation defined for this case.",
            )

        tools_used: list[str] = _get_tools_used(response)
        failures: list[str] = []

        # ── Correct tool check ────────────────────────────────────────────
        if te.expected_tool is None:
            correct_tool_ok = len(tools_used) == 0
            if not correct_tool_ok:
                failures.append(
                    f"expected no tool, but agent used: {tools_used}"
                )
        else:
            correct_tool_ok = te.expected_tool.lower() in [t.lower() for t in tools_used]
            if not correct_tool_ok:
                failures.append(
                    f"expected tool '{te.expected_tool}' not in tools_used={tools_used}"
                )

        correct_tool_score = 1.0 if correct_tool_ok else 0.0

        # ── Forbidden tools check ─────────────────────────────────────────
        used_lower = [t.lower() for t in tools_used]
        forbidden_used = [t for t in te.must_not_use_tool if t.lower() in used_lower]
        no_forbidden_ok = len(forbidden_used) == 0
        no_forbidden_score = 1.0 if no_forbidden_ok else 0.0
        if forbidden_used:
            failures.append(f"forbidden tools used: {forbidden_used}")

        total_score = correct_tool_score * 0.6 + no_forbidden_score * 0.4

        verdict = EvalVerdict.PASS if not failures else EvalVerdict.FAIL
        reason = (
            f"Tool selection correct. tools_used={tools_used}."
            if not failures
            else f"Tool selection failures: {'; '.join(failures)}"
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(total_score, 4),
            reason=reason,
            details={
                "tools_used": tools_used,
                "expected_tool": te.expected_tool,
                "must_not_use": te.must_not_use_tool,
                "forbidden_used": forbidden_used,
            },
        )
