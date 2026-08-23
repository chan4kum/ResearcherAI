"""
evals/evaluators/agent_success.py

Dimension: Agent Success
-------------------------
Operational question: Did the agent complete its task lifecycle correctly?

Checks:
  - AgentState.status == COMPLETED (when must_succeed=True)
  - AgentState.error is None
  - AgentState.plan is non-empty (planner node ran)
  - AgentState.trace contains "planning" (confirms planner was entered)
  - AgentState.answer is non-None and non-empty
  - AgentState.duration_ms > 0 (execution happened)

Score:
  Each of the 6 checks is worth 1/6 of the total.

Note on TaskResponse structure:
  - status, plan, answer, error live at the top level
  - tools_used, trace, duration_ms live inside .metadata
"""

from __future__ import annotations

from evals.dataset.cases import EvalCase, EvalType
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict


def _get(response: object, attr: str, default: object = None) -> object:
    """Get attr from response or response.metadata (TaskResponse nesting)."""
    # Try top level first
    val = getattr(response, attr, None)
    if val is not None:
        return val
    # Try nested metadata (TaskResponse / ExecutionMetadata)
    metadata = getattr(response, "metadata", None)
    if metadata is not None:
        val = getattr(metadata, attr, None)
        if val is not None:
            return val
    return default


class AgentSuccessEvaluator(BaseEvaluator):
    """Evaluates whether the agent completed its task lifecycle correctly."""

    @property
    def dimension(self) -> str:
        return "agent_success"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        if case.eval_type == EvalType.RAG:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="RAG case — agent lifecycle not applicable.",
            )

        aq = case.answer_quality
        checks: list[tuple[str, bool]] = []

        # 1. Status
        status = getattr(response, "status", None)
        status_val = status.value if hasattr(status, "value") else str(status)
        if aq.must_succeed:
            checks.append(("status=COMPLETED", status_val == "completed"))
        else:
            checks.append(("status=non-completed", status_val != "completed"))

        # 2. No error
        error = getattr(response, "error", None)
        if aq.must_succeed:
            checks.append(("no_error", error is None or error == ""))
        else:
            checks.append(("controlled_failure", True))

        # 3. Plan non-empty (planner ran)
        plan = getattr(response, "plan", []) or []
        checks.append(("plan_non_empty", len(plan) > 0))

        # 4. Trace contains planning stage — in TaskResponse this is metadata.trace
        trace = _get(response, "trace", []) or []
        checks.append(("trace_has_planning", "planning" in trace))

        # 5. Answer non-empty (when expected to succeed)
        answer = getattr(response, "answer", None) or ""
        if aq.must_succeed:
            checks.append(("answer_non_empty", bool(answer.strip())))
        else:
            checks.append(("answer_check_skipped", True))

        # 6. Duration >= 0 — in TaskResponse this is metadata.duration_ms
        duration_ms = float(_get(response, "duration_ms", 0.0) or 0.0)
        checks.append(("duration_positive", duration_ms >= 0))

        passed = sum(1 for _, ok in checks if ok)
        total = len(checks)
        score = passed / total

        failures = [name for name, ok in checks if not ok]
        verdict = EvalVerdict.PASS if not failures else EvalVerdict.FAIL
        reason = (
            f"Agent completed all {total} lifecycle checks."
            if not failures
            else f"Agent lifecycle failures: {'; '.join(failures)}"
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(score, 4),
            reason=reason,
            details={
                "status": status_val,
                "error": error,
                "plan_steps": len(plan),
                "trace": trace,
                "duration_ms": duration_ms,
                "checks": {name: ok for name, ok in checks},
            },
        )
