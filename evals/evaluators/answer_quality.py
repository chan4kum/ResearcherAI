"""
evals/evaluators/answer_quality.py

Dimension: Answer Quality
--------------------------
Operational question: Is the answer useful, complete, and free of errors?

Scoring logic (deterministic):
  - must_contain_any: at least one expected keyword must appear in the answer
  - must_not_contain: none of the forbidden strings may appear
  - word count: answer must be between min_word_count and max_word_count
  - execution success: if must_succeed=True, response must not have an error field

Score:
  = (keyword_ok * 0.35) + (not_contain_ok * 0.25) + (word_count_ok * 0.2) + (success_ok * 0.2)
"""

from __future__ import annotations

from evals.dataset.cases import EvalCase
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict


class AnswerQualityEvaluator(BaseEvaluator):
    """Evaluates answer completeness, relevance keywords, and structural quality."""

    @property
    def dimension(self) -> str:
        return "answer_quality"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        aq = case.answer_quality
        answer: str = getattr(response, "answer", "") or ""
        error: str | None = getattr(response, "error", None)
        answer_lower = answer.lower()

        failures: list[str] = []

        # ── Execution success check ───────────────────────────────────────
        if aq.must_succeed:
            success_ok = error is None or error == ""
            success_score = 1.0 if success_ok else 0.0
            if not success_ok:
                failures.append(f"execution failed with error: {str(error)[:80]}")
        else:
            # Case expects failure — success if there IS an error or answer is controlled
            success_ok = True
            success_score = 1.0

        # ── Keyword presence (must_contain_any) ───────────────────────────
        if aq.must_contain_any:
            hit = any(kw.lower() in answer_lower for kw in aq.must_contain_any)
            keyword_score = 1.0 if hit else 0.0
            if not hit:
                failures.append(
                    f"none of {aq.must_contain_any} found in answer"
                )
        else:
            hit = True
            keyword_score = 1.0

        # ── Forbidden strings (must_not_contain) ─────────────────────────
        forbidden_found = [s for s in aq.must_not_contain if s.lower() in answer_lower]
        not_contain_score = 1.0 if not forbidden_found else 0.0
        if forbidden_found:
            failures.append(f"forbidden strings in answer: {forbidden_found}")

        # ── Word count check ──────────────────────────────────────────────
        word_count = len(answer.split()) if answer else 0
        wc_ok = aq.min_word_count <= word_count <= aq.max_word_count
        word_count_score = 1.0 if wc_ok else 0.0
        if not wc_ok:
            failures.append(
                f"word count={word_count} not in [{aq.min_word_count}, {aq.max_word_count}]"
            )

        total_score = (
            keyword_score * 0.35
            + not_contain_score * 0.25
            + word_count_score * 0.20
            + success_score * 0.20
        )

        verdict = EvalVerdict.PASS if not failures else EvalVerdict.FAIL
        reason = (
            "Answer meets all quality criteria."
            if not failures
            else f"Quality failures: {'; '.join(failures)}"
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(total_score, 4),
            reason=reason,
            details={
                "word_count": word_count,
                "keyword_hit": hit,
                "forbidden_found": forbidden_found,
                "error": error,
            },
        )
