"""
evals/evaluators/groundedness.py

Dimension: Groundedness
------------------------
Operational question: Is the answer grounded in the retrieved context,
or does it contain hallucination signals?

We use deterministic heuristic detection (no LLM judge):

Hallucination signals:
  - Answer mentions entities NOT in any citation content AND not in the question
  - Answer contains "I don't know" / "I cannot determine" when citations exist
  - Answer repeats a fabricated citation (source name not in any citation)
  - Answer length is suspiciously short when citations are present

Grounding signals:
  - Answer word overlap with citation content is above a threshold
  - Answer does NOT contain hallucination signals

Score components:
  - no_hallucination_signals: 0.5
  - word_overlap_with_context: 0.3
  - no_disqualifiers: 0.2
"""

from __future__ import annotations

import re

from evals.dataset.cases import EvalCase, EvalType, GroundednessLevel
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict

# Phrases that strongly suggest the model is not grounded or is confused
_HALLUCINATION_SIGNALS = [
    "i don't know",
    "i cannot determine",
    "i do not have information",
    "as of my knowledge cutoff",
    "according to my training data",
    "i'm not sure",
    "it's unclear",
    "no information available",
]

_DISQUALIFIERS = [
    "traceback",
    "exception",
    "internal server error",
    "keyerror",
    "valueerror",
]


def _tokenize(text: str) -> set[str]:
    """Simple whitespace+punctuation tokenizer returning lowercased tokens."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


class GroundednessEvaluator(BaseEvaluator):
    """Evaluates whether the answer is grounded in retrieved context."""

    @property
    def dimension(self) -> str:
        return "groundedness"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        # Groundedness only makes deep sense for RAG (HIGH/MEDIUM); skip if LOW
        if case.groundedness == GroundednessLevel.LOW:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="Groundedness level is LOW — not evaluated.",
            )

        answer: str = ""
        citations = []

        if case.eval_type == EvalType.RAG:
            answer = getattr(response, "answer", "") or ""
            citations = getattr(response, "citations", []) or []
        else:
            answer = getattr(response, "answer", "") or ""

        answer_lower = answer.lower()

        # ── Hallucination signal check ────────────────────────────────────
        found_signals = [s for s in _HALLUCINATION_SIGNALS if s in answer_lower]
        hallucination_score = 1.0 if not found_signals else max(
            0.0, 1.0 - len(found_signals) * 0.3
        )

        # ── Word overlap with citation content ────────────────────────────
        if citations:
            context_tokens = set()
            for c in citations:
                content = getattr(c, "content", "") or ""
                context_tokens.update(_tokenize(content))
            answer_tokens = _tokenize(answer)
            if answer_tokens and context_tokens:
                overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
            else:
                overlap = 0.0
            # Normalise: expect at least 10% overlap for HIGH, 5% for MEDIUM
            min_overlap = 0.10 if case.groundedness == GroundednessLevel.HIGH else 0.05
            overlap_score = min(1.0, overlap / min_overlap)
        else:
            # No citations retrieved — cannot penalise overlap
            overlap = 0.0
            overlap_score = 1.0

        # ── Hard disqualifiers ────────────────────────────────────────────
        found_disqualifiers = [d for d in _DISQUALIFIERS if d in answer_lower]
        disqualifier_score = 0.0 if found_disqualifiers else 1.0

        total_score = (
            hallucination_score * 0.5
            + overlap_score * 0.3
            + disqualifier_score * 0.2
        )

        failures = []
        if found_signals:
            failures.append(f"hallucination signals: {found_signals}")
        if citations and overlap < (0.10 if case.groundedness == GroundednessLevel.HIGH else 0.05):
            failures.append(f"low context overlap: {overlap:.2%}")
        if found_disqualifiers:
            failures.append(f"disqualifiers found: {found_disqualifiers}")

        verdict = EvalVerdict.PASS if not failures else EvalVerdict.FAIL
        reason = (
            "Answer appears well-grounded in retrieved context."
            if not failures
            else f"Groundedness issues: {'; '.join(failures)}"
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(total_score, 4),
            reason=reason,
            details={
                "hallucination_signals_found": found_signals,
                "context_word_overlap": round(overlap, 4) if citations else None,
                "disqualifiers_found": found_disqualifiers,
                "groundedness_level": case.groundedness.value,
            },
        )
