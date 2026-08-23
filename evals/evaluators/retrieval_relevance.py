"""
evals/evaluators/retrieval_relevance.py

Dimension: Retrieval Relevance
-------------------------------
Operational question: Are the retrieved chunks related to the user's question?

Scoring logic (deterministic, no LLM judge required):
  - Check that the number of returned citations meets the minimum threshold.
  - Check that the average similarity score meets the minimum threshold.
  - Verify each citation has a non-empty source and content field.
  - Verify the retrieval strategy matches the expected strategy (if specified).

Score:
  = (citations_ok * 0.4) + (similarity_ok * 0.3) + (schema_ok * 0.2) + (strategy_ok * 0.1)
"""

from __future__ import annotations

from evals.dataset.cases import EvalCase, EvalType
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict


class RetrievalRelevanceEvaluator(BaseEvaluator):
    """Evaluates whether retrieved citations are plentiful, similar, and well-formed."""

    @property
    def dimension(self) -> str:
        return "retrieval_relevance"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        if case.eval_type != EvalType.RAG:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="Not a RAG case — retrieval relevance not applicable.",
            )

        exp = case.retrieval_expectation
        if exp is None:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="No retrieval expectation defined for this case.",
            )

        # Expect a RAGResponse-like object
        citations = getattr(response, "citations", []) or []
        strategy = getattr(response, "strategy", None)

        # 1. Citation count check
        count_ok = len(citations) >= exp.min_citations
        count_score = 1.0 if count_ok else max(0.0, len(citations) / max(exp.min_citations, 1))

        # 2. Average similarity check
        avg_sim = (
            sum(c.similarity for c in citations) / len(citations) if citations else 0.0
        )
        sim_ok = avg_sim >= exp.min_avg_similarity
        sim_score = 1.0 if sim_ok else (avg_sim / max(exp.min_avg_similarity, 0.001))

        # 3. Citation schema correctness (source and chunk_id non-empty)
        schema_errors = 0
        for c in citations:
            if not getattr(c, "source", ""):
                schema_errors += 1
            if not getattr(c, "chunk_id", ""):
                schema_errors += 1
        schema_score = 1.0 if not citations or schema_errors == 0 else max(
            0.0, 1 - (schema_errors / (len(citations) * 2))
        )

        # 4. Strategy match
        strategy_ok = exp.strategy is None or strategy == exp.strategy
        strategy_score = 1.0 if strategy_ok else 0.0

        total_score = (
            count_score * 0.4
            + sim_score * 0.3
            + schema_score * 0.2
            + strategy_score * 0.1
        )

        failures = []
        if not count_ok:
            failures.append(
                f"citations={len(citations)} < min={exp.min_citations}"
            )
        if not sim_ok:
            failures.append(
                f"avg_similarity={avg_sim:.3f} < min={exp.min_avg_similarity:.3f}"
            )
        if schema_errors:
            failures.append(f"{schema_errors} citation schema errors")
        if not strategy_ok:
            failures.append(f"strategy={strategy!r} != expected={exp.strategy!r}")

        verdict = EvalVerdict.PASS if not failures else EvalVerdict.FAIL
        reason = (
            "All retrieval checks passed."
            if not failures
            else f"Retrieval failures: {'; '.join(failures)}"
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(total_score, 4),
            reason=reason,
            details={
                "citations_count": len(citations),
                "avg_similarity": round(avg_sim, 4),
                "strategy": strategy,
                "schema_errors": schema_errors,
            },
        )
