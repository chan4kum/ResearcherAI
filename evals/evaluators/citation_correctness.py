"""
evals/evaluators/citation_correctness.py

Dimension: Citation Correctness
---------------------------------
Operational question: Are the citations structurally valid and traceable?

Every citation in a RAG response must satisfy:
  - chunk_id is a non-empty string
  - source is a non-empty string
  - similarity is a float in [0.0, 1.0]
  - content is a non-empty string (proof the chunk is not empty)
  - doc_id is present (provenance traceability)

Additionally, any expected_sources listed in the case must appear in
the set of returned citation sources.

Score:
  per-citation schema score (0.6 weight) + source presence score (0.4 weight)
"""

from __future__ import annotations

from evals.dataset.cases import EvalCase, EvalType
from evals.evaluators.base import BaseEvaluator, EvalResult, EvalVerdict


class CitationCorrectnessEvaluator(BaseEvaluator):
    """Evaluates structural validity and provenance of RAG citations."""

    @property
    def dimension(self) -> str:
        return "citation_correctness"

    def evaluate(self, case: EvalCase, response: object) -> EvalResult:
        if case.eval_type != EvalType.RAG:
            return EvalResult(
                dimension=self.dimension,
                verdict=EvalVerdict.SKIP,
                score=1.0,
                reason="Not a RAG case — citation correctness not applicable.",
            )

        exp = case.retrieval_expectation
        citations = getattr(response, "citations", []) or []

        # ── Schema validation ─────────────────────────────────────────────
        required_fields = ["chunk_id", "source", "content", "doc_id"]
        schema_issues: list[str] = []
        similarity_issues: list[str] = []

        for i, c in enumerate(citations):
            for field in required_fields:
                val = getattr(c, field, None)
                if not val or (isinstance(val, str) and not val.strip()):
                    schema_issues.append(f"citation[{i}].{field} is empty")

            sim = getattr(c, "similarity", None)
            if sim is None or not (0.0 <= float(sim) <= 1.0):
                similarity_issues.append(
                    f"citation[{i}].similarity={sim!r} is not in [0.0, 1.0]"
                )

        total_checks = len(citations) * (len(required_fields) + 1) if citations else 1
        bad_checks = len(schema_issues) + len(similarity_issues)
        schema_score = 1.0 - (bad_checks / total_checks) if total_checks else 1.0
        schema_score = max(0.0, schema_score)

        # ── Expected sources presence ─────────────────────────────────────
        expected_sources = (exp.expected_sources if exp else []) or []
        if expected_sources:
            returned_sources = {getattr(c, "source", "") for c in citations}
            missing = [s for s in expected_sources if s not in returned_sources]
            source_score = 1.0 - len(missing) / len(expected_sources)
            source_issues = [f"expected source {s!r} not found in citations" for s in missing]
        else:
            source_score = 1.0
            source_issues = []

        total_score = schema_score * 0.6 + source_score * 0.4
        all_issues = schema_issues + similarity_issues + source_issues

        verdict = EvalVerdict.PASS if not all_issues else EvalVerdict.FAIL
        reason = (
            f"All {len(citations)} citations are structurally valid."
            if not all_issues
            else f"Citation issues: {'; '.join(all_issues[:5])}"  # cap display
        )

        return EvalResult(
            dimension=self.dimension,
            verdict=verdict,
            score=round(total_score, 4),
            reason=reason,
            details={
                "citations_count": len(citations),
                "schema_issues": schema_issues,
                "similarity_issues": similarity_issues,
                "missing_expected_sources": source_issues,
            },
        )
