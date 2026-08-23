from app.core.logging import get_logger
from app.services.rag.models import Citation
from app.services.rag.research.models import SubquestionExecutionResult

logger = get_logger("app.services.rag.research.store")


class ResearchEvidenceStore:
    """In-memory structured repository for storing and aggregating multi-step research evidence."""

    def __init__(self) -> None:
        self._results: dict[str, SubquestionExecutionResult] = {}

    def add_result(self, result: SubquestionExecutionResult) -> None:
        """Store or update execution outcome for a subquestion."""
        self._results[result.subquestion_id] = result
        logger.debug(
            "research_evidence_stored",
            subquestion_id=result.subquestion_id,
            evidence_count=len(result.evidence),
            citations_count=len(result.citations),
        )

    def get_result(self, subquestion_id: str) -> SubquestionExecutionResult | None:
        """Retrieve execution outcome by subquestion ID."""
        return self._results.get(subquestion_id)

    def get_all_results(self) -> list[SubquestionExecutionResult]:
        """Return all recorded subquestion results in chronological index order."""
        return sorted(self._results.values(), key=lambda r: r.index)

    def get_all_evidence(self) -> list[str]:
        """Collect all retrieved evidence snippets across subquestions."""
        snippets: list[str] = []
        for res in self.get_all_results():
            snippets.extend(res.evidence)
        return snippets

    def get_all_citations(self) -> list[Citation]:
        """Collect and deduplicate all citations across subquestions."""
        seen_keys: set[tuple[str, str]] = set()
        deduped: list[Citation] = []

        for res in self.get_all_results():
            for cit in res.citations:
                key = (cit.source, cit.chunk_id)
                if key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(cit)

        return deduped

    def get_intermediate_answers(self) -> list[tuple[str, str]]:
        """Return list of (subquestion_query, intermediate_sub_answer) pairs."""
        return [
            (res.query, res.sub_answer)
            for res in self.get_all_results()
            if res.sub_answer.strip()
        ]

    @property
    def total_evidence_items(self) -> int:
        """Total count of evidence snippets across all subquestions."""
        return len(self.get_all_evidence())

    def format_synthesis_context(self) -> str:
        """Format aggregated findings into structured markdown context for final LLM synthesis."""
        sections: list[str] = []
        for res in self.get_all_results():
            src_str = ", ".join(res.sources) or "none"
            sections.append(
                f"### Subquestion {res.index}: {res.query}\n"
                f"**Status**: {res.status.value} | **Sources**: {src_str}\n\n"
                f"**Key Findings**:\n{res.sub_answer or 'No intermediate answer generated.'}\n\n"
                f"**Evidence Snippets**:\n"
                + (
                    "\n".join(f"- {snip}" for snip in res.evidence)
                    if res.evidence
                    else "- (No direct textual evidence retrieved)"
                )
            )
        return "\n\n---\n\n".join(sections)

    def clear(self) -> None:
        """Clear all stored evidence."""
        self._results.clear()
