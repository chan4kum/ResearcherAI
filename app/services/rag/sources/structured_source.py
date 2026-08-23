import json
from typing import Any

from app.core.logging import get_logger
from app.services.document.models import MetadataFilter
from app.services.rag.bm25 import tokenize
from app.services.rag.models import Citation
from app.services.rag.sources.base import BaseRetrievalSource
from app.services.rag.sources.models import SourceResult, SourceType

logger = get_logger("app.services.rag.sources.structured")

DEFAULT_STRUCTURED_RECORDS: list[dict[str, Any]] = [
    {
        "table": "aircraft_maintenance_log",
        "primary_key": "LOG-2026-777X-001",
        "aircraft_model": "Boeing 777-9",
        "tail_number": "N779XX",
        "component": "Wing Flutter Damper",
        "inspection_status": "COMPLETED",
        "finding": "Ultrasonic probe identified micro-fissure on hydraulic actuator mount.",
        "corrective_action": (
            "Actuator bracket replaced with reinforced titanium alloy SB-2026-X99."
        ),
        "certifying_inspector": "ID-98214",
        "date": "2026-03-01",
    },
    {
        "table": "aircraft_maintenance_log",
        "primary_key": "LOG-2026-A321-044",
        "aircraft_model": "Airbus A321-271NX",
        "tail_number": "F-WXAB",
        "component": "PW1133G Turbofan Engine",
        "inspection_status": "PENDING_PARTS",
        "finding": "High-pressure turbine blade thermal barrier coating wear exceeding threshold.",
        "corrective_action": "Engine awaiting replacement stage-1 turbine assembly from supplier.",
        "certifying_inspector": "ID-44109",
        "date": "2026-02-14",
    },
    {
        "table": "supplier_quality_audits",
        "primary_key": "AUDIT-2026-SUP-088",
        "supplier_name": "Precision Titanium Forgings Corp",
        "supplied_components": "Hydraulic fittings, flutter damper pins",
        "audit_score": 94.5,
        "conformance_status": "APPROVED_WITH_CONDITIONS",
        "audit_summary": "All titanium bar stock lots certified to AMS 4928 standard.",
        "date": "2026-01-20",
    },
]


class StructuredDatabasePlaceholderSource(BaseRetrievalSource):
    """Placeholder retrieval source simulating relational SQL and tabular records."""

    def __init__(
        self,
        source_name: str = "relational_db_source",
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        self._source_name = source_name
        self._records = records or list(DEFAULT_STRUCTURED_RECORDS)

    @property
    def source_type(self) -> SourceType:
        return SourceType.STRUCTURED_DB

    @property
    def source_name(self) -> str:
        return self._source_name

    def add_record(self, record: dict[str, Any]) -> None:
        """Insert a mock structured row for isolated testing."""
        self._records.append(record)

    async def search(
        self,
        query: str,
        top_k: int = 5,
        filters: MetadataFilter | dict[str, Any] | None = None,
        min_relevance: float = 0.0,
    ) -> list[SourceResult]:
        """Query tabular records and return serialized SQL record rows."""
        clean_query = query.strip()
        if not clean_query:
            return []

        query_tokens = set(tokenize(clean_query.lower()))
        scored_records: list[tuple[dict[str, Any], float]] = []

        for rec in self._records:
            # Flatten record values into a single search text
            row_text = " ".join(str(v) for v in rec.values()).lower()
            row_tokens = set(tokenize(row_text))

            overlap = len(query_tokens.intersection(row_tokens))
            if overlap > 0 or not query_tokens:
                relevance = round(overlap / (len(query_tokens) or 1), 4)
                if relevance >= min_relevance:
                    scored_records.append((rec, relevance))

        scored_records.sort(key=lambda x: x[1], reverse=True)
        top_records = scored_records[:top_k]

        results: list[SourceResult] = []
        for idx, (rec, score) in enumerate(top_records, start=1):
            table = rec.get("table", "database_table")
            pk = rec.get("primary_key", f"row_{idx}")

            # Format record as structured text block
            formatted_lines = [f"Table: {table} | Record: {pk}"]
            for k, v in rec.items():
                if k not in ("table", "primary_key"):
                    formatted_lines.append(f"  {k}: {v}")
            content_str = "\n".join(formatted_lines)

            citation = Citation(
                chunk_id=f"db_{table}_{pk}",
                doc_id=f"table_{table}",
                source=f"sql://{table}/{pk}",
                file_type="json",
                chunk_index=idx - 1,
                content=content_str,
                similarity=score,
                metadata={
                    "table": table,
                    "primary_key": pk,
                    "raw_record": json.dumps(rec),
                },
            )

            result = SourceResult(
                source=self.source_name,
                source_type=self.source_type,
                content=content_str,
                relevance=score,
                metadata={
                    "table": table,
                    "primary_key": pk,
                    "raw_record": rec,
                },
                citation=citation,
            )
            results.append(result)

        logger.info(
            "structured_source_search_completed",
            source=self.source_name,
            query=clean_query[:80],
            results_found=len(results),
        )
        return results
