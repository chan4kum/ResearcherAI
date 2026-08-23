import json
from pathlib import Path

from app.core.logging import get_logger
from app.services.rag.agentic_retrieval.models import AgenticRetrievalTrace

logger = get_logger("app.services.rag.agentic.trace_store")


class RetrievalTraceStore:
    """Persistent and in-memory store for Agentic Retrieval telemetry traces."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._traces: dict[str, AgenticRetrievalTrace] = {}
        self._storage_dir = Path(storage_dir) if storage_dir else Path("data/retrieval_traces")
        try:
            self._storage_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def save_trace(self, trace: AgenticRetrievalTrace) -> None:
        """Persist trace in-memory and to disk."""
        self._traces[trace.session_id] = trace
        try:
            file_path = self._storage_dir / f"{trace.session_id}.json"
            file_path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(
                "failed_to_write_trace_to_disk",
                session_id=trace.session_id,
                error=str(exc),
            )

        logger.info(
            "retrieval_trace_saved",
            session_id=trace.session_id,
            steps_count=len(trace.steps),
            termination_reason=trace.termination_reason,
        )

    def get_trace(self, session_id: str) -> AgenticRetrievalTrace | None:
        """Retrieve trace by its session ID."""
        if session_id in self._traces:
            return self._traces[session_id]

        file_path = self._storage_dir / f"{session_id}.json"
        if file_path.exists():
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                trace = AgenticRetrievalTrace(**data)
                self._traces[session_id] = trace
                return trace
            except Exception as exc:
                logger.warning("failed_to_read_trace_file", session_id=session_id, error=str(exc))
        return None

    def list_traces(self, limit: int = 50) -> list[AgenticRetrievalTrace]:
        """List recently recorded traces."""
        return list(self._traces.values())[-limit:]
