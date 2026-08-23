import json
from pathlib import Path

from app.core.logging import get_logger
from app.services.document.models import Document, DocumentMetadata

logger = get_logger("app.services.document.store")


class DocumentStore:
    """In-memory and filesystem-backed storage for documents with checksum deduplication."""

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._storage_path = Path(storage_dir) if storage_dir else None
        self._documents: dict[str, Document] = {}
        self._checksum_to_id: dict[str, str] = {}

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def _load_from_disk(self) -> None:
        """Hydrate in-memory store from persisted JSON files on disk."""
        if not self._storage_path or not self._storage_path.exists():
            return

        for file in self._storage_path.glob("*.json"):
            if file.name == "vectors.json":
                continue
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                doc = Document.model_validate(data)
                self._documents[doc.doc_id] = doc
                self._checksum_to_id[doc.metadata.checksum] = doc.doc_id
            except Exception as exc:
                logger.warning("failed_to_load_persisted_document", file=str(file), error=str(exc))

    def _persist_doc(self, doc: Document) -> None:
        """Persist a single document representation to disk."""
        if not self._storage_path:
            return
        file_path = self._storage_path / f"{doc.doc_id}.json"
        try:
            file_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("failed_to_persist_document", doc_id=doc.doc_id, error=str(exc))

    def _delete_persisted_doc(self, doc_id: str) -> None:
        """Remove document file from disk."""
        if not self._storage_path:
            return
        file_path = self._storage_path / f"{doc_id}.json"
        if file_path.exists():
            try:
                file_path.unlink()
            except Exception as exc:
                logger.error("failed_to_delete_persisted_document", doc_id=doc_id, error=str(exc))

    def has_checksum(self, checksum: str) -> bool:
        """Check if a document with the exact same content checksum is already stored."""
        return checksum in self._checksum_to_id

    def get_by_checksum(self, checksum: str) -> Document | None:
        """Retrieve existing document matching the given checksum."""
        doc_id = self._checksum_to_id.get(checksum)
        if doc_id:
            return self._documents.get(doc_id)
        return None

    def get(self, doc_id: str) -> Document | None:
        """Retrieve document by its unique ID."""
        return self._documents.get(doc_id)

    def add(self, document: Document) -> None:
        """Store document in memory and persist to disk."""
        self._documents[document.doc_id] = document
        self._checksum_to_id[document.metadata.checksum] = document.doc_id
        self._persist_doc(document)
        logger.info(
            "document_stored",
            doc_id=document.doc_id,
            source=document.metadata.source,
            checksum=document.metadata.checksum[:12],
            words=document.metadata.word_count,
        )

    def delete(self, doc_id: str) -> bool:
        """Delete document from in-memory index and disk."""
        doc = self._documents.pop(doc_id, None)
        if doc:
            self._checksum_to_id.pop(doc.metadata.checksum, None)
            self._delete_persisted_doc(doc_id)
            logger.info("document_deleted", doc_id=doc_id)
            return True
        return False

    def list_documents(self) -> list[DocumentMetadata]:
        """List metadata for all ingested documents."""
        return [doc.metadata for doc in self._documents.values()]

    def count(self) -> int:
        """Return total count of ingested documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Clear all in-memory and persisted documents."""
        doc_ids = list(self._documents.keys())
        for doc_id in doc_ids:
            self.delete(doc_id)
        self._documents.clear()
        self._checksum_to_id.clear()
