from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.core.guardrails.document_safety import DocumentSafetyValidator
from app.core.logging import get_logger
from app.db.repository import BaseVectorRepository, create_vector_repository
from app.services.document.chunker import DocumentChunker
from app.services.document.loaders.base import compute_checksum
from app.services.document.loaders.factory import DocumentLoaderFactory
from app.services.document.models import (
    ChunkingConfig,
    Document,
    DocumentChunk,
    DocumentMetadata,
    EmbeddedChunk,
    IngestResult,
    IngestStatus,
)
from app.services.document.store import DocumentStore
from app.services.embedding.service import EmbeddingService

logger = get_logger("app.services.document.service")

SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".md", ".json", ".csv", ".log", ".yaml", ".yml"}


class DocumentService:
    """Domain service managing document loading, extraction, chunking, and vector storage."""

    def __init__(
        self,
        store: DocumentStore | None = None,
        chunker: DocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_repository: BaseVectorRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = store or DocumentStore(storage_dir=self._settings.storage_dir)
        default_chunk_config = ChunkingConfig(
            chunk_size=self._settings.default_chunk_size,
            chunk_overlap=self._settings.default_chunk_overlap,
        )
        self._chunker = chunker or DocumentChunker(default_config=default_chunk_config)
        self._embedding_service = embedding_service or EmbeddingService(settings=self._settings)
        self._vector_repository = vector_repository or create_vector_repository(self._settings)

    @property
    def store(self) -> DocumentStore:
        return self._store

    @property
    def chunker(self) -> DocumentChunker:
        return self._chunker

    @property
    def embedding_service(self) -> EmbeddingService:
        return self._embedding_service

    @property
    def vector_repository(self) -> BaseVectorRepository:
        return self._vector_repository

    def ingest_bytes(
        self,
        content_bytes: bytes,
        source_name: str,
        custom_metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest raw document bytes, skipping duplicates based on SHA-256 checksum."""
        checksum = compute_checksum(content_bytes)

        # Deduplication check: Do not re-ingest duplicate content
        if self._store.has_checksum(checksum):
            existing = self._store.get_by_checksum(checksum)
            assert existing is not None
            logger.info(
                "document_ingestion_skipped_duplicate",
                doc_id=existing.doc_id,
                source=source_name,
                checksum=checksum[:12],
            )
            return IngestResult(
                doc_id=existing.doc_id,
                status=IngestStatus.SKIPPED_DUPLICATE,
                source=source_name,
                checksum=checksum,
                metadata=existing.metadata,
                message=(
                    "Duplicate document detected. Ingestion skipped to prevent "
                    "redundant processing."
                ),
            )

        # Document Safety & Integrity Validation
        safety_validator = DocumentSafetyValidator()
        safety_check = safety_validator.validate_content(content_bytes, filename=source_name)
        if not safety_check.is_valid:
            logger.warning("document_safety_rejected", source=source_name, error=safety_check.error)
            return IngestResult(
                doc_id="",
                status=IngestStatus.FAILED,
                source=source_name,
                checksum=checksum,
                metadata=None,
                message=f"Security validation failed: {safety_check.error}",
            )

        try:
            doc = DocumentLoaderFactory.load_bytes(
                content_bytes=content_bytes,
                source_name=source_name,
                custom_metadata=custom_metadata,
            )
            self._store.add(doc)
            return IngestResult(
                doc_id=doc.doc_id,
                status=IngestStatus.INGESTED,
                source=source_name,
                checksum=checksum,
                metadata=doc.metadata,
                message="Document successfully ingested and extracted.",
            )
        except Exception as exc:
            logger.error("document_ingestion_failed", source=source_name, error=str(exc))
            return IngestResult(
                doc_id="",
                status=IngestStatus.FAILED,
                source=source_name,
                checksum=checksum,
                metadata=None,
                message=f"Extraction failed: {exc}",
            )

    def ingest_file(
        self,
        file_path: str | Path,
        custom_metadata: dict[str, Any] | None = None,
    ) -> IngestResult:
        """Ingest a single local document file from disk."""
        path = Path(file_path)
        if not path.is_file():
            return IngestResult(
                doc_id="",
                status=IngestStatus.FAILED,
                source=str(file_path),
                checksum="",
                metadata=None,
                message=f"File not found on disk: {file_path}",
            )

        content_bytes = path.read_bytes()
        return self.ingest_bytes(
            content_bytes=content_bytes,
            source_name=path.name,
            custom_metadata=custom_metadata,
        )

    def ingest_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
    ) -> list[IngestResult]:
        """Incrementally ingest all supported documents in a directory.

        Only newly added files will be ingested; existing files are identified via
        checksum and marked as SKIPPED_DUPLICATE.
        """
        path = Path(dir_path)
        if not path.is_dir():
            logger.error("ingest_directory_invalid_path", path=str(dir_path))
            return []

        pattern = "**/*" if recursive else "*"
        results: list[IngestResult] = []

        for file_path in sorted(path.glob(pattern)):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                result = self.ingest_file(file_path)
                results.append(result)

        logger.info(
            "directory_ingestion_completed",
            directory=str(dir_path),
            total_files=len(results),
            ingested=sum(1 for r in results if r.status == IngestStatus.INGESTED),
            skipped=sum(1 for r in results if r.status == IngestStatus.SKIPPED_DUPLICATE),
        )
        return results

    async def sync_knowledge_base(
        self,
        kb_dir: str | Path | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> dict[str, Any]:
        """Scan KB folder, incrementally ingest, chunk, embed, and index into vector DB."""
        target_dir = Path(kb_dir or self._settings.knowledge_base_dir)
        if not target_dir.is_dir():
            logger.error("kb_directory_not_found", path=str(target_dir))
            return {
                "kb_dir": str(target_dir),
                "status": "error",
                "total_files": 0,
                "ingested": 0,
                "skipped": 0,
                "indexed_documents": 0,
                "total_indexed_chunks": 0,
            }

        ingest_results = self.ingest_directory(target_dir, recursive=True)
        total_indexed_chunks = 0
        indexed_docs = 0

        for r in ingest_results:
            if r.doc_id:
                embedded_chunks = await self.embed_and_index_document(
                    doc_id=r.doc_id,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
                total_indexed_chunks += len(embedded_chunks)
                indexed_docs += 1

        logger.info(
            "kb_sync_completed",
            kb_dir=str(target_dir),
            total_files=len(ingest_results),
            ingested=sum(1 for r in ingest_results if r.status == IngestStatus.INGESTED),
            skipped=sum(1 for r in ingest_results if r.status == IngestStatus.SKIPPED_DUPLICATE),
            indexed_documents=indexed_docs,
            total_indexed_chunks=total_indexed_chunks,
        )

        return {
            "kb_dir": str(target_dir),
            "status": "success",
            "total_files": len(ingest_results),
            "ingested": sum(1 for r in ingest_results if r.status == IngestStatus.INGESTED),
            "skipped": sum(1 for r in ingest_results if r.status == IngestStatus.SKIPPED_DUPLICATE),
            "indexed_documents": indexed_docs,
            "total_indexed_chunks": total_indexed_chunks,
        }

    def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a specific document by its ID."""
        return self._store.get(doc_id)

    def list_documents(self) -> list[DocumentMetadata]:
        """List metadata for all ingested documents."""
        return self._store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        """Delete an ingested document by its ID."""
        return self._store.delete(doc_id)

    def chunk_document(
        self,
        doc_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[DocumentChunk]:
        """Retrieve an ingested document by ID and partition into DocumentChunk objects."""
        doc = self.get_document(doc_id)
        if not doc:
            raise KeyError(f"Document with ID '{doc_id}' not found in store")

        config = None
        if chunk_size is not None or chunk_overlap is not None:
            resolved_size = chunk_size or self._settings.default_chunk_size
            resolved_overlap = (
                chunk_overlap if chunk_overlap is not None else self._settings.default_chunk_overlap
            )
            config = ChunkingConfig(chunk_size=resolved_size, chunk_overlap=resolved_overlap)
        return self._chunker.chunk_document(doc, config=config)

    def chunk_text(
        self,
        text: str,
        source_name: str = "direct_input.txt",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
        """Chunk raw text directly without persisting the parent document."""
        config = None
        if chunk_size is not None or chunk_overlap is not None:
            resolved_size = chunk_size or self._settings.default_chunk_size
            resolved_overlap = (
                chunk_overlap if chunk_overlap is not None else self._settings.default_chunk_overlap
            )
            config = ChunkingConfig(chunk_size=resolved_size, chunk_overlap=resolved_overlap)
        return self._chunker.chunk_text(
            text=text,
            source=source_name,
            custom_metadata=custom_metadata,
            config=config,
        )

    async def embed_document_chunks(
        self,
        doc_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[EmbeddedChunk]:
        """Partition an ingested document into chunks and generate embeddings for every chunk."""
        chunks = self.chunk_document(doc_id, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return await self._embedding_service.embed_chunks(chunks)

    async def embed_and_index_document(
        self,
        doc_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> list[EmbeddedChunk]:
        """Execute Document -> Chunks -> Embeddings -> Vector Database storage."""
        doc = self.get_document(doc_id)
        if not doc:
            raise KeyError(f"Document with ID '{doc_id}' not found in store")

        await self._vector_repository.store_document(doc)
        embedded_chunks = await self.embed_document_chunks(
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        await self._vector_repository.store_chunks(embedded_chunks)
        return embedded_chunks

    async def search_similar_chunks(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: Any = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Perform similarity search over stored vector chunks with optional filtering."""
        query_embedding = await self._embedding_service.embed_text(query)
        return await self._vector_repository.search_similar_chunks(
            query_embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
            filters=filters,
        )



