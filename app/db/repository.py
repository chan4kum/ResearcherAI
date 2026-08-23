import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models import DocumentChunkRecord, DocumentRecord
from app.db.session import DatabaseManager
from app.services.document.models import (
    ChunkMetadata,
    Document,
    DocumentMetadata,
    EmbeddedChunk,
    MetadataFilter,
    normalize_metadata_filter,
)
from app.services.embedding.base import cosine_similarity

logger = get_logger("app.db.repository")


class BaseVectorRepository(ABC):
    """Abstract interface defining document and vector chunk storage and retrieval operations."""

    @abstractmethod
    async def store_document(self, doc: Document) -> None:
        """Store or update parent document record."""
        pass

    @abstractmethod
    async def store_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """Store a batch of embedded document chunks."""
        pass

    @abstractmethod
    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        """Perform vector similarity search returning top-k chunks with optional filtering."""
        pass

    @abstractmethod
    async def list_chunks(
        self,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[EmbeddedChunk]:
        """List all stored chunks matching optional metadata filters."""
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Document | None:
        """Retrieve a stored document by its unique identifier."""
        pass

    @abstractmethod
    async def list_documents(self) -> list[Document]:
        """List all stored documents."""
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its associated vector chunks."""
        pass


class InMemoryVectorRepository(BaseVectorRepository):
    """In-memory vector repository with disk snapshot persistence for local interoperability."""

    def __init__(self, storage_path: Path | str | None = None) -> None:
        self._documents: dict[str, Document] = {}
        self._chunks: dict[str, EmbeddedChunk] = {}
        self._storage_path: Path | None = Path(storage_path) if storage_path else None
        self._last_loaded_mtime: float = 0.0
        self._load_from_disk()

    async def list_chunks(
        self,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[EmbeddedChunk]:
        self._load_from_disk()
        filter_obj = normalize_metadata_filter(filters)
        if filter_obj is None:
            return list(self._chunks.values())
        return [c for c in self._chunks.values() if filter_obj.matches(c.metadata)]

    def _load_from_disk(self) -> None:
        """Load persistent vector index from disk if available."""
        if not self._storage_path or not self._storage_path.is_file():
            return
        try:
            mtime = self._storage_path.stat().st_mtime
            if mtime <= self._last_loaded_mtime and self._chunks:
                return
            with open(self._storage_path, encoding="utf-8") as f:
                data = json.load(f)
            self._documents = {
                k: Document.model_validate(v) for k, v in data.get("documents", {}).items()
            }
            self._chunks = {
                k: EmbeddedChunk.model_validate(v) for k, v in data.get("chunks", {}).items()
            }
            self._last_loaded_mtime = mtime
            logger.info(
                "loaded_vectors_from_disk",
                path=str(self._storage_path),
                documents=len(self._documents),
                chunks=len(self._chunks),
            )
        except Exception as exc:
            logger.warning("failed_to_load_vectors_from_disk", error=str(exc))

    def _save_to_disk(self) -> None:
        """Persist in-memory vector index to disk."""
        if not self._storage_path:
            return
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "documents": {k: v.model_dump(mode="json") for k, v in self._documents.items()},
                "chunks": {k: v.model_dump(mode="json") for k, v in self._chunks.items()},
            }
            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._last_loaded_mtime = self._storage_path.stat().st_mtime
        except Exception as exc:
            logger.warning("failed_to_save_vectors_to_disk", error=str(exc))

    async def store_document(self, doc: Document) -> None:
        self._documents[doc.doc_id] = doc
        self._save_to_disk()

    async def store_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        self._save_to_disk()

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        self._load_from_disk()
        filter_obj = normalize_metadata_filter(filters)
        results: list[tuple[EmbeddedChunk, float]] = []
        for chunk in self._chunks.values():
            if filter_obj is not None and not filter_obj.matches(chunk.metadata):
                continue
            sim = cosine_similarity(chunk.embedding, query_embedding)
            if sim >= min_similarity:
                results.append((chunk, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    async def get_document(self, doc_id: str) -> Document | None:
        self._load_from_disk()
        return self._documents.get(doc_id)

    async def list_documents(self) -> list[Document]:
        self._load_from_disk()
        return list(self._documents.values())

    async def delete_document(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False
        del self._documents[doc_id]
        chunks_to_remove = [
            cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id
        ]
        for cid in chunks_to_remove:
            del self._chunks[cid]
        self._save_to_disk()
        return True


class PgVectorRepository(BaseVectorRepository):
    """Production vector repository utilizing PostgreSQL with pgvector with in-memory fallback."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        fallback_repo: InMemoryVectorRepository | None = None,
    ) -> None:
        self._db_manager = db_manager
        self._fallback_repo = fallback_repo or InMemoryVectorRepository()

    async def store_document(self, doc: Document) -> None:
        try:
            async with self._db_manager.get_session() as session:
                stmt = select(DocumentRecord).where(DocumentRecord.id == doc.doc_id)
                res = await session.execute(stmt)
                record = res.scalar_one_or_none()

                meta_dict = doc.metadata.custom_metadata.copy()
                if doc.metadata.document_type:
                    meta_dict["document_type"] = doc.metadata.document_type
                if doc.metadata.department:
                    meta_dict["department"] = doc.metadata.department
                if doc.metadata.date:
                    meta_dict["date"] = doc.metadata.date
                if doc.metadata.author:
                    meta_dict["author"] = doc.metadata.author
                if doc.metadata.tags:
                    meta_dict["tags"] = doc.metadata.tags

                if record is None:
                    record = DocumentRecord(
                        id=doc.doc_id,
                        source=doc.metadata.source,
                        file_type=doc.metadata.file_type,
                        checksum=doc.metadata.checksum,
                        content=doc.content,
                        character_count=doc.metadata.character_count,
                        word_count=doc.metadata.word_count,
                        page_count=doc.metadata.page_count or 1,
                        custom_metadata=meta_dict,
                    )
                    session.add(record)
                else:
                    record.source = doc.metadata.source
                    record.file_type = doc.metadata.file_type
                    record.checksum = doc.metadata.checksum
                    record.content = doc.content
                    record.character_count = doc.metadata.character_count
                    record.word_count = doc.metadata.word_count
                    record.page_count = doc.metadata.page_count or 1
                    record.custom_metadata = meta_dict
        except Exception as exc:
            logger.warning("pgvector_store_document_failed_using_fallback", error=str(exc))
            await self._fallback_repo.store_document(doc)

    async def store_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        try:
            async with self._db_manager.get_session() as session:
                for chunk in chunks:
                    stmt = select(DocumentChunkRecord).where(
                        DocumentChunkRecord.id == chunk.chunk_id
                    )
                    res = await session.execute(stmt)
                    record = res.scalar_one_or_none()

                    meta_dict = chunk.metadata.custom_metadata.copy()
                    if chunk.metadata.document_type:
                        meta_dict["document_type"] = chunk.metadata.document_type
                    if chunk.metadata.department:
                        meta_dict["department"] = chunk.metadata.department
                    if chunk.metadata.date:
                        meta_dict["date"] = chunk.metadata.date
                    if chunk.metadata.author:
                        meta_dict["author"] = chunk.metadata.author
                    if chunk.metadata.tags:
                        meta_dict["tags"] = chunk.metadata.tags

                    if record is None:
                        record = DocumentChunkRecord(
                            id=chunk.chunk_id,
                            doc_id=chunk.doc_id,
                            chunk_index=chunk.metadata.index,
                            content=chunk.content,
                            start_char=chunk.metadata.start_char,
                            end_char=chunk.metadata.end_char,
                            character_count=chunk.metadata.character_count,
                            word_count=chunk.metadata.word_count,
                            custom_metadata=meta_dict,
                            embedding=chunk.embedding,
                        )
                        session.add(record)
                    else:
                        record.content = chunk.content
                        record.start_char = chunk.metadata.start_char
                        record.end_char = chunk.metadata.end_char
                        record.character_count = chunk.metadata.character_count
                        record.word_count = chunk.metadata.word_count
                        record.custom_metadata = meta_dict
                        record.embedding = chunk.embedding
        except Exception as exc:
            logger.warning("pgvector_store_chunks_failed_using_fallback", error=str(exc))
            await self._fallback_repo.store_chunks(chunks)

    async def search_similar_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float = 0.0,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[tuple[EmbeddedChunk, float]]:
        filter_obj = normalize_metadata_filter(filters)
        try:
            async with self._db_manager.get_session() as session:
                # pgvector cosine distance: <=> operator
                distance_col = DocumentChunkRecord.embedding.cosine_distance(query_embedding).label(
                    "distance"
                )
                fetch_limit = top_k * 5 if filter_obj is not None else top_k
                stmt = (
                    select(DocumentChunkRecord, distance_col)
                    .options(selectinload(DocumentChunkRecord.document))
                    .where(DocumentChunkRecord.embedding.is_not(None))
                    .order_by(distance_col)
                    .limit(fetch_limit)
                )

                result = await session.execute(stmt)
                rows = result.all()

                results: list[tuple[EmbeddedChunk, float]] = []
                for record, distance in rows:
                    # Cosine distance = 1 - cosine_similarity => similarity = 1 - distance
                    similarity = 1.0 - float(distance) if distance is not None else 0.0
                    if similarity >= min_similarity:
                        meta_dict = record.custom_metadata or {}
                        raw_tags = meta_dict.get("tags", [])
                        tags_list = raw_tags if isinstance(raw_tags, list) else []
                        meta = ChunkMetadata(
                            chunk_id=record.id,
                            doc_id=record.doc_id,
                            index=record.chunk_index,
                            start_char=record.start_char,
                            end_char=record.end_char,
                            character_count=record.character_count,
                            word_count=record.word_count,
                            source=record.document.source if record.document else "unknown",
                            file_type=record.document.file_type if record.document else "txt",
                            checksum=record.document.checksum if record.document else "",
                            document_type=meta_dict.get("document_type"),
                            department=meta_dict.get("department"),
                            date=meta_dict.get("date"),
                            author=meta_dict.get("author"),
                            tags=tags_list,
                            custom_metadata=meta_dict,
                        )

                        if filter_obj is not None and not filter_obj.matches(meta):
                            continue

                        embedding_list = (
                            list(record.embedding) if record.embedding is not None else []
                        )
                        chunk = EmbeddedChunk(
                            chunk_id=record.id,
                            doc_id=record.doc_id,
                            content=record.content,
                            embedding=embedding_list,
                            metadata=meta,
                        )
                        results.append((chunk, round(similarity, 6)))
                        if len(results) >= top_k:
                            break

                return results
        except Exception as exc:
            logger.warning("pgvector_search_failed_using_fallback", error=str(exc))
            return await self._fallback_repo.search_similar_chunks(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                filters=filters,
            )

    async def list_chunks(
        self,
        filters: MetadataFilter | dict[str, Any] | None = None,
    ) -> list[EmbeddedChunk]:
        filter_obj = normalize_metadata_filter(filters)
        try:
            async with self._db_manager.get_session() as session:
                stmt = select(DocumentChunkRecord).options(
                    selectinload(DocumentChunkRecord.document)
                )
                res = await session.execute(stmt)
                records = res.scalars().all()
                chunks: list[EmbeddedChunk] = []
                for record in records:
                    meta_dict = record.custom_metadata or {}
                    raw_tags = meta_dict.get("tags", [])
                    tags_list = raw_tags if isinstance(raw_tags, list) else []
                    meta = ChunkMetadata(
                        chunk_id=record.id,
                        doc_id=record.doc_id,
                        index=record.chunk_index,
                        start_char=record.start_char,
                        end_char=record.end_char,
                        character_count=record.character_count,
                        word_count=record.word_count,
                        source=record.document.source if record.document else "unknown",
                        file_type=record.document.file_type if record.document else "txt",
                        checksum=record.document.checksum if record.document else "",
                        document_type=meta_dict.get("document_type"),
                        department=meta_dict.get("department"),
                        date=meta_dict.get("date"),
                        author=meta_dict.get("author"),
                        tags=tags_list,
                        custom_metadata=meta_dict,
                    )
                    if filter_obj is not None and not filter_obj.matches(meta):
                        continue
                    chunks.append(
                        EmbeddedChunk(
                            chunk_id=record.id,
                            doc_id=record.doc_id,
                            content=record.content,
                            embedding=record.embedding if record.embedding is not None else [],
                            metadata=meta,
                        )
                    )
                return chunks
        except Exception as exc:
            logger.warning("pgvector_list_chunks_failed_using_fallback", error=str(exc))
            return await self._fallback_repo.list_chunks(filters=filters)

    async def get_document(self, doc_id: str) -> Document | None:
        try:
            async with self._db_manager.get_session() as session:
                stmt = select(DocumentRecord).where(DocumentRecord.id == doc_id)
                res = await session.execute(stmt)
                record = res.scalar_one_or_none()
                if not record:
                    return None

                meta = DocumentMetadata(
                    doc_id=record.id,
                    source=record.source,
                    file_type=record.file_type,
                    checksum=record.checksum,
                    character_count=record.character_count,
                    word_count=record.word_count,
                    page_count=record.page_count,
                    custom_metadata=record.custom_metadata or {},
                )
                return Document(doc_id=record.id, content=record.content, metadata=meta)
        except Exception as exc:
            logger.warning("pgvector_get_document_failed_using_fallback", error=str(exc))
            return await self._fallback_repo.get_document(doc_id)

    async def list_documents(self) -> list[Document]:
        try:
            async with self._db_manager.get_session() as session:
                stmt = select(DocumentRecord).order_by(DocumentRecord.created_at.desc())
                res = await session.execute(stmt)
                records = res.scalars().all()

                docs: list[Document] = []
                for r in records:
                    meta = DocumentMetadata(
                        doc_id=r.id,
                        source=r.source,
                        file_type=r.file_type,
                        checksum=r.checksum,
                        character_count=r.character_count,
                        word_count=r.word_count,
                        page_count=r.page_count,
                        custom_metadata=r.custom_metadata or {},
                    )
                    docs.append(Document(doc_id=r.id, content=r.content, metadata=meta))
                return docs
        except Exception as exc:
            logger.warning("pgvector_list_documents_failed_using_fallback", error=str(exc))
            return await self._fallback_repo.list_documents()

    async def delete_document(self, doc_id: str) -> bool:
        try:
            async with self._db_manager.get_session() as session:
                stmt = delete(DocumentRecord).where(DocumentRecord.id == doc_id)
                res = await session.execute(stmt)
                rowcount = getattr(res, "rowcount", 0)
                return bool(isinstance(rowcount, int) and rowcount > 0)
        except Exception as exc:
            logger.warning("pgvector_delete_document_failed_using_fallback", error=str(exc))
            return await self._fallback_repo.delete_document(doc_id)


def create_vector_repository(
    settings: Settings | None = None,
    db_manager: DatabaseManager | None = None,
) -> BaseVectorRepository:
    """Factory creating appropriate vector repository instance with persistent fallback."""
    current_settings = settings or get_settings()
    repo_type = current_settings.vector_repository_type.lower().strip()
    storage_path = (
        Path(current_settings.storage_dir) / "vectors.json"
        if current_settings.storage_dir
        else None
    )

    fallback_in_memory = InMemoryVectorRepository(storage_path=storage_path)

    if repo_type == "postgres":
        manager = db_manager or DatabaseManager(settings=current_settings)
        return PgVectorRepository(manager, fallback_repo=fallback_in_memory)
    elif repo_type == "in_memory":
        return fallback_in_memory
    else:  # auto
        if current_settings.database_url and "postgres" in current_settings.database_url:
            manager = db_manager or DatabaseManager(settings=current_settings)
            return PgVectorRepository(manager, fallback_repo=fallback_in_memory)
        return fallback_in_memory
