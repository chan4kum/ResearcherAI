from app.core.logging import get_logger
from app.services.document.models import (
    ChunkingConfig,
    ChunkMetadata,
    Document,
    DocumentChunk,
    DocumentMetadata,
)

logger = get_logger("app.services.document.chunker")


class DocumentChunker:
    """Deterministic document chunker with overlap and metadata preservation."""

    def __init__(self, default_config: ChunkingConfig | None = None) -> None:
        self._default_config = default_config or ChunkingConfig()

    def chunk_document(
        self,
        document: Document,
        config: ChunkingConfig | None = None,
    ) -> list[DocumentChunk]:
        """Split a Document into deterministic DocumentChunk objects with preserved metadata."""
        active_config = config or self._default_config
        active_config.validate_overlap()

        text = document.content
        if not text or not text.strip():
            logger.info("chunk_document_empty", doc_id=document.doc_id)
            return []

        chunks: list[DocumentChunk] = []
        text_len = len(text)
        chunk_size = active_config.chunk_size
        overlap = active_config.chunk_overlap
        step = chunk_size - overlap

        # Handle documents smaller than or equal to chunk_size
        if text_len <= chunk_size:
            meta = ChunkMetadata(
                chunk_id=f"{document.doc_id}_chunk_0",
                doc_id=document.doc_id,
                index=0,
                start_char=0,
                end_char=text_len,
                character_count=text_len,
                word_count=len(text.split()),
                source=document.metadata.source,
                file_type=document.metadata.file_type,
                checksum=document.metadata.checksum,
                document_type=document.metadata.document_type,
                department=document.metadata.department,
                date=document.metadata.date,
                author=document.metadata.author,
                tags=list(document.metadata.tags),
                custom_metadata=document.metadata.custom_metadata.copy(),
            )
            return [
                DocumentChunk(
                    chunk_id=meta.chunk_id,
                    doc_id=document.doc_id,
                    content=text,
                    metadata=meta,
                )
            ]

        # Sliding window chunking
        start = 0
        chunk_index = 0

        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end]

            chunk_meta = ChunkMetadata(
                chunk_id=f"{document.doc_id}_chunk_{chunk_index}",
                doc_id=document.doc_id,
                index=chunk_index,
                start_char=start,
                end_char=end,
                character_count=len(chunk_text),
                word_count=len(chunk_text.split()),
                source=document.metadata.source,
                file_type=document.metadata.file_type,
                checksum=document.metadata.checksum,
                document_type=document.metadata.document_type,
                department=document.metadata.department,
                date=document.metadata.date,
                author=document.metadata.author,
                tags=list(document.metadata.tags),
                custom_metadata=document.metadata.custom_metadata.copy(),
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_meta.chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_text,
                    metadata=chunk_meta,
                )
            )

            if end >= text_len:
                break

            start += step
            chunk_index += 1

        logger.info(
            "document_chunked",
            doc_id=document.doc_id,
            chunks_generated=len(chunks),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return chunks

    def chunk_text(
        self,
        text: str,
        doc_id: str = "raw_input",
        source: str = "raw_text",
        file_type: str = "txt",
        custom_metadata: dict[str, object] | None = None,
        config: ChunkingConfig | None = None,
    ) -> list[DocumentChunk]:
        """Convenience method to chunk raw string content directly."""
        dummy_meta = DocumentMetadata(
            doc_id=doc_id,
            source=source,
            file_type=file_type,
            file_size_bytes=len(text.encode("utf-8")),
            checksum="raw_checksum",
            character_count=len(text),
            word_count=len(text.split()),
            custom_metadata=custom_metadata or {},
        )
        dummy_doc = Document(doc_id=doc_id, content=text, metadata=dummy_meta)
        return self.chunk_document(dummy_doc, config=config)


def display_document_chunks(
    document: Document,
    chunks: list[DocumentChunk],
    max_preview_length: int = 120,
) -> str:
    """Format and display document and its generated chunks as a clean ASCII diagram."""
    lines: list[str] = [
        "=" * 80,
        f"DOCUMENT: [{document.doc_id}]",
        f"Source: {document.metadata.source} | Format: {document.metadata.file_type} | "
        f"Chars: {document.metadata.character_count} | Words: {document.metadata.word_count}",
        "-" * 80,
        "Document Content:",
        document.content.strip(),
        "=" * 80,
        f"→ CHUNKS (Total: {len(chunks)})",
    ]

    if not chunks:
        lines.append("  (No chunks generated - document is empty)")
    else:
        for chunk in chunks:
            preview = chunk.content.replace("\n", " ")
            if len(preview) > max_preview_length:
                preview = preview[:max_preview_length] + "..."
            lines.extend(
                [
                    "-" * 80,
                    f"[{chunk.metadata.chunk_id}] (Index: {chunk.metadata.index} | "
                    f"Range: [{chunk.metadata.start_char}->{chunk.metadata.end_char}])",
                    f"Chars: {chunk.metadata.character_count} | Words: {chunk.metadata.word_count}",
                    f"Text: \"{preview}\"",
                ]
            )

    lines.append("=" * 80)
    return "\n".join(lines)
