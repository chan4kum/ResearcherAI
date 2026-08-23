from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.document.models import DocumentChunk, EmbeddedChunk
from app.services.embedding.base import (
    BaseEmbeddingProvider,
    EmbeddingResponse,
    cosine_similarity,
)
from app.services.embedding.factory import create_embedding_provider

logger = get_logger("app.services.embedding.service")


class EmbeddingService:
    """Domain service managing text embedding generation and vector similarity operations."""

    def __init__(
        self,
        provider: BaseEmbeddingProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or create_embedding_provider(self._settings)

    @property
    def provider(self) -> BaseEmbeddingProvider:
        return self._provider

    async def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Generate vector embeddings for a list of text strings."""
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model="empty",
                dimensions=0,
                total_tokens=0,
                duration_ms=0.0,
            )

        logger.info("embedding_service_embed_texts", count=len(texts))
        return await self._provider.embed_texts(texts)

    async def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a single text string."""
        return await self._provider.embed_text(text)

    async def embed_chunks(
        self,
        chunks: list[DocumentChunk],
    ) -> list[EmbeddedChunk]:
        """Generate vector embeddings for document chunks and return EmbeddedChunk objects."""
        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        response = await self.embed_texts(texts)

        embedded_chunks: list[EmbeddedChunk] = []
        for chunk, embedding in zip(chunks, response.embeddings, strict=True):
            # Also attach embedding to chunk model
            chunk.embedding = embedding
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    embedding=embedding,
                    metadata=chunk.metadata,
                )
            )

        logger.info(
            "embedded_chunks_created",
            total_chunks=len(embedded_chunks),
            dimensions=response.dimensions,
        )
        return embedded_chunks

    def compute_similarity(
        self,
        vector_a: list[float],
        vector_b: list[float],
    ) -> float:
        """Compute cosine similarity between two float vectors."""
        return cosine_similarity(vector_a, vector_b)
