"""Embedding service package providing vector representation abstractions and providers."""

from app.services.embedding.base import (
    BaseEmbeddingProvider,
    EmbeddingResponse,
    cosine_similarity,
)
from app.services.embedding.factory import create_embedding_provider
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding.openai import OpenAIEmbeddingProvider
from app.services.embedding.service import EmbeddingService

__all__ = [
    "BaseEmbeddingProvider",
    "EmbeddingResponse",
    "EmbeddingService",
    "MockEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "cosine_similarity",
    "create_embedding_provider",
]
