import math
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class EmbeddingResponse(BaseModel):
    """Standardized response from an embedding provider containing vector representations."""

    embeddings: list[list[float]] = Field(
        description="List of vector embeddings corresponding to input texts",
    )
    model: str = Field(description="The model name utilized for embedding generation")
    dimensions: int = Field(description="Dimensionality of each generated vector embedding")
    total_tokens: int = Field(
        default=0,
        description="Total tokens consumed during embedding generation",
    )
    duration_ms: float = Field(
        default=0.0,
        description="Latency of embedding generation in milliseconds",
    )


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Calculate the cosine similarity between two numeric vectors in pure Python.

    Cosine Similarity measures the cosine of the angle between two non-zero vectors in an
    inner product space. Formula: (A · B) / (||A|| * ||B||).
    Range: [-1.0, 1.0], where 1.0 indicates identical orientation and 0.0 indicates orthogonality.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(
            f"Cannot calculate cosine similarity: vector dimension mismatch "
            f"({len(vector_a)} != {len(vector_b)})"
        )

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    # Clamp float precision boundary to [-1.0, 1.0]
    return max(-1.0, min(1.0, float(similarity)))


class BaseEmbeddingProvider(ABC):
    """Abstract port for embedding model providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Generate vector embeddings for a batch of text strings."""
        pass

    async def embed_text(self, text: str) -> list[float]:
        """Convenience method to generate embedding for a single text string."""
        response = await self.embed_texts([text])
        if not response.embeddings:
            raise RuntimeError("Embedding provider returned empty embeddings list")
        return response.embeddings[0]
