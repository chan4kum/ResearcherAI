import hashlib
import math
import random
import time

from app.core.logging import get_logger
from app.services.embedding.base import BaseEmbeddingProvider, EmbeddingResponse

logger = get_logger("app.services.embedding.mock")


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic, mock embedding provider for 100% offline, credential-free testing."""

    def __init__(
        self,
        dimensions: int = 1536,
        model: str = "mock-embedding-v1",
        should_fail: bool = False,
        failure_message: str = "Simulated MockEmbeddingProvider failure",
    ) -> None:
        self.dimensions = dimensions
        self.model = model
        self.should_fail = should_fail
        self.failure_message = failure_message
        self.calls: list[dict[str, object]] = []

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generate a deterministic, unit-normalized float vector based on text hash."""
        # Use SHA-256 hash integer as pseudo-random seed
        seed_val = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
        rng = random.Random(seed_val)

        # Generate Gaussian random components
        raw_vector = [rng.gauss(0.0, 1.0) for _ in range(self.dimensions)]

        # L2-normalize to unit length
        norm = math.sqrt(sum(x * x for x in raw_vector))
        if norm == 0.0:
            return [1.0 / math.sqrt(self.dimensions)] * self.dimensions

        return [round(x / norm, 6) for x in raw_vector]

    async def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Generate deterministic vector embeddings for input texts."""
        start_time = time.perf_counter()

        self.calls.append({"texts_count": len(texts), "model": self.model})
        logger.info("mock_embedding_embed_texts", texts_count=len(texts), model=self.model)

        if self.should_fail:
            raise RuntimeError(self.failure_message)

        embeddings: list[list[float]] = []
        total_tokens = 0

        for t in texts:
            vec = self._generate_deterministic_vector(t)
            embeddings.append(vec)
            # Approximate token count: 1 word ~ 1.3 tokens
            total_tokens += max(1, int(len(t.split()) * 1.3))

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return EmbeddingResponse(
            embeddings=embeddings,
            model=self.model,
            dimensions=self.dimensions,
            total_tokens=total_tokens,
            duration_ms=duration_ms,
        )
