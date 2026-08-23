import time
from typing import Any

from openai import AsyncOpenAI, AuthenticationError, OpenAIError

from app.core.errors import AppException
from app.core.logging import get_logger
from app.services.embedding.base import BaseEmbeddingProvider, EmbeddingResponse

logger = get_logger("app.services.embedding.openai")


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding provider adapter interfacing with OpenAI and OpenAI-compatible embeddings APIs."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        dimensions: int | None = 1536,
        base_url: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    async def embed_texts(self, texts: list[str]) -> EmbeddingResponse:
        """Generate vector embeddings using OpenAI API."""
        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                model=self.model,
                dimensions=self.dimensions or 0,
                total_tokens=0,
                duration_ms=0.0,
            )

        start_time = time.perf_counter()
        logger.info("openai_embedding_request_start", model=self.model, count=len(texts))

        try:
            kwargs: dict[str, Any] = {"input": texts, "model": self.model}
            if self.dimensions and "text-embedding-3" in self.model:
                kwargs["dimensions"] = self.dimensions

            response = await self.client.embeddings.create(**kwargs)
            duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

            embeddings = [item.embedding for item in response.data]
            actual_dims = len(embeddings[0]) if embeddings else (self.dimensions or 0)
            total_tokens = response.usage.total_tokens if response.usage else 0

            logger.info(
                "openai_embedding_request_success",
                model=self.model,
                count=len(embeddings),
                dimensions=actual_dims,
                duration_ms=duration_ms,
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=self.model,
                dimensions=actual_dims,
                total_tokens=total_tokens,
                duration_ms=duration_ms,
            )

        except AuthenticationError as exc:
            logger.error("openai_embedding_auth_error", error=str(exc))
            raise AppException(
                code="LLM_AUTH_ERROR",
                message="Invalid or missing OpenAI API key for embeddings",
                status_code=401,
            ) from exc

        except OpenAIError as exc:
            logger.error("openai_embedding_api_error", error=str(exc))
            raise AppException(
                code="LLM_PROVIDER_ERROR",
                message=f"OpenAI embedding provider error: {exc}",
                status_code=502,
            ) from exc
