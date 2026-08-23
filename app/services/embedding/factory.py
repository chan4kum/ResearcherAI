from app.config import Settings, get_settings
from app.core.errors import AppException
from app.core.logging import get_logger
from app.services.embedding.base import BaseEmbeddingProvider
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding.openai import OpenAIEmbeddingProvider

logger = get_logger("app.services.embedding.factory")


def create_embedding_provider(
    settings: Settings | None = None,
) -> BaseEmbeddingProvider:
    """Instantiate and configure the appropriate BaseEmbeddingProvider from Settings."""
    current_settings = settings or get_settings()
    provider_type = current_settings.embedding_provider.lower().strip()

    logger.info(
        "creating_embedding_provider",
        provider=provider_type,
        model=current_settings.embedding_model,
        dimensions=current_settings.embedding_dimensions,
    )

    if provider_type == "mock":
        return MockEmbeddingProvider(
            dimensions=current_settings.embedding_dimensions,
            model=current_settings.embedding_model,
        )

    elif provider_type in {"openai", "openai-compatible"}:
        if not current_settings.openai_api_key:
            raise AppException(
                code="CONFIG_ERROR",
                message="OPENAI_API_KEY must be configured when EMBEDDING_PROVIDER='openai'",
                status_code=500,
            )
        return OpenAIEmbeddingProvider(
            api_key=current_settings.openai_api_key,
            model=current_settings.embedding_model,
            dimensions=current_settings.embedding_dimensions,
            base_url=current_settings.openai_base_url,
            timeout_seconds=current_settings.llm_timeout_seconds,
        )

    else:
        raise AppException(
            code="UNSUPPORTED_EMBEDDING_PROVIDER",
            message=f"Unsupported embedding provider: '{provider_type}'",
            status_code=500,
        )
