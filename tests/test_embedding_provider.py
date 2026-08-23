from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.config import Settings
from app.core.errors import AppException
from app.services.embedding.base import cosine_similarity
from app.services.embedding.factory import create_embedding_provider
from app.services.embedding.mock import MockEmbeddingProvider
from app.services.embedding.openai import OpenAIEmbeddingProvider
from openai import AuthenticationError, OpenAIError


def test_cosine_similarity_identical_vectors() -> None:
    """Verify identical vectors yield a cosine similarity of exactly 1.0."""
    vec = [0.5, 0.5, 0.5, 0.5]
    sim = cosine_similarity(vec, vec)
    assert pytest.approx(sim, abs=1e-5) == 1.0


def test_cosine_similarity_orthogonal_vectors() -> None:
    """Verify orthogonal vectors yield a cosine similarity of 0.0."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    sim = cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, abs=1e-5) == 0.0


def test_cosine_similarity_opposite_vectors() -> None:
    """Verify opposite vectors yield a cosine similarity of -1.0."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    sim = cosine_similarity(vec1, vec2)
    assert pytest.approx(sim, abs=1e-5) == -1.0


def test_cosine_similarity_zero_vectors() -> None:
    """Verify zero vectors return 0.0 without division by zero errors."""
    vec1 = [0.0, 0.0, 0.0]
    vec2 = [1.0, 2.0, 3.0]
    assert cosine_similarity(vec1, vec2) == 0.0


def test_cosine_similarity_dimension_mismatch() -> None:
    """Verify dimension mismatch raises ValueError."""
    vec1 = [1.0, 2.0]
    vec2 = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="vector dimension mismatch"):
        cosine_similarity(vec1, vec2)


@pytest.mark.asyncio
async def test_mock_embedding_provider_deterministic() -> None:
    """Verify MockEmbeddingProvider generates consistent, normalized embeddings."""
    provider = MockEmbeddingProvider(dimensions=384, model="mock-384")
    text = "Artificial intelligence and agentic workflows."

    resp1 = await provider.embed_texts([text])
    resp2 = await provider.embed_texts([text])

    assert resp1.dimensions == 384
    assert resp1.model == "mock-384"
    assert len(resp1.embeddings) == 1
    assert len(resp1.embeddings[0]) == 384

    # Determinism: same input produces identical vector
    assert resp1.embeddings[0] == resp2.embeddings[0]

    # Unit length check (L2 norm ~ 1.0)
    norm = sum(x * x for x in resp1.embeddings[0])
    assert pytest.approx(norm, abs=1e-2) == 1.0


@pytest.mark.asyncio
async def test_mock_embedding_provider_different_texts() -> None:
    """Verify distinct texts produce different embeddings."""
    provider = MockEmbeddingProvider(dimensions=128)
    resp = await provider.embed_texts(["Semiconductors", "Deep ocean marine biology"])

    assert len(resp.embeddings) == 2
    assert resp.embeddings[0] != resp.embeddings[1]


@pytest.mark.asyncio
async def test_mock_embedding_provider_simulated_failure() -> None:
    """Verify MockEmbeddingProvider raises error when should_fail=True."""
    provider = MockEmbeddingProvider(should_fail=True)
    with pytest.raises(RuntimeError, match="Simulated MockEmbeddingProvider failure"):
        await provider.embed_texts(["test string"])


@pytest.mark.asyncio
async def test_openai_embedding_provider_success() -> None:
    """Verify OpenAIEmbeddingProvider parses successful API response."""
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test-key",
        model="text-embedding-3-small",
        dimensions=4,
    )

    mock_item = MagicMock()
    mock_item.embedding = [0.1, 0.2, 0.3, 0.4]

    mock_resp = MagicMock()
    mock_resp.data = [mock_item]
    mock_resp.usage.total_tokens = 10

    with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_resp
        res = await provider.embed_texts(["Hello world"])

        assert res.dimensions == 4
        assert res.total_tokens == 10
        assert res.embeddings == [[0.1, 0.2, 0.3, 0.4]]
        mock_create.assert_awaited_once()


@pytest.mark.asyncio
async def test_openai_embedding_provider_empty_texts() -> None:
    """Verify OpenAIEmbeddingProvider handles empty text list cleanly."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test-key")
    res = await provider.embed_texts([])
    assert res.embeddings == []
    assert res.total_tokens == 0


@pytest.mark.asyncio
async def test_openai_embedding_provider_auth_error() -> None:
    """Verify OpenAIEmbeddingProvider wraps AuthenticationError in 401 AppException."""
    provider = OpenAIEmbeddingProvider(api_key="sk-invalid-key")

    with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = AuthenticationError(
            "Invalid key", response=MagicMock(), body=None
        )
        with pytest.raises(AppException) as exc_info:
            await provider.embed_texts(["Hello"])
        assert exc_info.value.status_code == 401
        assert exc_info.value.code == "LLM_AUTH_ERROR"


@pytest.mark.asyncio
async def test_openai_embedding_provider_api_error() -> None:
    """Verify OpenAIEmbeddingProvider wraps OpenAIError in 502 AppException."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test-key")

    with patch.object(provider.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
        mock_create.side_effect = OpenAIError("Service unavailable")
        with pytest.raises(AppException) as exc_info:
            await provider.embed_texts(["Hello"])
        assert exc_info.value.status_code == 502
        assert exc_info.value.code == "LLM_PROVIDER_ERROR"


def test_factory_returns_mock_embedding_provider() -> None:
    """Verify create_embedding_provider returns MockEmbeddingProvider for 'mock'."""
    settings = Settings(embedding_provider="mock", embedding_dimensions=768)
    provider = create_embedding_provider(settings)
    assert isinstance(provider, MockEmbeddingProvider)
    assert provider.dimensions == 768


def test_factory_returns_openai_embedding_provider() -> None:
    """Verify create_embedding_provider returns OpenAIEmbeddingProvider when configured."""
    settings = Settings(embedding_provider="openai", openai_api_key="sk-test-12345")
    provider = create_embedding_provider(settings)
    assert isinstance(provider, OpenAIEmbeddingProvider)


def test_factory_raises_when_openai_key_missing() -> None:
    """Verify create_embedding_provider raises when openai_api_key is omitted."""
    settings = Settings(embedding_provider="openai", openai_api_key=None)
    with pytest.raises(AppException) as exc:
        create_embedding_provider(settings)
    assert exc.value.status_code == 500
    assert exc.value.code == "CONFIG_ERROR"


def test_factory_raises_for_unsupported_provider() -> None:
    """Verify create_embedding_provider raises for invalid provider string."""
    settings = Settings(embedding_provider="unknown-provider")
    with pytest.raises(AppException) as exc:
        create_embedding_provider(settings)
    assert exc.value.code == "UNSUPPORTED_EMBEDDING_PROVIDER"
