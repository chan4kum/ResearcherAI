from fastapi import APIRouter, Depends, Request

from app.models.schemas import (
    EmbeddingGenerateRequest,
    EmbeddingGenerateResponse,
    SimilarityRequest,
    SimilarityResponse,
)
from app.services.embedding.service import EmbeddingService

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


def get_embedding_service(request: Request) -> EmbeddingService:
    """Extract EmbeddingService instance from FastAPI application state."""
    service = getattr(request.app.state, "embedding_service", None)
    if not isinstance(service, EmbeddingService):
        service = EmbeddingService()
        request.app.state.embedding_service = service
    return service


@router.post(
    "/generate",
    response_model=EmbeddingGenerateResponse,
    summary="Generate vector embeddings for input texts",
)
async def generate_embeddings(
    payload: EmbeddingGenerateRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> EmbeddingGenerateResponse:
    """Generate dense float vector embeddings for a batch of input text strings."""
    response = await embedding_service.embed_texts(payload.texts)
    return EmbeddingGenerateResponse(
        model=response.model,
        dimensions=response.dimensions,
        total_embeddings=len(response.embeddings),
        embeddings=response.embeddings,
        total_tokens=response.total_tokens,
        duration_ms=response.duration_ms,
    )


@router.post(
    "/similarity",
    response_model=SimilarityResponse,
    summary="Calculate semantic cosine similarity between two text strings",
)
async def compute_similarity(
    payload: SimilarityRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> SimilarityResponse:
    """Generate embeddings for two strings and calculate their cosine similarity score."""
    vec_a = await embedding_service.embed_text(payload.text_a)
    vec_b = await embedding_service.embed_text(payload.text_b)
    similarity = embedding_service.compute_similarity(vec_a, vec_b)

    return SimilarityResponse(
        text_a=payload.text_a,
        text_b=payload.text_b,
        cosine_similarity=round(similarity, 6),
    )
