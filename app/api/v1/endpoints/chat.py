from fastapi import APIRouter, Depends, Request

from app.api.dependencies import get_llm_service
from app.models.schemas import ChatRequest, ChatResponse
from app.services.llm.service import LLMService

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Chat completion",
    description="Send a message to the configured LLM provider and receive a generated response.",
)
async def chat_completion(
    payload: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
) -> ChatResponse:
    """Execute a single-turn chat completion using the configured LLM service."""
    result = await llm_service.chat(
        message=payload.message,
        system_prompt=payload.system_prompt,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )

    return ChatResponse(
        answer=result.content,
        model=result.model,
        provider=result.provider,
    )
