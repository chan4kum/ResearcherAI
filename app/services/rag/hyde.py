from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService

logger = get_logger("app.services.rag.hyde")

HYDE_SYSTEM_PROMPT = """You are an expert technical and scientific document synthesizer.
Given a user query, write a plausible, highly-detailed excerpt or passage from an
authoritative document (engineering report, technical manual, regulatory filing)
that answers the question.

Rules:
- Write in the formal style of a factual technical document.
- Include plausible technical terminology, parameters, procedures, and domain concepts.
- Do NOT include conversational preambles, introductory greetings, or disclaimers.
- Output ONLY the hypothetical passage text.
"""


class HyDEResult(BaseModel):
    """Telemetry item recording HyDE generation and retrieval metadata."""

    original_query: str = Field(description="Original user query")
    hypothetical_document: str = Field(
        description="Synthetic document passage generated to bridge the semantic gap"
    )
    strategy: str = Field(default="hyde", description="Retrieval strategy used")
    retrieved_documents_count: int = Field(
        default=0, description="Number of real documents retrieved via HyDE"
    )


class HyDEGenerator:
    """Generates synthetic hypothetical document passages for query embedding."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)

    def _heuristic_generate(self, query: str) -> str:
        """Deterministic heuristic fallback generating a plausible technical passage."""
        clean_q = query.strip()
        return (
            f"Technical Documentation and Operational Specification Bulletin: {clean_q}. "
            f"Official investigation findings, component specifications, maintenance procedures, "
            f"and root-cause analyses provide verified engineering data regarding {clean_q}. "
            f"Key metrics, regulatory compliance records, and diagnostics confirm specific "
            f"operational parameters and system resolutions."
        )

    async def generate(self, query: str) -> str:
        """Synthesize a hypothetical passage that answers the user question."""
        clean_q = query.strip()
        if not clean_q:
            raise ValueError("Query cannot be empty for HyDE generation.")

        logger.info("hyde_generation_started", query=clean_q[:80])

        if getattr(self._settings, "llm_provider", "mock") == "mock":
            hypo_doc = self._heuristic_generate(clean_q)
            logger.info("hyde_generated_heuristic", length=len(hypo_doc))
            return hypo_doc

        try:
            user_prompt = f"User Question: {clean_q}\n\nHypothetical Passage:"
            response = await self._llm_service.generate(
                prompt=user_prompt,
                system_prompt=HYDE_SYSTEM_PROMPT,
                temperature=0.4,
            )
            content = response.content.strip()
            if not content:
                content = self._heuristic_generate(clean_q)
            logger.info("hyde_generated_llm", length=len(content))
            return content
        except Exception as exc:
            logger.warning("hyde_generation_fallback_to_heuristic", error=str(exc))
            return self._heuristic_generate(clean_q)
