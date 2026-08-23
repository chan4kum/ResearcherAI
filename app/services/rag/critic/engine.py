import time

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.critic.agent import CriticAgent
from app.services.rag.critic.models import (
    CriticEvaluation,
    SelfCorrectionAttempt,
    SelfCorrectionResult,
)
from app.services.rag.models import Citation

logger = get_logger("app.services.rag.critic.engine")

REVISION_SYSTEM_PROMPT = """You are an Expert Fact-Check and Answer Revision Assistant.
Your task is to revise and improve a draft answer based on specific critic feedback.

Rules:
1. Address every identified critic issue strictly.
2. Remove any unsupported claims or hallucinations.
3. Resolve contradictions to ensure absolute factual grounding in the provided evidence.
4. Keep the answer structured, professional, and clear.
"""

REVISION_USER_PROMPT = """Question:
{question}

Evidence:
{evidence}

Previous Draft:
{previous_draft}

Critic Issues and Feedback:
{feedback}

Issues Identified:
{issues_list}

Produce the corrected, fully grounded answer.
"""


class SelfCorrectionEngine:
    """Orchestrates iterative self-correction between generator and critic with strict limits."""

    def __init__(
        self,
        critic_agent: CriticAgent | None = None,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)
        self._critic = critic_agent or CriticAgent(
            llm_service=self._llm_service,
            settings=self._settings,
        )

    async def correct_answer(
        self,
        question: str,
        evidence: list[str] | str,
        draft_answer: str,
        citations: list[Citation] | None = None,
        max_corrections: int = 2,
    ) -> SelfCorrectionResult:
        """Run bounded self-correction loop on draft answer until accepted or max iterations."""
        clean_question = question.strip()
        current_draft = draft_answer.strip()
        bounded_max = max(1, min(max_corrections, 5))
        start_time = time.perf_counter()

        evidence_text = (
            "\n\n".join(evidence) if isinstance(evidence, list) else evidence.strip()
        )
        active_citations = citations or []
        attempts: list[SelfCorrectionAttempt] = []

        logger.info(
            "self_correction_loop_start",
            question=clean_question[:80],
            max_corrections=bounded_max,
        )

        final_evaluation: CriticEvaluation | None = None

        for iteration in range(1, bounded_max + 1):
            iter_start = time.perf_counter()

            # 1. Critic evaluation
            evaluation = await self._critic.evaluate(
                question=clean_question,
                evidence=evidence_text,
                draft_answer=current_draft,
                citations=active_citations,
            )
            final_evaluation = evaluation

            # 2. Check if acceptable or last iteration
            if evaluation.is_acceptable:
                iter_ms = round((time.perf_counter() - iter_start) * 1000, 2)
                attempts.append(
                    SelfCorrectionAttempt(
                        iteration=iteration,
                        draft_answer=current_draft,
                        evaluation=evaluation,
                        revised_answer=None,
                        duration_ms=iter_ms,
                    )
                )
                logger.info(
                    "self_correction_accepted",
                    iteration=iteration,
                    score=evaluation.critique_score,
                )
                break

            # If not acceptable and more iterations allowed, generate revision
            issues_formatted = "\n".join(
                f"- [{i.issue_type.value.upper()}] ({i.severity.value}): "
                f"{i.reason} -> Fix: {i.suggested_fix}"
                for i in evaluation.issues
            )

            prompt = REVISION_USER_PROMPT.format(
                question=clean_question,
                evidence=evidence_text,
                previous_draft=current_draft,
                feedback=evaluation.feedback_summary,
                issues_list=issues_formatted,
            )

            llm_resp = await self._llm_service.generate(
                prompt=prompt,
                system_prompt=REVISION_SYSTEM_PROMPT,
                temperature=0.2,
            )
            revised_answer = llm_resp.content.strip()
            iter_ms = round((time.perf_counter() - iter_start) * 1000, 2)

            attempts.append(
                SelfCorrectionAttempt(
                    iteration=iteration,
                    draft_answer=current_draft,
                    evaluation=evaluation,
                    revised_answer=revised_answer,
                    duration_ms=iter_ms,
                )
            )

            current_draft = revised_answer
            logger.info(
                "self_correction_iteration_completed",
                iteration=iteration,
                score=evaluation.critique_score,
                issues_count=len(evaluation.issues),
            )

        # Evaluate final answer if revisions took place
        if len(attempts) > 0 and attempts[-1].revised_answer is not None:
            final_evaluation = await self._critic.evaluate(
                question=clean_question,
                evidence=evidence_text,
                draft_answer=current_draft,
                citations=active_citations,
            )

        total_ms = round((time.perf_counter() - start_time) * 1000, 2)
        is_corrected = current_draft != draft_answer.strip()

        assert final_evaluation is not None

        return SelfCorrectionResult(
            question=clean_question,
            original_draft=draft_answer.strip(),
            final_answer=current_draft,
            iterations=len(attempts),
            max_iterations=bounded_max,
            is_corrected=is_corrected,
            final_evaluation=final_evaluation,
            attempts=attempts,
            total_duration_ms=total_ms,
            citations=active_citations,
            metadata={"bounded_max": bounded_max},
        )
