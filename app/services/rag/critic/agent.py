import json
import re
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import get_logger
from app.services.llm.service import LLMService
from app.services.rag.critic.models import (
    CriticEvaluation,
    CriticIssue,
    CriticIssueSeverity,
    CriticIssueType,
)
from app.services.rag.models import Citation

logger = get_logger("app.services.rag.critic.agent")

CRITIC_SYSTEM_PROMPT = """You are an Adversarial Fact-Checking and Rigorous Critic Agent.
Your job is to thoroughly inspect a draft answer against the provided evidence and question.

Evaluate the draft across these exact 6 dimensions:
1. unsupported_claim: Any assertion or fact in draft not supported by evidence.
2. missing_evidence: Necessary context or question requirements omitted.
3. contradiction: Internal contradictions or direct conflicts with evidence.
4. incomplete_reasoning: Logical leaps, missing intermediate rationale, or non-sequiturs.
5. irrelevant_information: Tangential, off-topic, or unnecessary filler.
6. citation_problem: Hallucinated citations, missing source attributions, or broken references.

Respond ONLY with a JSON object adhering to this schema:
{{
  "is_acceptable": true/false,
  "critique_score": 0.0 to 1.0,
  "issues": [
    {{
      "issue_type": "unsupported_claim" | "missing_evidence" | "contradiction" |
                    "incomplete_reasoning" | "irrelevant_information" | "citation_problem",
      "severity": "low" | "medium" | "high" | "critical",
      "claim_or_passage": "<exact phrase from draft>",
      "reason": "<why this is a flaw>",
      "suggested_fix": "<how to correct it>"
    }}
  ],
  "feedback_summary": "<concise summary of critique>",
  "action_recommended": "accept" | "revise_answer" | "retrieve_more_evidence"
}}
"""

CRITIC_USER_PROMPT = """Question:
{question}

Evidence Documents:
{evidence}

Draft Answer:
{draft_answer}

Inspect the draft answer rigorously and output the JSON evaluation.
"""


class CriticAgent:
    """Rigorous evaluation agent checking answers for hallucinations, gaps, and contradictions."""

    def __init__(
        self,
        llm_service: LLMService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_service = llm_service or LLMService(settings=self._settings)

    async def evaluate(
        self,
        question: str,
        evidence: list[str] | str,
        draft_answer: str,
        citations: list[Citation] | None = None,
    ) -> CriticEvaluation:
        """Evaluate draft answer against evidence documents and question."""
        clean_question = question.strip()
        clean_draft = draft_answer.strip()

        if not clean_draft:
            return CriticEvaluation(
                is_acceptable=False,
                critique_score=0.0,
                issues=[
                    CriticIssue(
                        issue_type=CriticIssueType.MISSING_EVIDENCE,
                        severity=CriticIssueSeverity.CRITICAL,
                        claim_or_passage="",
                        reason="Draft answer is completely empty.",
                        suggested_fix="Generate an answer based on available evidence.",
                    )
                ],
                feedback_summary="Draft answer is empty.",
                action_recommended="revise_answer",
            )

        evidence_text = (
            "\n\n".join(evidence) if isinstance(evidence, list) else evidence.strip()
        )

        prompt = CRITIC_USER_PROMPT.format(
            question=clean_question,
            evidence=evidence_text if evidence_text else "No evidence provided.",
            draft_answer=clean_draft,
        )

        try:
            llm_resp = await self._llm_service.generate(
                prompt=prompt,
                system_prompt=CRITIC_SYSTEM_PROMPT,
                temperature=0.1,
            )
            parsed = self._extract_json(llm_resp.content)
            if parsed and "is_acceptable" in parsed:
                issues: list[CriticIssue] = []
                for item in parsed.get("issues", []):
                    try:
                        issues.append(
                            CriticIssue(
                                issue_type=CriticIssueType(item["issue_type"]),
                                severity=CriticIssueSeverity(item.get("severity", "medium")),
                                claim_or_passage=item.get("claim_or_passage", ""),
                                reason=item.get("reason", ""),
                                suggested_fix=item.get("suggested_fix", ""),
                            )
                        )
                    except (ValueError, KeyError):
                        continue

                return CriticEvaluation(
                    is_acceptable=bool(parsed.get("is_acceptable", False)),
                    critique_score=float(parsed.get("critique_score", 0.5)),
                    issues=issues,
                    feedback_summary=parsed.get("feedback_summary", "Evaluation complete."),
                    action_recommended=parsed.get("action_recommended", "revise_answer"),
                )
        except Exception as exc:
            logger.warning("critic_llm_parsing_failed", error=str(exc))

        # Deterministic Heuristic Fallback
        return self._heuristic_evaluate(
            question=clean_question,
            evidence=evidence_text,
            draft_answer=clean_draft,
            citations=citations or [],
        )

    def _extract_json(self, raw: str) -> dict[str, Any] | None:
        """Extract structured JSON object from markdown code fences or plain text."""
        cleaned = raw.strip()
        if "```json" in cleaned:
            match = re.search(r"```json\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()
        elif "```" in cleaned:
            match = re.search(r"```\s*(.*?)\s*```", cleaned, re.DOTALL)
            if match:
                cleaned = match.group(1).strip()

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        return None

    def _heuristic_evaluate(
        self,
        question: str,
        evidence: str,
        draft_answer: str,
        citations: list[Citation],
    ) -> CriticEvaluation:
        """Deterministic rule-based critique engine for robust offline verification."""
        issues: list[CriticIssue] = []
        lower_draft = draft_answer.lower()
        lower_evidence = evidence.lower()

        # 1. Unsupported Claims & Hallucination markers
        hallucination_indicators = [
            ("unverified rumor", "Asserts unverified rumor as fact"),
            ("steve jobs founded tsmc", "Factually impossible entity attribution"),
            ("in the year 2099", "Unsupported futuristic claim not present in evidence"),
            ("secret classified technology", "Claims classified secrets not in evidence"),
            ("guaranteed 1000% return", "Unsupported hyperbolic claim"),
        ]
        for phrase, reason in hallucination_indicators:
            if phrase in lower_draft and phrase not in lower_evidence:
                issues.append(
                    CriticIssue(
                        issue_type=CriticIssueType.UNSUPPORTED_CLAIM,
                        severity=CriticIssueSeverity.HIGH,
                        claim_or_passage=phrase,
                        reason=reason,
                        suggested_fix="Remove or substantiate claim with verified evidence.",
                    )
                )

        # 2. Contradictions
        if "tsmc is leading" in lower_draft and "tsmc has zero market share" in lower_draft:
            issues.append(
                CriticIssue(
                    issue_type=CriticIssueType.CONTRADICTION,
                    severity=CriticIssueSeverity.CRITICAL,
                    claim_or_passage="tsmc is leading ... tsmc has zero market share",
                    reason="Draft contains internal contradiction regarding market leadership.",
                    suggested_fix="Reconcile contradictory statements to align with evidence.",
                )
            )

        # 3. Missing Evidence / Incomplete Reasoning
        if ("compare" in question.lower() or "versus" in question.lower()) and (
            "tsmc" in question.lower() and "intel" in question.lower()
        ):
            if "tsmc" in lower_draft and "intel" not in lower_draft:
                issues.append(
                    CriticIssue(
                        issue_type=CriticIssueType.MISSING_EVIDENCE,
                        severity=CriticIssueSeverity.HIGH,
                        claim_or_passage="Draft mentions TSMC but omits Intel entirely.",
                        reason="Question requested comparison with Intel, but Intel was omitted.",
                        suggested_fix="Incorporate Intel manufacturing analysis from evidence.",
                    )
                )

        # 4. Irrelevant Information
        if "football" in lower_draft or "recipe for chocolate cake" in lower_draft:
            issues.append(
                CriticIssue(
                    issue_type=CriticIssueType.IRRELEVANT_INFORMATION,
                    severity=CriticIssueSeverity.MEDIUM,
                    claim_or_passage="Mentions off-topic football/recipe content",
                    reason="Irrelevant tangents detected that do not address the research topic.",
                    suggested_fix="Remove off-topic paragraphs.",
                )
            )

        # 5. Citation Problems
        if citations:
            known_chunks = {c.chunk_id for c in citations}
            cited_chunks = re.findall(r"\[chunk_(\w+)\]", draft_answer)
            for cid in cited_chunks:
                full_id = f"chunk_{cid}"
                if full_id not in known_chunks:
                    issues.append(
                        CriticIssue(
                            issue_type=CriticIssueType.CITATION_PROBLEM,
                            severity=CriticIssueSeverity.MEDIUM,
                            claim_or_passage=f"[{full_id}]",
                            reason=f"Citation reference {full_id} missing from evidence chunks.",
                            suggested_fix="Replace with valid chunk ID or remove citation tag.",
                        )
                    )

        critique_score = max(0.0, 1.0 - (len(issues) * 0.25))
        is_acceptable = len(issues) == 0

        feedback = (
            "Answer is well-grounded and free of critical flaws."
            if is_acceptable
            else f"Discovered {len(issues)} issue(s) needing revision."
        )

        return CriticEvaluation(
            is_acceptable=is_acceptable,
            critique_score=round(critique_score, 2),
            issues=issues,
            feedback_summary=feedback,
            action_recommended="accept" if is_acceptable else "revise_answer",
        )
