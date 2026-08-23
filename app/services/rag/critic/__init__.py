"""Critic Agent and Self-Correction package for detecting flaws and refining answers."""

from app.services.rag.critic.agent import CriticAgent
from app.services.rag.critic.engine import SelfCorrectionEngine
from app.services.rag.critic.models import (
    CriticEvaluation,
    CriticIssue,
    CriticIssueSeverity,
    CriticIssueType,
    SelfCorrectionAttempt,
    SelfCorrectionResult,
)

__all__ = [
    "CriticAgent",
    "CriticEvaluation",
    "CriticIssue",
    "CriticIssueSeverity",
    "CriticIssueType",
    "SelfCorrectionAttempt",
    "SelfCorrectionEngine",
    "SelfCorrectionResult",
]
