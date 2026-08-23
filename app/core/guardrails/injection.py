"""
app/core/guardrails/injection.py — Prompt Injection & Jailbreak Guard

Detects and neutralizes:
- Direct prompt injections (instruction overrides, system prompt manipulation)
- Jailbreaks and persona switching (DAN mode, developer mode, safety bypasses)
- System prompt extraction and leakage attacks
- Chat template delimiter attacks (<|im_start|>, [INST], <<SYS>>)
- Indirect prompt injections in ingested or retrieved documents
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InjectionThreat(str, Enum):
    DIRECT_INJECTION = "direct_injection"
    INDIRECT_INJECTION = "indirect_injection"
    JAILBREAK = "jailbreak"
    SYSTEM_EXTRACTION = "system_extraction"
    DELIMITER_ATTACK = "delimiter_attack"


class InjectionScanResult(BaseModel):
    """Result of an injection and safety scan."""

    is_threat: bool = Field(description="Whether a prompt injection or safety threat was detected")
    threat_type: InjectionThreat | None = Field(default=None, description="Class of threat detected")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Heuristic confidence score")
    matched_pattern: str | None = Field(default=None, description="Pattern or keyword that triggered")
    reason: str | None = Field(default=None, description="Human-readable explanation")
    sanitized_text: str = Field(description="Sanitized version of the input text")


# Direct instruction override patterns
DIRECT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|directives|prompts|rules)"),
    re.compile(r"(?i)\b(?:you\s+must\s+now|from\s+now\s+on\s+you\s+are)\s+(?:a\s+different|an\s+unrestricted|a\s+new)\s+(?:model|ai|agent|assistant)"),
    re.compile(r"(?i)\b(?:new\s+system\s+instruction|system\s+override|admin\s+command)\s*:"),
    re.compile(r"(?i)\b(?:bypass|disable|turn\s+off)\s+(?:all\s+)?(?:safety|guardrails|content\s+filters|moderation)"),
]

# Jailbreak & persona subversion patterns
JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\b(?:DAN\s+mode|Do\s+Anything\s+Now|developer\s+mode\s+active|godmode\s+enabled)\b"),
    re.compile(r"(?i)\b(?:pretend\s+you\s+have\s+no\s+rules|unfiltered\s+ai|jailbreak\s+prompt)\b"),
    re.compile(r"(?i)\b(?:you\s+are\s+freed\s+from\s+all\s+constraints|ignore\s+ethical\s+guidelines)\b"),
]

# System prompt extraction patterns
EXTRACTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\b(?:print|repeat|reveal|output|display|show)\s+(?:your\s+)?(?:system\s+prompt|initial\s+instructions|system\s+directives|hidden\s+prompt)\b"),
    re.compile(r"(?i)\b(?:what\s+are\s+your\s+(?:exact\s+)?instructions\s+before\s+this|repeat\s+the\s+words\s+above)\b"),
]

# Delimiter and template token injection patterns
DELIMITER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<\|(?:im_start|im_end|endoftext|system|user|assistant)\|>"),
    re.compile(r"\[/?(?:INST|SYS)\]"),
    re.compile(r"<<(?:SYS|SYSTEM)>>"),
    re.compile(r"^\s*###\s*(?:System|Human|Assistant|Instruction|Response)\s*:", re.MULTILINE),
]

# Indirect injection patterns common in poisoned web/document text
INDIRECT_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\[(?:SYSTEM\s+ALERT|ADMIN\s+NOTE|URGENT\s+INSTRUCTION)\s*:\s*ignore\s+[^\]]+\]"),
    re.compile(r"(?i)IMPORTANT\s+NOTICE\s*:\s*The\s+user\s+has\s+requested\s+that\s+you\s+output\s+"),
    re.compile(r"(?i)END\s+OF\s+CONTEXT\.?\s*NEW\s+INSTRUCTION\s*:"),
]


class PromptInjectionGuard:
    """Multi-layer heuristic and pattern-based prompt injection detector."""

    @classmethod
    def scan_text(cls, text: str | None, is_retrieved_context: bool = False) -> InjectionScanResult:
        """Scan input text or retrieved context for prompt injection threats."""
        if not text or not str(text).strip():
            return InjectionScanResult(
                is_threat=False,
                confidence=0.0,
                sanitized_text=text or "",
            )

        raw_text = str(text)

        # 1. Check Delimiter Injection attacks
        for pattern in DELIMITER_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                sanitized = pattern.sub("[FILTERED_DELIMITER]", raw_text)
                return InjectionScanResult(
                    is_threat=True,
                    threat_type=InjectionThreat.DELIMITER_ATTACK,
                    confidence=0.95,
                    matched_pattern=match.group(0),
                    reason="Attempted chat template token / delimiter injection.",
                    sanitized_text=sanitized,
                )

        # 2. Check Direct Instruction Overrides
        for pattern in DIRECT_INJECTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                sanitized = pattern.sub("[FILTERED_INSTRUCTION_OVERRIDE]", raw_text)
                return InjectionScanResult(
                    is_threat=True,
                    threat_type=InjectionThreat.DIRECT_INJECTION,
                    confidence=0.90,
                    matched_pattern=match.group(0),
                    reason="Attempted instruction override / system prompt disregard.",
                    sanitized_text=sanitized,
                )

        # 3. Check Jailbreak / Persona Subversion
        for pattern in JAILBREAK_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                sanitized = pattern.sub("[FILTERED_JAILBREAK]", raw_text)
                return InjectionScanResult(
                    is_threat=True,
                    threat_type=InjectionThreat.JAILBREAK,
                    confidence=0.90,
                    matched_pattern=match.group(0),
                    reason="Attempted jailbreak or safety constraint subversion.",
                    sanitized_text=sanitized,
                )

        # 4. Check System Prompt Extraction
        for pattern in EXTRACTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                return InjectionScanResult(
                    is_threat=True,
                    threat_type=InjectionThreat.SYSTEM_EXTRACTION,
                    confidence=0.85,
                    matched_pattern=match.group(0),
                    reason="Attempted system prompt / instruction extraction.",
                    sanitized_text=raw_text,
                )

        # 5. Check Indirect Injection (especially relevant for retrieved docs)
        for pattern in INDIRECT_INJECTION_PATTERNS:
            match = pattern.search(raw_text)
            if match:
                sanitized = pattern.sub("[FILTERED_INDIRECT_INJECTION]", raw_text)
                return InjectionScanResult(
                    is_threat=True,
                    threat_type=InjectionThreat.INDIRECT_INJECTION,
                    confidence=0.90,
                    matched_pattern=match.group(0),
                    reason="Indirect prompt injection detected in retrieved text.",
                    sanitized_text=sanitized,
                )

        return InjectionScanResult(
            is_threat=False,
            confidence=0.0,
            sanitized_text=raw_text,
        )

    @classmethod
    def sanitize_retrieved_context(cls, context_text: str) -> str:
        """Neutralize instruction delimiters in retrieved context to treat as passive data."""
        if not context_text:
            return ""

        # First scan for indirect injection patterns
        scan = cls.scan_text(context_text, is_retrieved_context=True)
        sanitized = scan.sanitized_text

        # Neutralize markdown instruction headers
        sanitized = re.sub(
            r"(?i)^\s*(###\s*(?:Instruction|System|Prompt|Task|Role|Command|Directive)\s*:)",
            r"[Data Section: \1]",
            sanitized,
            flags=re.MULTILINE,
        )

        return sanitized
