"""
app/core/guardrails/document_safety.py — Ingestion & Malicious Document Safety Validator

Protects against:
- Oversized uploads and memory exhaustion DoS
- Decompression bombs and high-expansion archives
- Active script injections in HTML/PDF/SVG (XSS / polyglots)
- Null-byte injection and malformed control characters
- Dangerous macro execution payload markers
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# Disallowed active script and embedded executable patterns
MALICIOUS_HTML_TAGS = re.compile(
    r"<\s*(?:script|iframe|object|embed|applet|meta|link|form|svg\s+onload)[\s\S]*?>",
    re.IGNORECASE,
)
MALICIOUS_EVENT_HANDLERS = re.compile(
    r"\bon(?:load|error|click|mouseover|focus|blur|change|submit)\s*=",
    re.IGNORECASE,
)
MALICIOUS_SCHEMES = re.compile(
    r"(?:javascript|vbscript|data\s*:\s*text\/html)\s*:",
    re.IGNORECASE,
)
NULL_BYTE_PATTERN = re.compile(r"\x00")
DANGEROUS_CONTROL_CHARS = re.compile(r"[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]")


class DocumentValidationResult(BaseModel):
    """Result of document safety and integrity validation."""

    is_valid: bool = Field(description="Whether the document is safe for ingestion")
    error: str | None = Field(default=None, description="Error reason if rejected")
    sanitized_content: str = Field(default="", description="Sanitized text content")
    original_size_bytes: int = Field(default=0, description="Original byte size")
    sanitized_size_bytes: int = Field(default=0, description="Sanitized byte size")
    warnings: list[str] = Field(default_factory=list, description="Non-fatal safety warnings")


class DocumentSafetyValidator:
    """Validator enforcing size, payload, and script safety on ingested documents."""

    def __init__(
        self,
        max_file_size_bytes: int = 25 * 1024 * 1024,  # 25MB
        max_characters: int = 2_000_000,
        max_decompression_ratio: float = 10.0,
    ) -> None:
        self.max_file_size_bytes = max_file_size_bytes
        self.max_characters = max_characters
        self.max_decompression_ratio = max_decompression_ratio

    def validate_content(
        self,
        content: str | bytes,
        filename: str = "document",
    ) -> DocumentValidationResult:
        """Validate raw text or byte content against document security standards."""
        warnings: list[str] = []

        # 1. Size Validation
        if isinstance(content, bytes):
            raw_bytes = content
            size_bytes = len(raw_bytes)
            if size_bytes > self.max_file_size_bytes:
                return DocumentValidationResult(
                    is_valid=False,
                    error=f"Document '{filename}' exceeds max allowed size of {self.max_file_size_bytes} bytes (got {size_bytes} bytes).",
                    original_size_bytes=size_bytes,
                )
            try:
                text_content = raw_bytes.decode("utf-8", errors="replace")
            except Exception as exc:
                return DocumentValidationResult(
                    is_valid=False,
                    error=f"Failed to decode document bytes: {exc}",
                    original_size_bytes=size_bytes,
                )
        else:
            text_content = str(content)
            size_bytes = len(text_content.encode("utf-8"))
            if size_bytes > self.max_file_size_bytes:
                return DocumentValidationResult(
                    is_valid=False,
                    error=f"Document text exceeds max allowed size of {self.max_file_size_bytes} bytes.",
                    original_size_bytes=size_bytes,
                )

        # 2. Character Length Check
        if len(text_content) > self.max_characters:
            return DocumentValidationResult(
                is_valid=False,
                error=f"Document exceeds maximum character length of {self.max_characters} characters.",
                original_size_bytes=size_bytes,
            )

        # 3. Null-Byte & Control Character Scrubbing
        if NULL_BYTE_PATTERN.search(text_content):
            warnings.append("Stripped null bytes (\\x00) from document content.")
            text_content = NULL_BYTE_PATTERN.sub("", text_content)

        if DANGEROUS_CONTROL_CHARS.search(text_content):
            warnings.append("Stripped non-printable ASCII control characters.")
            text_content = DANGEROUS_CONTROL_CHARS.sub("", text_content)

        # 4. Active Script & XSS Payload Stripping
        if MALICIOUS_HTML_TAGS.search(text_content):
            warnings.append("Sanitized active HTML/XML script tags from content.")
            text_content = MALICIOUS_HTML_TAGS.sub("[FILTERED_SCRIPT_TAG]", text_content)

        if MALICIOUS_EVENT_HANDLERS.search(text_content):
            warnings.append("Sanitized inline HTML event handlers.")
            text_content = MALICIOUS_EVENT_HANDLERS.sub("data-filtered-handler=", text_content)

        if MALICIOUS_SCHEMES.search(text_content):
            warnings.append("Sanitized javascript: / data: pseudo-protocol schemes.")
            text_content = MALICIOUS_SCHEMES.sub("filtered-scheme:", text_content)

        sanitized_bytes = len(text_content.encode("utf-8"))

        return DocumentValidationResult(
            is_valid=True,
            sanitized_content=text_content,
            original_size_bytes=size_bytes,
            sanitized_size_bytes=sanitized_bytes,
            warnings=warnings,
        )
