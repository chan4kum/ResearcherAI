"""
app/core/guardrails/secrets_filter.py — Cryptographic and Credential Data Leakage Scrubber

Scans and sanitizes strings, dictionaries, logs, and LLM completions to prevent
accidental leakage of:
- OpenAI, Anthropic, and generic API keys
- AWS credentials (AKIA / Secret Keys)
- GitHub Personal Access Tokens (classic and fine-grained)
- JWT bearer tokens
- Private encryption keys (RSA, EC, OPENSSH)
- Database connection strings with embedded passwords
"""

from __future__ import annotations

import re
from typing import Any

# Comprehensive regex patterns for high-entropy secrets and credentials
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1. OpenAI API Keys (legacy 51-char and newer project keys)
    ("OPENAI_KEY", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,80}")),
    # 2. Anthropic API Keys
    ("ANTHROPIC_KEY", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,80}")),
    # 3. AWS Access Key ID
    ("AWS_ACCESS_KEY", re.compile(r"(?:A3T[A-Z0-9]|AKIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}")),
    # 4. GitHub Tokens
    ("GITHUB_TOKEN", re.compile(r"(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})")),
    # 5. JWT Bearer Tokens
    ("JWT_TOKEN", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_\-+/=]{10,}")),
    # 6. Private Key Headers
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    # 7. Database URLs with passwords (postgres, mysql, redis, mongodb)
    ("DATABASE_URL_PASSWORD", re.compile(r"(?:postgres(?:ql)?|mysql|redis|mongodb):\/\/[^:\s]+:([^@\s]+)@", re.IGNORECASE)),
    # 8. Generic Bearer Token Headers in text
    ("BEARER_AUTH", re.compile(r"(?i)bearer\s+[A-Za-z0-9\-\._~\+\/]{16,}={0,2}")),
]


class SecretsScrubber:
    """Scrubber removing credential patterns from inputs, outputs, errors, and traces."""

    @classmethod
    def scrub_text(cls, text: str | None) -> str:
        """Replace all recognized secret patterns with sanitized redact tags."""
        if not text:
            return ""

        scrubbed = str(text)
        for name, pattern in SECRET_PATTERNS:
            if name == "DATABASE_URL_PASSWORD":
                scrubbed = pattern.sub(r"://\g<0>".replace(r"\g<0>", "***REDACTED_PASSWORD***@"), scrubbed)
            elif name == "PRIVATE_KEY":
                scrubbed = pattern.sub("[REDACTED_PRIVATE_KEY]", scrubbed)
            else:
                scrubbed = pattern.sub(f"[REDACTED_{name}]", scrubbed)

        return scrubbed

    @classmethod
    def scrub_data(cls, data: Any) -> Any:
        """Recursively scrub strings within nested dictionaries and lists."""
        if isinstance(data, str):
            return cls.scrub_text(data)
        if isinstance(data, dict):
            return {cls.scrub_text(str(k)): cls.scrub_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [cls.scrub_data(item) for item in data]
        if isinstance(data, tuple):
            return tuple(cls.scrub_data(item) for item in data)
        return data

    @classmethod
    def contains_secrets(cls, text: str | None) -> bool:
        """Check if a string contains any recognized secret pattern."""
        if not text:
            return False
        for _, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                return True
        return False
