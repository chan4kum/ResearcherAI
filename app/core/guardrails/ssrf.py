"""
app/core/guardrails/ssrf.py — Server-Side Request Forgery (SSRF) Protection

Validates outbound URLs for webhooks, external fetch tools, and MCP server
endpoints to prevent access to private networks, loopback addresses, and cloud
metadata endpoints (e.g. AWS IMDS / GCP metadata).
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

# Prohibited hostnames and domain suffixes
FORBIDDEN_HOSTNAMES: set[str] = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "metadata.internal",
}

FORBIDDEN_DOMAIN_SUFFIXES: tuple[str, ...] = (
    ".internal",
    ".local",
    ".corp",
    ".lan",
    ".localhost",
)

# Allowed URL schemes
ALLOWED_SCHEMES: set[str] = {"http", "https"}


class SSRFValidationError(ValueError):
    """Raised when a URL fails SSRF safety validation."""
    pass


def is_ip_prohibited(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> tuple[bool, str | None]:
    """Verify whether an IP address belongs to private, loopback, or metadata ranges."""
    if ip.is_loopback:
        return (True, f"Loopback IP address prohibited: {ip}")
    if ip.is_link_local or str(ip).startswith("169.254."):
        return (True, f"Link-local / Cloud metadata IP address prohibited: {ip}")
    if ip.is_private:
        return (True, f"Private RFC1918 / RFC4193 IP address prohibited: {ip}")
    if ip.is_multicast:
        return (True, f"Multicast IP address prohibited: {ip}")
    if ip.is_reserved:
        return (True, f"Reserved IP address prohibited: {ip}")
    if ip.is_unspecified:
        return (True, f"Unspecified 0.0.0.0 IP address prohibited: {ip}")
    return (False, None)


def validate_safe_url(
    url: str,
    allowed_schemes: set[str] | None = None,
    allow_private_ips: bool = False,
) -> tuple[bool, str | None]:
    """Validate a URL against SSRF vulnerabilities.

    Returns:
        (is_safe: bool, reason: str | None)
    """
    if not url or not isinstance(url, str):
        return (False, "URL must be a non-empty string.")

    cleaned_url = url.strip()
    schemes = allowed_schemes or ALLOWED_SCHEMES

    try:
        parsed = urlparse(cleaned_url)
    except Exception as exc:
        return (False, f"Malformed URL syntax: {exc}")

    # 1. Validate Scheme
    if not parsed.scheme or parsed.scheme.lower() not in schemes:
        return (
            False,
            f"Prohibited URL scheme '{parsed.scheme}'. Only {sorted(schemes)} are permitted.",
        )

    # 2. Validate Host
    hostname = parsed.hostname
    if not hostname:
        return (False, "URL is missing a valid hostname.")

    hostname_lower = hostname.lower().strip(".")

    # 3. Check Forbidden Hostnames
    if hostname_lower in FORBIDDEN_HOSTNAMES:
        return (False, f"Prohibited destination host '{hostname}'.")

    for suffix in FORBIDDEN_DOMAIN_SUFFIXES:
        if hostname_lower.endswith(suffix):
            return (False, f"Prohibited internal domain suffix in '{hostname}'.")

    # 4. Check Literal IP addresses
    try:
        # Check if hostname is directly an IPv4/IPv6 literal
        ip_obj = ipaddress.ip_address(hostname_lower)
        if not allow_private_ips:
            is_bad, reason = is_ip_prohibited(ip_obj)
            if is_bad:
                return (False, reason)
    except ValueError:
        # Hostname is a domain name, not an IP literal.
        pass

    return (True, None)


def ensure_safe_url(url: str, allow_private_ips: bool = False) -> str:
    """Validate URL and return it, or raise SSRFValidationError on failure."""
    is_safe, reason = validate_safe_url(url, allow_private_ips=allow_private_ips)
    if not is_safe:
        raise SSRFValidationError(f"SSRF Security Violation: {reason}")
    return url.strip()
