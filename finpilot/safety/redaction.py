from __future__ import annotations

import re
from typing import Any

SENSITIVE_FIELD_NAMES = {
    "access_token",
    "account_no",
    "api_key",
    "card_no",
    "credential",
    "password",
    "secret",
    "token",
}

PATTERNS = [
    re.compile(r"\b1[3-9]\d{9}\b"),
    re.compile(r"\b\d{15}(\d{2}[\dXx])?\b"),
    re.compile(r"\b\d{16,19}\b"),
    re.compile(r"\b(sk|pk|ak)-[A-Za-z0-9_\-]{8,}\b"),
]


def redact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _is_sensitive_field(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, str):
        return _redact_text(value)
    return value


def has_redaction(value: Any) -> bool:
    return redact_value(value) != value


def _is_sensitive_field(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return normalized in SENSITIVE_FIELD_NAMES or normalized.endswith("_token") or normalized.endswith("_secret")


def _redact_text(text: str) -> str:
    redacted = text
    for pattern in PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
