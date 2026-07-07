from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SafetyAction = Literal["ALLOW", "BLOCK", "REDACT", "ESCALATE", "REQUIRE_APPROVAL"]


class SafetyFinding(BaseModel):
    code: str
    reviewer: str
    action: SafetyAction
    message: str
    severity: Literal["info", "warning", "error"] = "error"
    detail: dict[str, Any] = Field(default_factory=dict)


class SafetyReviewResult(BaseModel):
    action: SafetyAction = "ALLOW"
    findings: list[SafetyFinding] = Field(default_factory=list)
    sanitized_payload: Any | None = None

    @classmethod
    def allow(cls, sanitized_payload: Any | None = None) -> "SafetyReviewResult":
        return cls(action="ALLOW", sanitized_payload=sanitized_payload)
