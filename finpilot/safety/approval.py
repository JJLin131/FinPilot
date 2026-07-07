from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from finpilot.safety.models import SafetyFinding
from finpilot.safety.redaction import redact_value

ApprovalScope = Literal["once", "session", "deny"]
ApprovalCallback = Callable[["ApprovalRequest"], "ApprovalDecision"]


@dataclass(frozen=True)
class ApprovalRequest:
    finding: SafetyFinding
    tool_name: str
    parameters: dict[str, Any]

    @property
    def parameter_summary(self) -> dict[str, Any]:
        return redact_value(self.parameters)


@dataclass(frozen=True)
class ApprovalDecision:
    scope: ApprovalScope

    @property
    def approved(self) -> bool:
        return self.scope in {"once", "session"}


@dataclass
class ApprovalResult:
    approved: bool
    scope: ApprovalScope


class ApprovalService:
    def __init__(self, callback: ApprovalCallback | None = None):
        self.callback = callback
        self.session_approvals: set[tuple[str, str]] = set()

    def resolve(self, finding: SafetyFinding, *, tool_name: str, parameters: dict[str, Any]) -> ApprovalResult:
        key = (tool_name, finding.code)
        if key in self.session_approvals:
            return ApprovalResult(approved=True, scope="session")
        if self.callback is None:
            return ApprovalResult(approved=False, scope="deny")

        decision = self.callback(ApprovalRequest(finding=finding, tool_name=tool_name, parameters=parameters))
        if decision.scope == "session":
            self.session_approvals.add(key)
        return ApprovalResult(approved=decision.approved, scope=decision.scope)
