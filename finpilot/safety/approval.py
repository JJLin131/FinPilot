from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
from typing import Any, Callable, Literal

from finpilot.models import GraphState
from finpilot.safety.models import SafetyFinding
from finpilot.safety.redaction import redact_value

ApprovalScope = Literal["once", "session", "deny"]
ApprovalCallback = Callable[["ApprovalRequest"], "ApprovalDecision"]


@dataclass(frozen=True)
class ApprovalRequest:
    finding: SafetyFinding
    tool_name: str
    parameters: dict[str, Any]
    user_id: str | None = None
    chat_id: str | None = None
    parameter_fingerprint: str | None = None

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
    reused: bool = False
    approval_key: str | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True)
class SessionApproval:
    key: str
    expires_at: datetime


class ApprovalService:
    def __init__(self, callback: ApprovalCallback | None = None, session_ttl: timedelta = timedelta(minutes=30)):
        self.callback = callback
        self.session_ttl = session_ttl
        self.session_approvals: dict[str, SessionApproval] = {}

    def resolve(
        self,
        finding: SafetyFinding,
        *,
        tool_name: str,
        parameters: dict[str, Any],
        state: GraphState | None = None,
    ) -> ApprovalResult:
        fingerprint = self._parameter_fingerprint(parameters)
        user_id = state.user_id if state is not None else ""
        chat_id = state.chat_id if state is not None else ""
        key = self._approval_key(user_id, chat_id, tool_name, finding.code, fingerprint)
        now = datetime.now(UTC)
        cached = self.session_approvals.get(key)
        if cached and cached.expires_at > now:
            return ApprovalResult(approved=True, scope="session", reused=True, approval_key=key, expires_at=cached.expires_at)
        if cached:
            self.session_approvals.pop(key, None)
        if self.callback is None:
            return ApprovalResult(approved=False, scope="deny", approval_key=key)

        request = ApprovalRequest(
            finding=finding,
            tool_name=tool_name,
            parameters=parameters,
            user_id=user_id or None,
            chat_id=chat_id or None,
            parameter_fingerprint=fingerprint,
        )
        decision = self.callback(request)
        expires_at = None
        if decision.scope == "session":
            expires_at = now + self.session_ttl
            self.session_approvals[key] = SessionApproval(key=key, expires_at=expires_at)
        return ApprovalResult(approved=decision.approved, scope=decision.scope, approval_key=key, expires_at=expires_at)

    def _approval_key(self, user_id: str, chat_id: str, tool_name: str, finding_code: str, fingerprint: str) -> str:
        raw = json.dumps(
            {
                "user_id": user_id,
                "chat_id": chat_id,
                "tool_name": tool_name,
                "finding_code": finding_code,
                "parameter_fingerprint": fingerprint,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _parameter_fingerprint(self, parameters: dict[str, Any]) -> str:
        raw = json.dumps(parameters, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
