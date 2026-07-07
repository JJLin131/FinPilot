from __future__ import annotations

from typing import Any

from finpilot.config import settings
from finpilot.models import GraphState
from finpilot.safety.models import SafetyFinding, SafetyReviewResult


class OperationRiskReviewer:
    name = "operation_risk"
    default_policies: dict[str, dict[str, Any]] = {
        "transfer_*": {"risk_level": "high", "approval_required": True},
        "write_*": {"risk_level": "high", "approval_required": True},
        "delete_*": {"risk_level": "high", "approval_required": True},
        "export_*": {"risk_level": "high", "approval_required": True},
        "create_payment": {"risk_level": "high", "approval_required": True},
    }

    def __init__(self, policies: dict[str, dict[str, Any]] | None = None):
        configured = getattr(settings, "tool_risk_policies", {}) or {}
        self.policies = {**self.default_policies, **configured, **(policies or {})}

    def review(
        self,
        state: GraphState,
        tool_name: str,
        parameters: dict[str, Any],
        reason: str | None = None,
    ) -> SafetyReviewResult:
        del state, parameters, reason
        policy = self._policy_for(tool_name)
        if policy and bool(policy.get("approval_required")):
            return SafetyReviewResult(
                action="REQUIRE_APPROVAL",
                findings=[
                    SafetyFinding(
                        code="TOOL_OPERATION_REQUIRES_APPROVAL",
                        reviewer=self.name,
                        action="REQUIRE_APPROVAL",
                        message="This tool call is a high-risk operation and requires user approval.",
                        severity="warning",
                        detail={
                            "tool_name": tool_name,
                            "risk_level": str(policy.get("risk_level") or "high"),
                            "policy": policy,
                        },
                    )
                ],
            )
        return SafetyReviewResult.allow()

    def _policy_for(self, tool_name: str) -> dict[str, Any] | None:
        if tool_name in self.policies:
            return self.policies[tool_name]
        for pattern, policy in self.policies.items():
            if pattern.endswith("*") and tool_name.startswith(pattern[:-1]):
                return policy
        return None
