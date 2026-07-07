from __future__ import annotations

from typing import Any

from finpilot.models import GraphState
from finpilot.safety.models import SafetyFinding, SafetyReviewResult


class OperationRiskReviewer:
    name = "operation_risk"

    def review(
        self,
        state: GraphState,
        spec,
        parameters: dict[str, Any],
        reason: str | None = None,
    ) -> SafetyReviewResult:
        del state, parameters, reason
        tool_name = str(getattr(spec, "name", "unknown_tool"))
        risk_level = str(getattr(spec, "risk_level", "low")).lower()
        if risk_level == "high":
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
                            "risk_level": risk_level,
                        },
                    )
                ],
            )
        return SafetyReviewResult.allow()
