from __future__ import annotations

from typing import Any

from finpilot.models import GraphState
from finpilot.safety.models import SafetyFinding, SafetyReviewResult


class OperationRiskReviewer:
    name = "operation_risk"
    risky_prefixes = ("transfer_", "write_", "delete_", "export_")
    risky_names = {"create_payment"}

    def review(
        self,
        state: GraphState,
        tool_name: str,
        parameters: dict[str, Any],
        reason: str | None = None,
    ) -> SafetyReviewResult:
        del state, parameters, reason
        if tool_name.startswith(self.risky_prefixes) or tool_name in self.risky_names:
            return SafetyReviewResult(
                action="REQUIRE_APPROVAL",
                findings=[
                    SafetyFinding(
                        code="TOOL_OPERATION_REQUIRES_APPROVAL",
                        reviewer=self.name,
                        action="REQUIRE_APPROVAL",
                        message="This tool call is a high-risk operation and requires user approval.",
                        severity="warning",
                        detail={"tool_name": tool_name},
                    )
                ],
            )
        return SafetyReviewResult.allow()
