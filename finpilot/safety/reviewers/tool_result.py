from __future__ import annotations

from finpilot.models import GraphState, ToolInvocation
from finpilot.safety.models import SafetyFinding, SafetyReviewResult
from finpilot.safety.redaction import redact_value


class ToolResultReviewer:
    name = "tool_result"

    def review(self, state: GraphState, invocation: ToolInvocation) -> SafetyReviewResult:
        del state
        redacted = redact_value(invocation.output)
        if redacted != invocation.output:
            return SafetyReviewResult(
                action="REDACT",
                findings=[
                    SafetyFinding(
                        code="TOOL_RESULT_REDACTED",
                        reviewer=self.name,
                        action="REDACT",
                        message="Tool result contained sensitive data and was redacted.",
                        severity="warning",
                    )
                ],
                sanitized_payload=redacted,
            )
        return SafetyReviewResult.allow(invocation.output)
