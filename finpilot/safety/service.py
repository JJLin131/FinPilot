from __future__ import annotations

from typing import Any

from finpilot.models import GraphState, ToolInvocation
from finpilot.safety.approval import ApprovalService
from finpilot.safety.models import SafetyFinding, SafetyReviewResult
from finpilot.safety.reviewers.argument import ArgumentReviewer
from finpilot.safety.reviewers.input import InputSafetyReviewer
from finpilot.safety.reviewers.operation_risk import OperationRiskReviewer
from finpilot.safety.reviewers.response import ResponseReviewer
from finpilot.safety.reviewers.tool_result import ToolResultReviewer

ACTION_PRIORITY = {
    "ALLOW": 0,
    "REDACT": 1,
    "ESCALATE": 2,
    "REQUIRE_APPROVAL": 3,
    "BLOCK": 4,
}


class SafetyReviewService:
    def __init__(
        self,
        *,
        argument_reviewer: ArgumentReviewer | None = None,
        operation_risk_reviewer: OperationRiskReviewer | None = None,
        input_reviewer: InputSafetyReviewer | None = None,
        tool_result_reviewer: ToolResultReviewer | None = None,
        response_reviewer: ResponseReviewer | None = None,
        approval_service: ApprovalService | None = None,
        interactive_approval: bool = False,
    ):
        self.argument_reviewer = argument_reviewer or ArgumentReviewer()
        self.operation_risk_reviewer = operation_risk_reviewer or OperationRiskReviewer()
        self.input_reviewer = input_reviewer or InputSafetyReviewer()
        self.tool_result_reviewer = tool_result_reviewer or ToolResultReviewer()
        self.response_reviewer = response_reviewer or ResponseReviewer()
        self.approval_service = approval_service or ApprovalService()
        self.interactive_approval = interactive_approval

    def review_input(self, state: GraphState) -> SafetyReviewResult:
        return self.input_reviewer.review(state)

    def review_tool_call(self, state: GraphState, spec, parameters: dict[str, Any], reason: str | None) -> SafetyReviewResult:
        argument_result = self.argument_reviewer.review(spec, parameters)
        if argument_result.action != "ALLOW":
            return argument_result
        validated = argument_result.sanitized_payload or {}
        risk_result = self.operation_risk_reviewer.review(state, spec.name, validated, reason)
        if risk_result.action == "REQUIRE_APPROVAL":
            if not self.interactive_approval or not risk_result.findings:
                return self._with_action(risk_result, "BLOCK")
            approval = self.approval_service.resolve(risk_result.findings[0], tool_name=spec.name, parameters=validated)
            if not approval.approved:
                return self._with_action(risk_result, "BLOCK")
        elif risk_result.action == "ESCALATE":
            return self._with_action(risk_result, "BLOCK")
        elif risk_result.action != "ALLOW":
            return risk_result
        return SafetyReviewResult.allow(validated)

    def review_tool_result(self, state: GraphState, invocation: ToolInvocation) -> SafetyReviewResult:
        return self.tool_result_reviewer.review(state, invocation)

    def review_response(self, state: GraphState) -> SafetyReviewResult:
        result = self.response_reviewer.review(state)
        if result.action == "ESCALATE":
            return self._with_action(result, "BLOCK")
        return result

    def _with_action(self, result: SafetyReviewResult, action) -> SafetyReviewResult:
        return SafetyReviewResult(
            action=action,
            findings=[
                SafetyFinding(
                    code=finding.code,
                    reviewer=finding.reviewer,
                    action=action,
                    message=finding.message,
                    severity=finding.severity,
                    detail=finding.detail,
                )
                for finding in result.findings
            ],
            sanitized_payload=result.sanitized_payload,
        )


def merge_review_results(results: list[SafetyReviewResult]) -> SafetyReviewResult:
    if not results:
        return SafetyReviewResult.allow()
    action = max((result.action for result in results), key=lambda item: ACTION_PRIORITY[item])
    findings = [finding for result in results for finding in result.findings]
    payload = next((result.sanitized_payload for result in reversed(results) if result.sanitized_payload is not None), None)
    return SafetyReviewResult(action=action, findings=findings, sanitized_payload=payload)
