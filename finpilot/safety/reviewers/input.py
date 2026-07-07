from __future__ import annotations

from finpilot.models import GraphState
from finpilot.safety.models import SafetyFinding, SafetyReviewResult


class InputSafetyReviewer:
    name = "input"
    blocked_phrases = (
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "绕过安全",
        "忽略之前",
        "忽略所有规则",
        "系统提示",
        "提示词",
    )

    def review(self, state: GraphState) -> SafetyReviewResult:
        message = state.user_message.lower()
        if any(phrase in message for phrase in self.blocked_phrases):
            return SafetyReviewResult(
                action="BLOCK",
                findings=[
                    SafetyFinding(
                        code="INPUT_PROMPT_INJECTION_BLOCKED",
                        reviewer=self.name,
                        action="BLOCK",
                        message="Input was blocked by the safety policy.",
                    )
                ],
            )
        return SafetyReviewResult.allow()
