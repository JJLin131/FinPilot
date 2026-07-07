from __future__ import annotations

from finpilot.models import AgentChatResponse, RouteDecision
from finpilot.responses import prepare_chat_response
from finpilot.safety.models import SafetyFinding


def _response() -> AgentChatResponse:
    return AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="FAILED",
        answer="blocked",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="UNKNOWN",
            reason="blocked",
            confidence=0,
            valid=False,
            target_agent="UNSUPPORTED",
            classifier_intent="UNKNOWN",
        ),
        safety_findings=[
            SafetyFinding(
                code="TOOL_ARGUMENT_VALIDATION_FAILED",
                reviewer="argument",
                action="BLOCK",
                message="invalid arguments",
                detail={"errors": [{"input": "sk-secret", "msg": "internal validation detail"}]},
            )
        ],
    )


def test_prepare_chat_response_scrubs_safety_finding_detail_when_debug_disabled():
    response = prepare_chat_response(_response(), debug_enabled=False)

    assert response.safety_findings[0].detail == {}


def test_prepare_chat_response_keeps_safety_finding_detail_when_debug_enabled():
    response = prepare_chat_response(_response(), debug_enabled=True)

    assert response.safety_findings[0].detail["errors"][0]["input"] == "sk-secret"
