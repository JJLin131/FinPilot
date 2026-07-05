from __future__ import annotations

from finpilot.issues import scrub_issue_details
from finpilot.models import AgentChatResponse


def prepare_chat_response(response: AgentChatResponse, *, debug_enabled: bool) -> AgentChatResponse:
    if not debug_enabled:
        # API 和 CLI 共用同一处脱敏逻辑，避免不同入口暴露不一致。
        response.route_debug = None
        response.retrieval_debug = None
        response.tool_calls = None
        response.issues = scrub_issue_details(response.issues)
    return response
