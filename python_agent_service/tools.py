from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from python_agent_service.models import AgentEvidence, GraphState, ToolInvocation
from python_agent_service.rag import RagKnowledgeService


@dataclass
class ToolResult:
    answer_fragment: str
    evidence: AgentEvidence | None = None
    extra: dict[str, Any] | None = None


class ToolRegistry:
    def __init__(self, rag_service: RagKnowledgeService):
        self.rag_service = rag_service

    def invoke(self, state: GraphState, tool_name: str, **parameters: Any) -> ToolInvocation:
        started = time.perf_counter()
        output: dict[str, Any]
        status = "SUCCEEDED"
        if tool_name == "query_account_balance":
            output = {"available_balance": 182340.25, "currency": "CNY"}
        elif tool_name == "check_transfer_status":
            output = {"transfer_status": "PROCESSING", "eta": "2h"}
        elif tool_name == "get_transfer_receipt":
            output = {"receipt_id": f"rcpt-{state.chat_id}", "download_url": f"/receipts/{state.chat_id}.pdf"}
        elif tool_name == "cancel_transfer":
            output = {"cancellation_status": "SUBMITTED", "reference": f"cancel-{state.chat_id}"}
        elif tool_name == "search_finance_knowledge":
            docs = [match.model_dump() for match in self.rag_service.search(state.user_message, limit=3)]
            output = {"documents": docs}
        else:
            status = "FAILED"
            output = {"error": f"Unsupported tool: {tool_name}"}
        duration_ms = int((time.perf_counter() - started) * 1000)
        return ToolInvocation(
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            duration_ms=duration_ms,
            output=output,
        )

