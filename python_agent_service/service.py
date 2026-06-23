from __future__ import annotations

from python_agent_service.audit import AuditStore, build_audit_store
from python_agent_service.graph import FinanceAgentGraph
from python_agent_service.langfuse_support import score_trace
from python_agent_service.models import AgentChatResponse
from python_agent_service.observability import setup_langfuse
from python_agent_service.rag import RagKnowledgeService
from python_agent_service.router import IntentRouter
from python_agent_service.tools import ToolRegistry
from python_agent_service.tracing import setup_tracing


class FinanceAgentService:
    def __init__(self, audit_store: AuditStore | None = None):
        setup_tracing()
        setup_langfuse()
        self.audit_store = audit_store or build_audit_store()
        self.rag_service = RagKnowledgeService()
        self.tools = ToolRegistry(self.rag_service)
        self.router = IntentRouter()
        self.graph = FinanceAgentGraph(self.router, self.tools, self.audit_store)

    def chat(self, tenant_id: str, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        response = self.graph.run(tenant_id, user_id, chat_id, content)
        score_trace(
            response.trace_id,
            name="runtime.tool_success",
            value=1.0 if all(tool.status == "SUCCEEDED" for tool in (response.tool_calls or [])) else 0.0,
            metadata={"tool_count": len(response.tool_calls or [])},
        )
        score_trace(
            response.trace_id,
            name="runtime.unknown_rate",
            value=1.0 if response.route.normalized_intent == "UNKNOWN" else 0.0,
            metadata={"intent": response.route.normalized_intent},
        )
        score_trace(
            response.trace_id,
            name="runtime.faithfulness_proxy",
            value=1.0 if response.evidence else 0.4,
            metadata={"evidence_count": len(response.evidence)},
        )
        return response
