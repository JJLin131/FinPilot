from __future__ import annotations

from finpilot.agent.graph import FinPilotGraph
from finpilot.agent.router import IntentRouter
from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.memory.service import MemoryManager
from finpilot.models import AgentChatResponse
from finpilot.observability import setup_langfuse
from finpilot.observability.audit import AuditStore, build_audit_store
from finpilot.observability.langfuse_support import score_trace
from finpilot.observability.tracing import setup_tracing
from finpilot.rag.service import RagKnowledgeService
from finpilot.reranker import RemoteReranker


class FinPilotService:
    def __init__(self, audit_store: AuditStore | None = None, memory_manager: MemoryManager | None = None):
        setup_tracing()
        setup_langfuse()
        self.audit_store = audit_store or build_audit_store()
        self.memory_manager = memory_manager or MemoryManager()
        self.rag_service = RagKnowledgeService(reranker=RemoteReranker())
        if settings.rag_bootstrap_on_startup:
            self.rag_service.bootstrap_resources()
        self.tools = ToolRegistry(self.rag_service)
        self.router = IntentRouter()
        self.graph = FinPilotGraph(self.router, self.tools, self.audit_store, self.memory_manager)

    def chat(self, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        response = self.graph.run(user_id, chat_id, content)
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

    def shutdown(self) -> None:
        self.memory_manager.shutdown(wait=True)

