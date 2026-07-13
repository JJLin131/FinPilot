from __future__ import annotations

from finpilot.agent.graph import FinPilotGraph
from finpilot.agent.runtime_events import AgentRuntimeEventCallback
from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.context.compression import ContextEventCallback
from finpilot.memory.service import MemoryManager
from finpilot.models import AgentChatResponse
from finpilot.observability import setup_langfuse
from finpilot.observability.audit import AuditStore, build_audit_store
from finpilot.observability.langfuse_support import score_trace
from finpilot.observability.tracing import setup_tracing
from finpilot.rag.service import RagKnowledgeService
from finpilot.reranker import RemoteReranker
from finpilot.safety.service import SafetyReviewService


class FinPilotService:
    def __init__(
        self,
        audit_store: AuditStore | None = None,
        memory_manager: MemoryManager | None = None,
        safety: SafetyReviewService | None = None,
        context_event_callback: ContextEventCallback | None = None,
        runtime_event_callback: AgentRuntimeEventCallback | None = None,
    ):
        setup_tracing()
        setup_langfuse()
        self.audit_store = audit_store or build_audit_store()
        self.memory_manager = memory_manager or MemoryManager()
        self.safety = safety or SafetyReviewService()
        self.rag_service = RagKnowledgeService(reranker=RemoteReranker())
        if settings.rag_bootstrap_on_startup:
            self.rag_service.bootstrap_resources()
        self.tools = ToolRegistry(self.rag_service, safety=self.safety)
        self.graph = FinPilotGraph(
            tools=self.tools,
            audit_store=self.audit_store,
            memory_manager=self.memory_manager,
            safety=self.safety,
            context_event_callback=context_event_callback,
            runtime_event_callback=runtime_event_callback,
        )

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
            name="planner.unsupported_rate",
            value=1.0 if response.status == "UNSUPPORTED" else 0.0,
            metadata={"planning_status": response.plan.get("status"), "status": response.status},
        )
        score_trace(
            response.trace_id,
            name="runtime.issue_count",
            value=float(len(response.issues)),
            metadata={"codes": [issue.code for issue in response.issues], "status": response.status},
        )
        score_trace(
            response.trace_id,
            name="runtime.faithfulness_proxy",
            value=1.0 if response.evidence else 0.4,
            metadata={"evidence_count": len(response.evidence)},
        )
        return response

    def evaluate_tool_selection(self, user_id: str, chat_id: str, content: str) -> dict:
        return self.graph.evaluate_tool_selection(user_id, chat_id, content)

    def shutdown(self) -> None:
        self.memory_manager.shutdown(wait=True)

