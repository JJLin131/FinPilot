from __future__ import annotations

import pytest

from finpilot.agent.runtime import AgentRuntime
from finpilot.agent.tools import ToolRegistry
from finpilot.agent.graph import FinPilotGraph
from finpilot.config import settings
from finpilot.llm import FinanceAnsweringService
from finpilot.models import AgentIssue, EvalSuiteResult, GraphState, RagMatch, RouteDecision, ToolInvocation
from finpilot.observability.audit import AuditStore


@pytest.fixture(autouse=True)
def disable_otel(monkeypatch):
    monkeypatch.setattr(settings, "otel_enabled", False)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)


def _state() -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="工资发放审批规则是什么？",
        normalized_intent="FINANCE_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


class FakeRagService:
    def __init__(self):
        self.issue = AgentIssue(
            code="RAG_VECTOR_RETRIEVAL_DEGRADED",
            component="rag:vector",
            message="Vector retrieval failed; continuing with remaining retrievers.",
            severity="warning",
            retryable=True,
            detail="RuntimeError: vector down",
        )

    def search(self, query: str, limit: int = 5) -> list[RagMatch]:
        return [
            RagMatch(
                document_id="doc-1",
                title="工资规则",
                source="manual",
                text="工资发放需要审批。",
                score=0.88,
            )
        ]

    def consume_issues(self) -> list[AgentIssue]:
        return [self.issue]


def test_tool_registry_includes_rag_degradation_issues():
    invocation = ToolRegistry(FakeRagService()).invoke(
        _state(),
        "search_finance_knowledge",
        query="宸ヨ祫鍙戞斁瀹℃壒瑙勫垯",
    )

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["documents"][0]["document_id"] == "doc-1"
    assert invocation.output["issues"][0]["code"] == "RAG_VECTOR_RETRIEVAL_DEGRADED"


def test_agent_runtime_merges_tool_output_issues_into_graph_state():
    state = _state()
    output = {
        "documents": [
            {
                "document_id": "doc-1",
                "title": "工资规则",
                "source": "manual",
                "text": "工资发放需要审批。",
                "score": 0.88,
            }
        ],
        "issues": [
            {
                "code": "RAG_VECTOR_RETRIEVAL_DEGRADED",
                "component": "rag:vector",
                "message": "Vector retrieval failed; continuing with remaining retrievers.",
                "severity": "warning",
                "retryable": True,
                "detail": "RuntimeError: vector down",
            }
        ],
    }

    AgentRuntime()._merge_tool_output(state, output)

    assert [issue.code for issue in state.issues] == ["RAG_VECTOR_RETRIEVAL_DEGRADED"]
    assert state.evidence[0].summary["document_id"] == "doc-1"


def test_agent_runtime_merges_tool_safety_findings_into_graph_state():
    state = _state()
    output = {
        "safety": {
            "findings": [
                {
                    "code": "TOOL_OPERATION_REQUIRES_APPROVAL",
                    "reviewer": "operation_risk",
                    "action": "BLOCK",
                    "message": "approval required",
                    "severity": "warning",
                    "detail": {"tool_name": "transfer_mock_funds"},
                }
            ]
        }
    }

    AgentRuntime()._merge_tool_output(state, output)

    assert [finding.code for finding in state.safety_findings] == ["TOOL_OPERATION_REQUIRES_APPROVAL"]


class FailingAnswerClient:
    def generate(self, prompt: str, *, model_name: str | None = None, system_prompt: str | None = None) -> str:
        raise RuntimeError("answer model down")


def test_finance_answering_service_records_fallback_issue():
    service = FinanceAnsweringService(client=FailingAnswerClient())

    answer = service.answer_with_context(
        {
            "session": {"user_message": "工资发放审批规则是什么？"},
            "evidence": [{"text": "工资发放需要审批。"}],
            "loop": {"working_notes": []},
        }
    )
    issues = service.consume_issues()

    assert "工资发放需要审批" in answer
    assert [issue.code for issue in issues] == ["ANSWER_LLM_DEGRADED"]
    assert issues[0].severity == "warning"


class FailingAuditStore(AuditStore):
    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        raise RuntimeError("audit table down")

    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        raise RuntimeError("audit table down")

    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        raise RuntimeError("audit table down")

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        raise RuntimeError("audit table down")


def test_graph_records_audit_persist_degradation_issue():
    state = _state()
    state.tool_invocations.append(ToolInvocation(tool_name="search_finance_knowledge"))
    graph = FinPilotGraph(router=object(), tools=object(), audit_store=FailingAuditStore(), memory_manager=object())

    result = GraphState.model_validate(graph._audit_persist(state.model_dump()))

    assert [issue.code for issue in result.issues] == ["AUDIT_PERSIST_DEGRADED"]
    assert result.issues[0].component == "audit"
    assert result.issues[0].severity == "warning"
