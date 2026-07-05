from __future__ import annotations

import json

import httpx
import pytest

from finpilot.agent.graph import FinPilotGraph
from finpilot.agent.router import IntentRouter
from finpilot.config import settings
from finpilot.memory.models import MemoryContext
from finpilot.memory.service import MemoryManager
from finpilot.models import AgentChatResponse, AgentEvidence, AgentIssue, EvalSuiteResult, GraphState, RouteDecision
from finpilot.observability.audit import AuditStore
from finpilot.responses import prepare_chat_response


@pytest.fixture(autouse=True)
def disable_otel(monkeypatch):
    monkeypatch.setattr(settings, "otel_enabled", False)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)


class RaisingClassifier:
    def __init__(self, exc: Exception):
        self.exc = exc

    def classify(self, query: str):
        raise self.exc


class StaticClassifier:
    def __init__(self, intent: str, reason: str):
        self.intent = intent
        self.reason = reason

    def classify(self, query: str):
        return self.intent, self.reason


def _http_status_error(status_code: int, body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://llm.example.test/chat/completions")
    response = httpx.Response(status_code, request=request, text=body)
    return httpx.HTTPStatusError("request failed", request=request, response=response)


def test_router_maps_expected_routing_llm_failures():
    cases = [
        (RuntimeError("DEEPSEEK_API_KEY is not configured."), "ROUTING_LLM_CONFIG_MISSING"),
        (httpx.ConnectTimeout("timed out"), "ROUTING_LLM_UNAVAILABLE"),
        (_http_status_error(404, '{"error":"model not found"}'), "ROUTING_LLM_MODEL_UNAVAILABLE"),
        (ValueError("DeepSeek returned an empty response."), "ROUTING_LLM_INVALID_RESPONSE"),
        (json.JSONDecodeError("Expecting value", "", 0), "ROUTING_LLM_INVALID_RESPONSE"),
    ]

    for exc, expected_code in cases:
        router = IntentRouter(classifier=RaisingClassifier(exc))
        result = router.classify_with_issues("hello outside scope")

        assert result.intent == "UNKNOWN"
        assert [issue.code for issue in result.issues] == [expected_code]
        assert result.issues[0].component == "routing_llm"
        assert result.issues[0].detail


class FakeMemory:
    def __init__(self):
        self.remembered: list[tuple[str, str, str, str, AgentChatResponse]] = []

    def load(self, memory_id: str, user_id: str, user_message: str = "") -> MemoryContext:
        return MemoryContext()

    def remember_interaction(
        self,
        memory_id: str,
        user_id: str,
        chat_id: str,
        content: str,
        response: AgentChatResponse,
    ) -> None:
        self.remembered.append((memory_id, user_id, chat_id, content, response))


class RecordingAuditStore(AuditStore):
    def __init__(self):
        self.unknown: list[tuple[GraphState, RouteDecision]] = []
        self.issues: list[tuple[GraphState, AgentIssue]] = []
        self.tools = []

    def record_tool(self, state: GraphState, invocation) -> None:
        self.tools.append((state, invocation))

    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        self.unknown.append((state, decision))

    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        self.issues.append((state, issue))

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        pass


class FakeQueryAgent:
    name = "QueryAgent"

    def handle(self, state: GraphState, tools) -> GraphState:
        state.final_answer = "finance answer"
        state.evidence.append(
            AgentEvidence(
                tool_name="search_finance_knowledge",
                source="manual",
                summary={"document_id": "doc-1"},
            )
        )
        return state


def _graph(router: IntentRouter, audit: RecordingAuditStore, memory: FakeMemory) -> FinPilotGraph:
    graph = FinPilotGraph(router, tools=object(), audit_store=audit, memory_manager=memory)
    graph.query_agent = FakeQueryAgent()
    return graph


def test_graph_returns_failed_when_routing_model_unavailable_and_heuristic_misses():
    audit = RecordingAuditStore()
    memory = FakeMemory()
    graph = _graph(
        IntentRouter(classifier=RaisingClassifier(_http_status_error(404, '{"error":"model not found"}'))),
        audit,
        memory,
    )

    response = graph.run("user-1", "chat-1", "hello outside scope")

    assert response.status == "FAILED"
    assert response.route.normalized_intent == "UNKNOWN"
    assert response.route.fallback_cause == "ROUTING_LLM_MODEL_UNAVAILABLE"
    assert response.issues[0].code == "ROUTING_LLM_MODEL_UNAVAILABLE"
    assert "路由侧 LLM 模型不可用" in response.answer
    assert not audit.unknown
    assert [issue.code for _, issue in audit.issues] == ["ROUTING_LLM_MODEL_UNAVAILABLE"]


def test_graph_returns_degraded_when_routing_llm_fails_but_heuristic_hits():
    audit = RecordingAuditStore()
    memory = FakeMemory()
    graph = _graph(
        IntentRouter(classifier=RaisingClassifier(RuntimeError("DEEPSEEK_API_KEY is not configured."))),
        audit,
        memory,
    )

    response = graph.run("user-1", "chat-1", "bank rule")

    assert response.status == "DEGRADED"
    assert response.route.normalized_intent == "FINANCE_KNOWLEDGE_QA"
    assert response.route.fallback_cause == "ROUTING_LLM_CONFIG_MISSING"
    assert response.answer == "finance answer"
    assert response.issues[0].severity == "warning"
    assert not audit.unknown


def test_graph_keeps_true_unknown_as_unsupported_without_issue():
    audit = RecordingAuditStore()
    memory = FakeMemory()
    graph = _graph(IntentRouter(classifier=StaticClassifier("UNKNOWN", "outside supported domain")), audit, memory)

    response = graph.run("user-1", "chat-1", "hello outside scope")

    assert response.status == "UNSUPPORTED"
    assert response.route.normalized_intent == "UNKNOWN"
    assert response.issues == []
    assert len(audit.unknown) == 1
    assert audit.issues == []


def test_prepare_chat_response_scrubs_issue_detail_without_debug():
    response = AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="FAILED",
        answer="failed",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="UNKNOWN",
            reason="failed",
            confidence=0.0,
            valid=False,
            target_agent="UNSUPPORTED",
            classifier_intent="UNKNOWN",
        ),
        issues=[
            AgentIssue(
                code="ROUTING_LLM_UNAVAILABLE",
                component="routing_llm",
                message="路由侧 LLM 服务不可用，请稍后重试。",
                detail="ConnectTimeout: timed out",
            )
        ],
        route_debug={"detail": "debug"},
        retrieval_debug={"detail": "debug"},
        tool_calls=[],
    )

    prepared = prepare_chat_response(response, debug_enabled=False)

    assert prepared.issues[0].detail is None
    assert prepared.route_debug is None
    assert prepared.retrieval_debug is None
    assert prepared.tool_calls is None


class FakeChatStore:
    def __init__(self):
        self.updated = False

    def get_messages(self, memory_id: str):
        return []

    def update_messages(self, memory_id: str, messages) -> None:
        self.updated = True


def test_memory_skips_failed_and_unsupported_chat_turns():
    chat_store = FakeChatStore()
    scheduled = []
    manager = MemoryManager(
        chat_store=chat_store,
        profile_store=object(),
        semantic_store=object(),
        extractor=object(),
        async_submitter=lambda *args: scheduled.append(args),
    )
    failed_response = AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="FAILED",
        answer="路由侧 LLM 模型不可用，请检查路由模型配置或稍后重试。",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="UNKNOWN",
            reason="failed",
            confidence=0.0,
            valid=False,
            target_agent="UNSUPPORTED",
            classifier_intent="UNKNOWN",
        ),
        issues=[
            AgentIssue(
                code="ROUTING_LLM_MODEL_UNAVAILABLE",
                component="routing_llm",
                message="路由侧 LLM 模型不可用，请检查路由模型配置或稍后重试。",
            )
        ],
    )

    manager.remember_interaction("chat:user:chat", "user", "chat", "hello", failed_response)

    assert not chat_store.updated
    assert scheduled == []
