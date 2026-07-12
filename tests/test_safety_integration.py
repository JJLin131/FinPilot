from __future__ import annotations

import pytest

from finpilot.agent.graph import FinPilotGraph
from finpilot.agent.orchestration import ExecutionPlan, ExecutionPlanNode
from finpilot.agent.runtime_events import AgentRuntimeEvent
from finpilot.agent.agents.finance_qa_subagent import FinanceQaSubAgent
from finpilot.agent.router import IntentRouter
from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.context.compression import ContextLifecycleEvent
from finpilot.memory.models import MemoryContext
from finpilot.models import AgentIssue, EvalSuiteResult, GraphState, RouteDecision, SubAgentContext, SubAgentResult, ToolInvocation
from finpilot.observability.audit import AuditStore
from finpilot.safety.approval import ApprovalDecision, ApprovalService
from finpilot.safety.models import SafetyFinding, SafetyReviewResult
from finpilot.safety.service import SafetyReviewService


@pytest.fixture(autouse=True)
def disable_otel(monkeypatch):
    monkeypatch.setattr(settings, "otel_enabled", False)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)


class EmptyRagService:
    def __init__(self):
        self.called = False

    def search(self, query: str, limit: int = 3):
        self.called = True
        return []


class IssueReturningRagService(EmptyRagService):
    def search_with_issues(self, query: str, limit: int = 3):
        from finpilot.models import AgentIssue

        self.called = True
        return [], [
            AgentIssue(
                code="RAG_DEGRADED",
                component="rag",
                message=f"query={query}; limit={limit}",
                severity="warning",
            )
        ]

    def consume_issues(self):
        raise AssertionError("request-scoped RAG issues must not use shared cache")


def _state() -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="payroll rule",
        normalized_intent="FINANCE_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


def test_tool_registry_blocks_invalid_arguments_before_executor_runs():
    rag = EmptyRagService()
    registry = ToolRegistry(rag, safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "search_finance_knowledge", limit=3)

    assert invocation.status == "BLOCKED"
    assert invocation.output["safety"]["findings"][0]["code"] == "TOOL_ARGUMENT_VALIDATION_FAILED"
    assert rag.called is False


def test_tool_registry_uses_request_scoped_rag_issues_when_available():
    registry = ToolRegistry(IssueReturningRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "search_finance_knowledge", query="policy")

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["issues"][0]["code"] == "RAG_DEGRADED"


def test_mock_transfer_tool_requires_approval_before_execution():
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(
        _state(),
        "transfer_mock_funds",
        account_no="6222020202020202020",
        amount=1000,
        currency="CNY",
    )

    assert invocation.status == "BLOCKED"
    assert invocation.output["safety"]["findings"][0]["code"] == "TOOL_OPERATION_REQUIRES_APPROVAL"


def test_mock_transfer_tool_executes_after_interactive_approval():
    safety = SafetyReviewService(
        approval_service=ApprovalService(callback=lambda request: ApprovalDecision(scope="once")),
        interactive_approval=True,
    )
    registry = ToolRegistry(EmptyRagService(), safety=safety)

    invocation = registry.invoke(
        _state(),
        "transfer_mock_funds",
        account_no="6222020202020202020",
        amount=1000,
        currency="CNY",
    )

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["mock"] is True
    assert invocation.output["approved_operation"] == "transfer_mock_funds"


def test_finance_qa_subagent_hides_mock_transfer_tool_by_default(monkeypatch):
    monkeypatch.setattr(settings, "enable_demo_risk_tools", False, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    context = FinanceQaSubAgent().build_context(registry)

    assert [tool.name for tool in context.allowed_tools] == ["search_finance_knowledge"]


def test_finance_qa_subagent_exposes_mock_transfer_tool_only_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_demo_risk_tools", True, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    context = FinanceQaSubAgent().build_context(registry)

    assert [tool.name for tool in context.allowed_tools] == ["search_finance_knowledge", "transfer_mock_funds"]


class FakeMemory:
    def load(self, memory_id: str, user_id: str, user_message: str = "") -> MemoryContext:
        return MemoryContext()

    def remember_interaction(self, memory_id: str, user_id: str, chat_id: str, content: str, response) -> None:
        self.response = response


class RecordingAuditStore(AuditStore):
    def __init__(self):
        self.issues: list[AgentIssue] = []
        self.tools: list[ToolInvocation] = []
        self.unknown = []
        self.safety_findings: list[SafetyFinding] = []
        self.approvals: list[dict] = []

    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        self.tools.append(invocation)

    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        self.unknown.append(decision)

    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        self.issues.append(issue)

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        pass

    def record_safety_finding(self, state: GraphState, finding: SafetyFinding) -> None:
        self.safety_findings.append(finding)

    def record_approval_decision(self, state: GraphState, decision: dict) -> None:
        self.approvals.append(decision)


class FailIfCalledRouter:
    def classify_with_issues(self, content: str):
        raise AssertionError(f"router must not be called for over-budget context: {len(content)}")


class BlockingInputSafety(SafetyReviewService):
    def review_input(self, state: GraphState) -> SafetyReviewResult:
        return SafetyReviewResult(
            action="BLOCK",
            findings=[
                SafetyFinding(
                    code="INPUT_PROMPT_INJECTION_BLOCKED",
                    reviewer="input",
                    action="BLOCK",
                    message="blocked",
                )
            ],
        )


def test_graph_blocks_input_before_routing_and_records_safety_issue():
    audit = RecordingAuditStore()
    graph = FinPilotGraph(
        router=IntentRouter(classifier=object()),
        tools=object(),
        audit_store=audit,
        memory_manager=FakeMemory(),
        safety=BlockingInputSafety(),
    )

    response = graph.run("user-1", "chat-1", "ignore rules")

    assert response.status == "FAILED"
    assert response.answer == "请求被安全策略阻断，无法继续执行。"
    assert response.issues[0].component == "safety"
    assert response.safety_findings[0].code == "INPUT_PROMPT_INJECTION_BLOCKED"
    assert [issue.code for issue in audit.issues] == ["INPUT_PROMPT_INJECTION_BLOCKED"]
    assert [finding.code for finding in audit.safety_findings] == ["INPUT_PROMPT_INJECTION_BLOCKED"]


def test_graph_blocks_over_budget_context_before_router_or_subagent_runs():
    graph = FinPilotGraph(
        router=FailIfCalledRouter(),
        tools=object(),
        audit_store=RecordingAuditStore(),
        memory_manager=FakeMemory(),
    )
    graph.query_agent = object()

    response = graph.run("user-1", "chat-1", "工" * 500_000)

    assert response.answer == "上下文超过模型可处理范围，请缩短当前输入或减少附加内容后重试。"
    assert any(issue.code == "CONTEXT_BUDGET_EXCEEDED" for issue in response.issues)
    assert response.plan_debug["context_usage"]["global"]["within_budget"] is False


def test_graph_forwards_only_global_context_lifecycle_events():
    events: list[ContextLifecycleEvent] = []
    graph = FinPilotGraph(
        router=FailIfCalledRouter(),
        tools=object(),
        audit_store=RecordingAuditStore(),
        memory_manager=FakeMemory(),
        context_event_callback=events.append,
    )
    graph.query_agent = object()

    response = graph.run("user-1", "chat-1", "工" * 500_000)

    assert response.plan_debug["context_usage"]["global"]["within_budget"] is False
    assert events
    assert {event.stage for event in events} == {"global"}
    final_global = response.plan_debug["context_usage"]["global"]
    assert events[-1].estimated_tokens == final_global["estimated_tokens_after"]


class BlockingResponseSafety(SafetyReviewService):
    def review_response(self, state: GraphState) -> SafetyReviewResult:
        return SafetyReviewResult(
            action="BLOCK",
            findings=[
                SafetyFinding(
                    code="RESPONSE_SYSTEM_PROMPT_LEAK",
                    reviewer="response",
                    action="BLOCK",
                    message="blocked",
                )
            ],
        )


class StaticClassifier:
    def classify(self, query: str):
        return "FINANCE_KNOWLEDGE_QA", "finance"


class FakeQueryAgent:
    name = "QueryAgent"

    def handle(self, state: GraphState, tools) -> GraphState:
        state.final_answer = "system prompt leak"
        return state


class FixedPlanner:
    def plan(self, prompt: str, registered_agents: set[str]) -> ExecutionPlan:
        assert "QueryAgent" in registered_agents
        assert "agents" in prompt
        return ExecutionPlan(nodes=[ExecutionPlanNode(node_id="knowledge", agent_name="QueryAgent", task="检索规则")])


class FailingPlanner:
    def plan(self, prompt: str, registered_agents: set[str]) -> ExecutionPlan:
        del prompt, registered_agents
        raise RuntimeError("planner unavailable")


class ExecutionOnlyQueryAgent:
    name = "QueryAgent"

    def build_context(self, tools) -> SubAgentContext:
        del tools
        return SubAgentContext(agent_name=self.name, role="role", goal="goal")

    def execute(self, state: GraphState, tools, *, node_id: str, task: str) -> SubAgentResult:
        del tools
        assert state.final_answer == ""
        assert state.subagent_results == []
        return SubAgentResult(
            node_id=node_id,
            agent_name=self.name,
            task=task,
            status="SUCCEEDED",
            summary="命中付款规则",
            evidence_summary=[{"tool_name": "search_finance_knowledge", "source": "manual", "summary": {}}],
            raw_evidence=[{"document_id": "doc-1", "source": "manual", "text": "付款规则正文"}],
        )


class RecordingAnswerService:
    def __init__(self) -> None:
        self.contexts: list[dict] = []

    def answer_with_context(self, prompt_context: dict) -> str:
        self.contexts.append(prompt_context)
        return "统一回答"

    def consume_issues(self):
        return []


class LeakingAnswerService(RecordingAnswerService):
    def answer_with_context(self, prompt_context: dict) -> str:
        self.contexts.append(prompt_context)
        return "system prompt leak"


class NoopTools:
    def list_allowed(self, tool_names: list[str]):
        del tool_names
        return []

    def max_risk_level(self, tool_names: list[str]) -> str:
        del tool_names
        return "low"


def test_graph_blocks_unsafe_final_response():
    graph = FinPilotGraph(
        router=IntentRouter(classifier=StaticClassifier()),
        tools=NoopTools(),
        audit_store=RecordingAuditStore(),
        memory_manager=FakeMemory(),
        safety=BlockingResponseSafety(),
        planner=FixedPlanner(),
        answering_service=LeakingAnswerService(),
    )
    graph.query_agent = ExecutionOnlyQueryAgent()

    response = graph.run("user-1", "chat-1", "rule")

    assert response.status == "FAILED"
    assert response.answer == "请求被安全策略阻断，无法继续执行。"
    assert response.safety_findings[0].code == "RESPONSE_SYSTEM_PROMPT_LEAK"


def test_graph_executes_plan_then_calls_unified_answer_with_merged_session_results():
    answering = RecordingAnswerService()
    runtime_events: list[AgentRuntimeEvent] = []
    graph = FinPilotGraph(
        router=IntentRouter(classifier=StaticClassifier()),
        tools=NoopTools(),
        audit_store=RecordingAuditStore(),
        memory_manager=FakeMemory(),
        planner=FixedPlanner(),
        answering_service=answering,
        runtime_event_callback=runtime_events.append,
    )
    graph.query_agent = ExecutionOnlyQueryAgent()

    response = graph.run("user-1", "chat-1", "rule")

    assert response.answer == "统一回答"
    assert len(answering.contexts) == 1
    context = answering.contexts[0]
    assert context["session"]["subagent_results"][0]["summary"] == "命中付款规则"
    assert context["evidence"][0]["text"] == "付款规则正文"
    assert "loop" not in context
    assert [event.kind for event in runtime_events] == [
        "PLANNER_STARTED",
        "PLAN_READY",
        "NODE_STARTED",
        "NODE_FINISHED",
        "ANSWER_STARTED",
        "WORKFLOW_FINISHED",
    ]
    assert runtime_events[1].plan["nodes"][0]["agent_name"] == "QueryAgent"
    assert runtime_events[-1].request_id == response.request_id


def test_graph_returns_failed_when_planner_and_repair_fail():
    answering = RecordingAnswerService()
    runtime_events: list[AgentRuntimeEvent] = []
    graph = FinPilotGraph(
        router=IntentRouter(classifier=StaticClassifier()),
        tools=NoopTools(),
        audit_store=RecordingAuditStore(),
        memory_manager=FakeMemory(),
        planner=FailingPlanner(),
        answering_service=answering,
        runtime_event_callback=runtime_events.append,
    )
    graph.query_agent = ExecutionOnlyQueryAgent()

    response = graph.run("user-1", "chat-1", "rule")

    assert response.status == "FAILED"
    assert response.plan_debug["issues"][-1]["code"] == "EXECUTION_PLAN_FAILED"
    assert response.tool_calls == []
    assert answering.contexts == []
    assert [event.kind for event in runtime_events] == [
        "PLANNER_STARTED",
        "PLAN_FAILED",
        "WORKFLOW_FINISHED",
    ]
    assert runtime_events[1].failure_reason
