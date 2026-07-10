from __future__ import annotations

from finpilot.agent.agents.treasury_data_agent import TreasuryDataAgent
from finpilot.agent.agents.treasury_operation_agent import TreasuryOperationAgent
from finpilot.agent.runtime import AgentRuntime
from finpilot.agent.router import IntentRouter
from finpilot.agent.tools import ToolRegistry
from finpilot.intents import INTENT_ORDER
from finpilot.models import GraphState, ToolCard
from finpilot.safety.approval import ApprovalDecision, ApprovalService
from finpilot.safety.service import SafetyReviewService


class EmptyRagService:
    def search(self, query: str, limit: int = 3):
        return []


class FakeTools:
    def list_allowed(self, tool_names: list[str]) -> list[ToolCard]:
        return [
            ToolCard(
                name=name,
                description=f"{name} description",
                when_to_use=f"use {name}",
                arguments={"id": "string"},
            )
            for name in tool_names
        ]


class FailingDecisionService:
    def decide(self, prompt: str):
        raise RuntimeError("decision unavailable")


class RecordingAnsweringService:
    def __init__(self):
        self.snippets: list[str] = []

    def answer_with_context(self, session_context, snippets: list[str], working_notes: list[str]) -> str:
        self.snippets = snippets
        return "answer from treasury mock"


def _state(intent: str = "TREASURY_DATA_QUERY", target_agent: str = "TreasuryDataAgent") -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="query account balance for ACC-001",
        normalized_intent=intent,
        target_agent=target_agent,
    )


def test_router_targets_treasury_agents_by_intent():
    router = IntentRouter(classifier=object())

    data_route = router.route_from_scores("TREASURY_DATA_QUERY", "data", "TREASURY_DATA_QUERY", "UNKNOWN", 1.0, 1.0)
    operation_route = router.route_from_scores(
        "TREASURY_OPERATION",
        "operation",
        "TREASURY_OPERATION",
        "UNKNOWN",
        1.0,
        1.0,
    )

    assert data_route.target_agent == "TreasuryDataAgent"
    assert operation_route.target_agent == "TreasuryOperationAgent"


def test_treasury_knowledge_is_merged_into_finance_knowledge_intent():
    router = IntentRouter(classifier=object())

    intent, _ = router._heuristic_classify("treasury rule for payment approval")
    route = router.route_from_scores("FINANCE_KNOWLEDGE_QA", "knowledge", "FINANCE_KNOWLEDGE_QA", "UNKNOWN", 1.0, 1.0)

    assert "TREASURY_KNOWLEDGE_QA" not in INTENT_ORDER
    assert intent == "FINANCE_KNOWLEDGE_QA"
    assert route.target_agent == "QueryAgent"


def test_heuristic_router_classifies_data_and_operation_requests():
    router = IntentRouter(classifier=object())

    data_intent, _ = router._heuristic_classify("query account balance for ACC-001")
    operation_intent, _ = router._heuristic_classify("create transfer order from ACC-001 to ACC-002")

    assert data_intent == "TREASURY_DATA_QUERY"
    assert operation_intent == "TREASURY_OPERATION"


def test_heuristic_router_keeps_receipt_timing_question_as_knowledge_qa():
    router = IntentRouter(classifier=object())

    intent, _ = router._heuristic_classify("工商银行转账回单什么时候可以下载？")

    assert intent == "FINANCE_KNOWLEDGE_QA"


def test_treasury_agents_expose_only_their_tool_groups():
    data_tools = TreasuryDataAgent().build_context(FakeTools()).allowed_tools
    operation_tools = TreasuryOperationAgent().build_context(FakeTools()).allowed_tools

    assert [tool.name for tool in data_tools] == [
        "query_treasury_account",
        "query_account_balance",
        "query_transactions",
        "query_receipt_status",
        "query_cash_pool_position",
    ]
    assert [tool.name for tool in operation_tools] == [
        "create_payment_order",
        "create_transfer_order",
        "download_receipt",
    ]


def test_treasury_data_tool_returns_mock_content_without_approval():
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "query_account_balance", accountId="ACC-001")

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["tool_name"] == "query_account_balance"
    assert invocation.output["content"][0]["metadata"]["kind"] == "treasury_data"
    assert invocation.output["content"][0]["metadata"]["accountId"] == "ACC-001"


def test_treasury_data_agent_fallback_invokes_balance_query_when_decision_llm_fails():
    answering = RecordingAnsweringService()
    agent = TreasuryDataAgent(
        runtime=AgentRuntime(decision_service=FailingDecisionService(), answering_service=answering)
    )
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    state = agent.handle(_state(), registry)

    assert state.tool_invocations[0].tool_name == "query_account_balance"
    assert state.evidence[0].tool_name == "query_account_balance"
    assert answering.snippets


def test_high_risk_treasury_operation_requires_approval():
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(
        _state(intent="TREASURY_OPERATION", target_agent="TreasuryOperationAgent"),
        "create_transfer_order",
        fromAccountId="ACC-001",
        toAccountId="ACC-002",
        amount=1000,
        currency="CNY",
        purpose="working capital transfer",
    )

    assert invocation.status == "BLOCKED"
    assert invocation.output["safety"]["findings"][0]["code"] == "TOOL_OPERATION_REQUIRES_APPROVAL"


def test_high_risk_treasury_operation_executes_after_approval():
    safety = SafetyReviewService(
        approval_service=ApprovalService(callback=lambda request: ApprovalDecision(scope="once")),
        interactive_approval=True,
    )
    registry = ToolRegistry(EmptyRagService(), safety=safety)

    invocation = registry.invoke(
        _state(intent="TREASURY_OPERATION", target_agent="TreasuryOperationAgent"),
        "create_transfer_order",
        fromAccountId="ACC-001",
        toAccountId="ACC-002",
        amount=1000,
        currency="CNY",
        purpose="working capital transfer",
    )

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["mock"] is True
    assert invocation.output["operation"] == "create_transfer_order"
