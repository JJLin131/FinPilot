from __future__ import annotations

from finpilot.agent.runtime import AgentRuntime
from finpilot.agent.tools import ToolRegistry
from finpilot.intents import UNKNOWN_INTENT_ANSWER
from finpilot.models import GraphState, SubAgentContext, SubAgentResult


class TreasuryDataAgent:
    name = "TreasuryDataAgent"
    context_policy = "treasury_data_agent"
    supported_intents = {"TREASURY_DATA_QUERY"}
    default_tools = [
        "query_treasury_account",
        "query_account_balance",
        "query_transactions",
        "query_receipt_status",
        "query_cash_pool_position",
    ]

    def __init__(self, runtime: AgentRuntime | None = None):
        self.runtime = runtime or AgentRuntime()

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        if state.normalized_intent in self.supported_intents:
            return self.runtime.run(state, self.build_context(tools), tools)

        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state

    def execute(self, state: GraphState, tools: ToolRegistry, *, node_id: str, task: str) -> SubAgentResult:
        return self.runtime.execute(state, self.build_context(tools), tools, node_id=node_id, task=task)

    def build_context(self, tools: ToolRegistry) -> SubAgentContext:
        return SubAgentContext(
            agent_name=self.name,
            role="You are a treasury read-only data query sub-agent.",
            goal="Answer account, balance, transaction, receipt status, and cash pool questions using allowed read-only tools.",
            constraints=[
                "Only perform read-only treasury data queries.",
                "Do not create payment, transfer, or document download operations.",
                "Use the tool result as the source of truth.",
                "If required identifiers are missing, ask for the missing identifier instead of guessing.",
            ],
            success_criteria=[
                "A read-only treasury data result is returned from an allowed tool.",
                "Or the missing identifier is clearly requested.",
            ],
            allowed_tools=tools.list_allowed(self.default_tools),
            max_steps=2,
            context_policy=self.context_policy,
        )
