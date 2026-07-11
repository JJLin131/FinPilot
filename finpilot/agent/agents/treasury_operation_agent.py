from __future__ import annotations

from finpilot.agent.runtime import AgentRuntime
from finpilot.agent.tools import ToolRegistry
from finpilot.intents import UNKNOWN_INTENT_ANSWER
from finpilot.models import GraphState, SubAgentContext, SubAgentResult


class TreasuryOperationAgent:
    name = "TreasuryOperationAgent"
    context_policy = "treasury_operation_agent"
    supported_intents = {"TREASURY_OPERATION"}
    default_tools = [
        "create_payment_order",
        "create_transfer_order",
        "download_receipt",
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
            role="You are a treasury operation sub-agent.",
            goal="Prepare mock payment, transfer, and receipt download operations through governed tools.",
            constraints=[
                "Do not claim real money was moved.",
                "Operation tools may require safety approval before execution.",
                "Use backend or mock tool results as the source of truth.",
                "If required identifiers or payment details are missing, ask for them before calling tools.",
            ],
            success_criteria=[
                "A mock operation result is returned from an allowed tool.",
                "Or the operation is blocked by safety review with a clear reason.",
            ],
            allowed_tools=tools.list_allowed(self.default_tools),
            max_steps=2,
            context_policy=self.context_policy,
        )
