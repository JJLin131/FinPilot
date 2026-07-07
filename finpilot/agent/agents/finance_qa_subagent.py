from __future__ import annotations

from finpilot.agent.runtime import AgentRuntime
from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.intents import UNKNOWN_INTENT_ANSWER
from finpilot.models import GraphState, SubAgentContext


class FinanceQaSubAgent:
    name = "QueryAgent"
    context_policy = "finance_qa_agent"
    supported_intents = {"FINANCE_KNOWLEDGE_QA", "GENERAL_KNOWLEDGE_QA"}

    def __init__(self, runtime: AgentRuntime | None = None):
        self.runtime = runtime or AgentRuntime()

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        if state.normalized_intent in self.supported_intents:
            return self.runtime.run(state, self.build_context(tools), tools)

        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state

    def build_context(self, tools: ToolRegistry) -> SubAgentContext:
        configured = settings.agent_tool_allowlists.get(self.name)
        tool_names = list(configured) if configured is not None else ["search_finance_knowledge"]
        if settings.enable_demo_risk_tools:
            tool_names.append("transfer_mock_funds")
        return SubAgentContext(
            agent_name=self.name,
            role="You are a finance knowledge QA sub-agent.",
            goal="Answer the user's current finance question using retrieved knowledge evidence.",
            constraints=[
                "Do not invent finance rules.",
                "Use retrieved evidence when it is available.",
                "If evidence is insufficient, say that the knowledge base does not contain enough information.",
                "Do not perform route decisions, audit persistence, or system governance work.",
            ],
            success_criteria=[
                "A supported answer can be produced from evidence.",
                "Or the available evidence is insufficient and the agent stops clearly.",
            ],
            allowed_tools=tools.list_allowed(tool_names),
            max_steps=3,
            context_policy=self.context_policy,
        )


QuerySubAgent = FinanceQaSubAgent
