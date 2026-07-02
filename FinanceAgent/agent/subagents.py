from __future__ import annotations

from FinanceAgent.agent.runtime import AgentRuntime
from FinanceAgent.agent.tools import ToolRegistry
from FinanceAgent.context.builders import build_subagent_context
from FinanceAgent.intents import UNKNOWN_INTENT_ANSWER
from FinanceAgent.models import GraphState


class QuerySubAgent:
    name = "QueryAgent"

    def __init__(self, runtime: AgentRuntime | None = None):
        self.runtime = runtime or AgentRuntime()

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        if state.normalized_intent in {"FINANCE_KNOWLEDGE_QA", "GENERAL_KNOWLEDGE_QA"}:
            subagent_context = build_subagent_context(state, self.name, tools)
            return self.runtime.run(state, subagent_context, tools)

        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state


class TransferSubAgent:
    name = "TransferAgent"

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state
