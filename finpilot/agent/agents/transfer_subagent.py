from __future__ import annotations

from finpilot.agent.tools import ToolRegistry
from finpilot.intents import UNKNOWN_INTENT_ANSWER
from finpilot.models import GraphState


class TransferSubAgent:
    name = "TransferAgent"

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state

