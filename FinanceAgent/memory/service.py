from __future__ import annotations

from FinanceAgent.intents import UNKNOWN_INTENT_ANSWER
from FinanceAgent.memory.models import ChatTurn, MemoryContext
from FinanceAgent.memory.stores import AgentChatMemoryStore
from FinanceAgent.models import AgentChatResponse


class MemoryManager:
    def __init__(
        self,
        chat_store: AgentChatMemoryStore | None = None,
    ):
        self.chat_store = chat_store or AgentChatMemoryStore()

    def load(self, memory_id: str, user_id: str) -> MemoryContext:
        return MemoryContext(
            recent_messages=self._agent_visible_messages(self.chat_store.get_messages(memory_id))[-20:],
        )

    def remember_interaction(self, memory_id: str, user_id: str, user_message: str, response: AgentChatResponse) -> None:
        if response.route.normalized_intent != "UNKNOWN":
            messages = self._agent_visible_messages(self.chat_store.get_messages(memory_id))
            messages.extend(
                [
                    ChatTurn(role="user", content=user_message),
                    ChatTurn(role="assistant", content=response.answer),
                ]
            )
            self.chat_store.update_messages(memory_id, messages[-20:])

    @staticmethod
    def _agent_visible_messages(messages: list[ChatTurn]) -> list[ChatTurn]:
        visible: list[ChatTurn] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            next_turn = messages[index + 1] if index + 1 < len(messages) else None
            if (
                current.role == "user"
                and next_turn is not None
                and next_turn.role == "assistant"
                and next_turn.content == UNKNOWN_INTENT_ANSWER
            ):
                index += 2
                continue
            if current.role == "assistant" and current.content == UNKNOWN_INTENT_ANSWER:
                index += 1
                continue
            visible.append(current)
            index += 1
        return visible
