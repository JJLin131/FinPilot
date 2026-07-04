from __future__ import annotations

import logging
import threading
from typing import Any

from FinanceAgent.config import settings
from FinanceAgent.intents import UNKNOWN_INTENT_ANSWER
from FinanceAgent.memory.definitions import SEMANTIC_MEMORY_KEYS, STRUCTURED_MEMORY_FIELDS
from FinanceAgent.memory.extractor import MemoryExtractor
from FinanceAgent.memory.models import ChatTurn, ExtractedMemory, MemoryContext, SemanticMemoryItem, SemanticMemoryRecord
from FinanceAgent.memory.semantic_store import ChromaSemanticMemoryStore
from FinanceAgent.memory.stores import AgentChatMemoryStore, UserProfileMemoryStore
from FinanceAgent.models import AgentChatResponse

logger = logging.getLogger(__name__)

RECENT_MESSAGE_LIMIT = 20
MAX_STORED_MESSAGES = 80


class MemoryManager:
    def __init__(
        self,
        chat_store: AgentChatMemoryStore | None = None,
        profile_store: UserProfileMemoryStore | None = None,
        semantic_store: ChromaSemanticMemoryStore | None = None,
        extractor: MemoryExtractor | None = None,
        async_submitter=None,
    ):
        self.chat_store = chat_store or AgentChatMemoryStore()
        self.profile_store = profile_store or UserProfileMemoryStore()
        self.semantic_store = semantic_store or ChromaSemanticMemoryStore()
        self.extractor = extractor or MemoryExtractor()
        self.async_submitter = async_submitter or self._submit_daemon

    def load(self, memory_id: str, user_id: str, user_message: str = "") -> MemoryContext:
        visible_messages = self._load_recent_messages(memory_id)
        structured_memory = self._load_profile(user_id)
        semantic_memory = self._load_semantic_memory(user_id, user_message)
        return MemoryContext(
            recent_messages=visible_messages[-RECENT_MESSAGE_LIMIT:],
            structured_memory=structured_memory,
            semantic_memory=semantic_memory,
            long_term_memory=self._format_long_term_memory(structured_memory, semantic_memory),
        )

    def remember_interaction(
        self,
        memory_id: str,
        user_id: str,
        chat_id: str,
        user_message: str | AgentChatResponse,
        response: AgentChatResponse | None = None,
    ) -> None:
        if response is None and isinstance(user_message, AgentChatResponse):
            response = user_message
            user_message = chat_id
            chat_id = self._chat_id_from_memory_id(memory_id)
        if response is None:
            raise ValueError("response is required.")
        if response.route.normalized_intent != "UNKNOWN":
            messages = self._load_recent_messages(memory_id)
            messages.extend(
                [
                    ChatTurn(role="user", content=user_message),
                    ChatTurn(role="assistant", content=response.answer),
                ]
            )
            try:
                self.chat_store.update_messages(memory_id, messages[-MAX_STORED_MESSAGES:])
            except Exception as exc:
                logger.warning("Short-term chat memory write failed for %s: %s", memory_id, exc)
            self._schedule_long_term_memory(user_id, chat_id, user_message, response)

    def persist_extracted_memory(self, user_id: str, extracted: ExtractedMemory) -> None:
        structured = self._valid_structured_memories(extracted.structured_memories)
        if structured:
            self.profile_store.upsert_profile(user_id, structured)
        for item in extracted.semantic_memories:
            if self._valid_semantic_memory(item):
                self.semantic_store.upsert_summary(user_id, item)

    def _schedule_long_term_memory(
        self,
        user_id: str,
        chat_id: str,
        user_message: str,
        response: AgentChatResponse,
    ) -> None:
        if not settings.memory_extraction_enabled:
            return
        self.async_submitter(self._extract_and_store_long_term_memory, user_id, chat_id, user_message, response)

    def _extract_and_store_long_term_memory(
        self,
        user_id: str,
        chat_id: str,
        user_message: str,
        response: AgentChatResponse,
    ) -> None:
        try:
            extracted = self.extractor.extract(
                user_id=user_id,
                chat_id=chat_id,
                user_message=user_message,
                assistant_answer=response.answer,
                route=response.route,
            )
            self.persist_extracted_memory(user_id, extracted)
        except Exception as exc:
            logger.warning("Long-term memory extraction/write failed for user %s: %s", user_id, exc)

    def _load_recent_messages(self, memory_id: str) -> list[ChatTurn]:
        try:
            return self._agent_visible_messages(self.chat_store.get_messages(memory_id))
        except Exception as exc:
            logger.warning("Short-term chat memory load failed for %s: %s", memory_id, exc)
            return []

    def _load_profile(self, user_id: str) -> dict[str, Any]:
        try:
            return self.profile_store.get_profile(user_id)
        except Exception as exc:
            logger.warning("Structured user memory load failed for %s: %s", user_id, exc)
            return {}

    def _load_semantic_memory(self, user_id: str, user_message: str) -> list[SemanticMemoryRecord]:
        try:
            return self.semantic_store.search(user_id, user_message, settings.memory_semantic_search_limit)
        except Exception as exc:
            logger.warning("Semantic user memory load failed for %s: %s", user_id, exc)
            return []

    def _valid_structured_memories(self, values: dict[str, Any]) -> dict[str, Any]:
        valid: dict[str, Any] = {}
        for key, value in values.items():
            if key not in STRUCTURED_MEMORY_FIELDS:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            valid[key] = value
        return valid

    def _valid_semantic_memory(self, item: SemanticMemoryItem) -> bool:
        return (
            item.memory_key in SEMANTIC_MEMORY_KEYS
            and bool(item.memory_value.strip())
            and item.confidence >= settings.memory_min_confidence
        )

    def _format_long_term_memory(
        self,
        structured_memory: dict[str, Any],
        semantic_memory: list[SemanticMemoryRecord],
    ) -> list[str]:
        values: list[str] = []
        if structured_memory:
            values.append(
                "Structured user profile: "
                + "; ".join(f"{field}={value}" for field, value in structured_memory.items() if value is not None)
            )
        for item in semantic_memory:
            if item.memory_key and item.memory_value:
                values.append(f"{item.memory_key}: {item.memory_value}")
        return values

    def _submit_daemon(self, call, *args) -> None:
        thread = threading.Thread(target=call, args=args, daemon=True)
        thread.start()

    def _chat_id_from_memory_id(self, memory_id: str) -> str:
        return memory_id.rsplit(":", 1)[-1] if ":" in memory_id else memory_id

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
