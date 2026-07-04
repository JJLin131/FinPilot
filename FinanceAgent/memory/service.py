from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
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

MEMORY_SIGNAL_TERMS = (
    "记住",
    "以后",
    "偏好",
    "喜欢",
    "不喜欢",
    "忘记",
    "别再",
    "不要再",
    "更正",
    "纠正",
    "我的",
    "我常",
    "我通常",
    "我希望",
    "每个月",
    "每月",
    "remember",
    "prefer",
    "forget",
    "from now on",
    "always",
    "never",
    "my ",
)
UNSAFE_MEMORY_PATTERNS = (
    re.compile(r"\b(api[_-]?key|secret|password|token)\b", re.IGNORECASE),
    re.compile(r"(密码|口令|密钥|验证码|身份证号|银行卡号|完整卡号|卡号)"),
    re.compile(r"\b\d{15,19}\b"),
    re.compile(r"ignore (all )?(previous|prior) instructions", re.IGNORECASE),
    re.compile(r"(系统提示词|开发者消息|system prompt|developer message)", re.IGNORECASE),
)


class MemoryManager:
    def __init__(
        self,
        chat_store: AgentChatMemoryStore | None = None,
        profile_store: UserProfileMemoryStore | None = None,
        semantic_store: ChromaSemanticMemoryStore | None = None,
        extractor: MemoryExtractor | None = None,
        async_submitter=None,
        extraction_executor: ThreadPoolExecutor | None = None,
    ):
        self.chat_store = chat_store or AgentChatMemoryStore()
        self.profile_store = profile_store or UserProfileMemoryStore()
        self.semantic_store = semantic_store or ChromaSemanticMemoryStore()
        self.extractor = extractor or MemoryExtractor()
        self._executor: ThreadPoolExecutor | None = None
        self._owns_executor = False
        if async_submitter is None:
            max_workers = max(1, settings.memory_extraction_max_workers)
            self._executor = extraction_executor or ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="memory-extract",
            )
            self._owns_executor = extraction_executor is None
            self.async_submitter = self._submit_executor
        else:
            self.async_submitter = async_submitter

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
        if not self._should_extract_long_term_memory(user_message, response):
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
            key = "occupation" if key == "job" else key
            if key not in STRUCTURED_MEMORY_FIELDS:
                continue
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
                if self._contains_unsafe_memory_text(value):
                    continue
            valid[key] = value
        return valid

    def _valid_semantic_memory(self, item: SemanticMemoryItem) -> bool:
        return (
            item.memory_key in SEMANTIC_MEMORY_KEYS
            and bool(item.memory_value.strip())
            and item.confidence >= settings.memory_min_confidence
            and not self._contains_unsafe_memory_text(item.memory_value)
            and not self._contains_unsafe_memory_text(item.evidence or "")
        )

    def _should_extract_long_term_memory(self, user_message: str, response: AgentChatResponse) -> bool:
        if not settings.memory_extraction_enabled:
            return False
        if response.route.normalized_intent == "UNKNOWN":
            return False
        if not response.answer.strip() or response.answer == UNKNOWN_INTENT_ANSWER:
            return False
        if self._contains_unsafe_memory_text(user_message) or self._contains_unsafe_memory_text(response.answer):
            return False
        if self._has_memory_signal(user_message):
            return True
        # Without an explicit user-memory signal, avoid extracting durable memory from ordinary RAG answers.
        return False

    def _has_memory_signal(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in MEMORY_SIGNAL_TERMS)

    def _contains_unsafe_memory_text(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in UNSAFE_MEMORY_PATTERNS)

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

    def _submit_executor(self, call, *args) -> None:
        if self._executor is None:
            call(*args)
            return
        try:
            future = self._executor.submit(call, *args)
        except RuntimeError as exc:
            logger.warning("Long-term memory extraction schedule failed: %s", exc)
            return
        future.add_done_callback(self._log_background_failure)

    def _log_background_failure(self, future: Future) -> None:
        if future.cancelled():
            return
        try:
            exc = future.exception()
        except Exception as callback_exc:
            logger.warning("Long-term memory extraction callback failed: %s", callback_exc)
            return
        if exc is not None:
            logger.warning("Long-term memory extraction task failed: %s", exc)

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=cancel_futures)
            self._executor = None

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
