from __future__ import annotations

from threading import Event

from finpilot.memory.models import ExtractedMemory, SemanticMemoryItem
from finpilot.memory.service import MemoryManager
from finpilot.models import AgentChatResponse, RouteDecision


class RecordingExtractor:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def extract(self, **kwargs) -> ExtractedMemory:
        self.calls.append(kwargs)
        return ExtractedMemory(isForgetIntent=True, structuredMemories={"city": None})


class NoopProfileStore:
    def clear_profile_fields(self, user_id: str, fields: list[str]) -> list[str]:
        return fields


class RecordingChatStore:
    def __init__(self) -> None:
        self.appends: list[dict] = []
        self.loads: list[tuple[str, str, int]] = []

    def get_messages(self, user_id: str, chat_id: str, limit: int):
        self.loads.append((user_id, chat_id, limit))
        return []

    def append_interaction(self, **kwargs) -> None:
        self.appends.append(kwargs)


def _unknown_response(answer: str) -> AgentChatResponse:
    return AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="UNSUPPORTED",
        answer=answer,
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="UNKNOWN",
            reason="memory-only request",
            confidence=0.0,
            valid=False,
            target_agent="UNSUPPORTED",
            classifier_intent="UNKNOWN",
        ),
    )


def _success_response(request_id: str = "req-1") -> AgentChatResponse:
    return AgentChatResponse(
        request_id=request_id,
        trace_id="trace-1",
        domain="FINANCE",
        status="SUCCEEDED",
        answer="已记录",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="FINANCE_KNOWLEDGE_QA",
            reason="test",
            confidence=1.0,
            valid=True,
            target_agent="QueryAgent",
            classifier_intent="FINANCE_KNOWLEDGE_QA",
        ),
    )


def test_forget_extraction_receives_original_message_for_reusable_memory():
    extractor = RecordingExtractor()
    original = "忘记我的银行卡号 6222020202020202020"
    manager = MemoryManager(
        chat_store=object(),
        profile_store=NoopProfileStore(),
        semantic_store=object(),
        extractor=extractor,
        async_submitter=lambda call, *args: call(*args),
    )

    manager.remember_interaction(
        "chat:user-1:chat-1",
        "user-1",
        "chat-1",
        original,
        _unknown_response("已处理银行卡号 6222020202020202020 的忘记请求"),
    )

    assert original == "忘记我的银行卡号 6222020202020202020"
    assert extractor.calls[0]["user_message"] == original
    assert extractor.calls[0]["assistant_answer"] == "已处理银行卡号 6222020202020202020 的忘记请求"


def test_short_term_memory_appends_messages_without_loading_existing_history():
    chat_store = RecordingChatStore()
    manager = MemoryManager(
        chat_store=chat_store,
        profile_store=object(),
        semantic_store=object(),
        extractor=object(),
        async_submitter=lambda *args: None,
    )

    manager.remember_interaction(
        "chat:user-1:chat-1",
        "user-1",
        "chat-1",
        "查询银行卡 6222020202020202020",
        _success_response(),
    )

    assert chat_store.loads == []
    assert chat_store.appends == [
        {
            "user_id": "user-1",
            "chat_id": "chat-1",
            "request_id": "req-1",
            "user_content": "查询银行卡 6222020202020202020",
            "assistant_content": "已记录",
        }
    ]


def test_memory_load_uses_user_and_chat_scope():
    chat_store = RecordingChatStore()
    manager = MemoryManager(
        chat_store=chat_store,
        profile_store=type("Profile", (), {"get_profile": lambda self, user_id: {}})(),
        semantic_store=type("Semantic", (), {"search": lambda self, user_id, query, limit: []})(),
        extractor=object(),
        async_submitter=lambda *args: None,
    )

    manager.load(user_id="user-1", chat_id="chat-1", user_message="hello")

    assert chat_store.loads == [("user-1", "chat-1", 6)]


class OrderedExtractor:
    def __init__(self) -> None:
        self.first_started = Event()
        self.release_first = Event()
        self.second_started = Event()
        self.other_user_started = Event()
        self.calls: list[str] = []

    def extract(self, **kwargs) -> ExtractedMemory:
        message = kwargs["user_message"]
        self.calls.append(message)
        if message == "记住 first":
            self.first_started.set()
            self.release_first.wait(timeout=2)
        elif message == "记住 second":
            self.second_started.set()
        elif message == "记住 other":
            self.other_user_started.set()
        return ExtractedMemory()


def test_long_term_tasks_are_ordered_per_user_but_allow_other_users_to_run_concurrently():
    extractor = OrderedExtractor()
    manager = MemoryManager(
        chat_store=RecordingChatStore(),
        profile_store=object(),
        semantic_store=object(),
        extractor=extractor,
    )

    manager.remember_interaction("chat:user-1:chat-1", "user-1", "chat-1", "记住 first", _success_response("req-1"))
    assert extractor.first_started.wait(timeout=1)

    manager.remember_interaction("chat:user-1:chat-1", "user-1", "chat-1", "记住 second", _success_response("req-2"))
    manager.remember_interaction("chat:user-2:chat-2", "user-2", "chat-2", "记住 other", _success_response("req-3"))

    assert extractor.other_user_started.wait(timeout=1)
    assert not extractor.second_started.wait(timeout=0.1)

    extractor.release_first.set()
    assert extractor.second_started.wait(timeout=1)
    manager.shutdown(wait=True)
    assert extractor.calls.index("记住 first") < extractor.calls.index("记住 second")


def test_encrypted_semantic_memory_accepts_explicit_card_identifier_but_rejects_credentials():
    manager = MemoryManager(
        chat_store=object(),
        profile_store=object(),
        semantic_store=object(),
        extractor=object(),
        async_submitter=lambda *args: None,
    )

    assert manager._valid_semantic_memory(
        SemanticMemoryItem(
            memoryKey="userCard",
            memoryValue="用户要求记住银行卡号 6222020202020202020。",
            confidence=0.95,
        )
    )
    assert not manager._valid_semantic_memory(
        SemanticMemoryItem(
            memoryKey="userCard",
            memoryValue="用户的 password 是 secret-value。",
            confidence=0.95,
        )
    )


def test_memory_manager_rejects_new_background_work_after_shutdown():
    extractor = RecordingExtractor()
    manager = MemoryManager(
        chat_store=RecordingChatStore(),
        profile_store=NoopProfileStore(),
        semantic_store=object(),
        extractor=extractor,
    )
    manager.shutdown(wait=True)

    manager.remember_interaction(
        "chat:user-1:chat-1",
        "user-1",
        "chat-1",
        "记住我的城市",
        _success_response(),
    )

    assert extractor.calls == []
