from __future__ import annotations

from datetime import UTC, datetime

from finpilot.memory.crypto import MemoryCipher
from finpilot.memory.stores import AgentChatMemoryStore


class RecordingCursor:
    def __init__(self, rows=None, row=None) -> None:
        self.rows = rows or []
        self.row = row
        self.executions: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql: str, params=None) -> None:
        self.executions.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.row


class RecordingConnection:
    def __init__(self, cursor: RecordingCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self._cursor


class FakeCipher:
    def encrypt_text(self, value: str) -> str:
        return f"enc:{value}"

    def decrypt_text(self, value: str) -> str:
        return value.removeprefix("enc:")

    def encrypt_json(self, value: dict) -> str:
        fields = ",".join(f"{key}={item}" for key, item in sorted(value.items()))
        return f"json:{fields}"

    def decrypt_json(self, value: str) -> dict:
        result = {}
        for item in value.removeprefix("json:").split(","):
            if not item:
                continue
            key, raw = item.split("=", 1)
            result[key] = int(raw) if raw.isdigit() else raw
        return result


def _store(cursor: RecordingCursor) -> AgentChatMemoryStore:
    store = AgentChatMemoryStore.__new__(AgentChatMemoryStore)
    store._connect = lambda: RecordingConnection(cursor)
    store.cipher = FakeCipher()
    return store


def test_append_interaction_inserts_two_idempotent_message_rows():
    cursor = RecordingCursor()
    store = _store(cursor)

    store.append_interaction(
        user_id="user-1",
        chat_id="chat-1",
        request_id="req-1",
        user_content="hello",
        assistant_content="world",
    )

    sql, params = cursor.executions[0]
    assert "insert ignore into agent_chat_message" in sql.lower()
    assert params == (
        "user-1",
        "chat-1",
        "req-1",
        "user",
        "enc:hello",
        "user-1",
        "chat-1",
        "req-1",
        "assistant",
        "enc:world",
    )


def test_get_messages_reverses_descending_rows_back_to_conversation_order():
    now = datetime.now(UTC)
    cursor = RecordingCursor(rows=[("assistant", "enc:answer", now), ("user", "enc:question", now)])
    store = _store(cursor)

    messages = store.get_messages("user-1", "chat-1", 6)

    assert [(message.role, message.content) for message in messages] == [("user", "question"), ("assistant", "answer")]
    sql, params = cursor.executions[0]
    assert "where user_id = %s and chat_id = %s" in sql.lower()
    assert "order by id desc" in sql.lower()
    assert params == ("user-1", "chat-1", 6)


def test_memory_cipher_round_trips_text_with_randomized_aes_gcm_payloads():
    cipher = MemoryCipher.from_base64_key("MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=")

    first = cipher.encrypt_text("银行卡号 6222020202020202020")
    second = cipher.encrypt_text("银行卡号 6222020202020202020")

    assert first.startswith("enc:v1:")
    assert second.startswith("enc:v1:")
    assert first != second
    assert cipher.decrypt_text(first) == "银行卡号 6222020202020202020"
    assert cipher.decrypt_text(second) == "银行卡号 6222020202020202020"


def test_memory_cipher_rejects_plaintext_during_runtime_but_allows_explicit_legacy_migration():
    cipher = MemoryCipher.from_base64_key("MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=")

    import pytest

    with pytest.raises(ValueError, match="not encrypted"):
        cipher.decrypt_text("legacy plaintext")
    assert cipher.decrypt_legacy_text("legacy plaintext") == "legacy plaintext"


def test_profile_store_decrypts_encrypted_json_instead_of_exposing_ciphertext():
    from finpilot.memory.stores import UserProfileMemoryStore

    legacy_columns = (None,) * len(UserProfileMemoryStore.fields)
    cursor = RecordingCursor(row=("json:age=32,city=南京", *legacy_columns))
    store = UserProfileMemoryStore.__new__(UserProfileMemoryStore)
    store._connect = lambda: RecordingConnection(cursor)
    store.cipher = FakeCipher()

    profile = store.get_profile("user-1")

    assert profile == {"age": 32, "city": "南京"}
    assert "profile_ciphertext" in cursor.executions[0][0]


def test_delete_messages_is_scoped_by_user_and_chat():
    cursor = RecordingCursor()
    store = _store(cursor)

    store.delete_messages("user-1", "chat-1")

    sql, params = cursor.executions[0]
    assert "delete from agent_chat_message where user_id = %s and chat_id = %s" in sql.lower()
    assert params == ("user-1", "chat-1")


def test_list_sessions_uses_append_only_message_aggregates():
    now = datetime.now(UTC)
    cursor = RecordingCursor(rows=[("chat-1", 4, now)])
    store = _store(cursor)
    store.get_messages = lambda user_id, chat_id, limit: [
        type("Turn", (), {"role": "user", "content": "question"})(),
        type("Turn", (), {"role": "assistant", "content": "answer"})(),
    ]

    sessions = store.list_sessions("user-1", 10)

    assert sessions[0].chat_id == "chat-1"
    assert sessions[0].memory_id == "chat:user-1:chat-1"
    assert sessions[0].message_count == 4
    assert sessions[0].last_user_message == "question"


def test_profile_update_merges_fields_and_delete_removes_only_target_field():
    from finpilot.memory.stores import UserProfileMemoryStore

    store = UserProfileMemoryStore.__new__(UserProfileMemoryStore)
    store.cipher = FakeCipher()
    stored: list[dict] = []
    current = {"city": "杭州", "occupation": "财务经理"}
    store.get_profile = lambda user_id: dict(current)
    store._write_encrypted_profile = lambda user_id, profile: stored.append(dict(profile))

    store.upsert_profile("user-1", {"city": "南京"})
    store.clear_profile_fields("user-1", ["occupation"])

    assert stored == [
        {"city": "南京", "occupation": "财务经理"},
        {"city": "杭州"},
    ]
