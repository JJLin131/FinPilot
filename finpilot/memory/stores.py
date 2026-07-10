from __future__ import annotations

from abc import ABC
from datetime import datetime
from typing import Any

from finpilot.config import settings
from finpilot.memory.crypto import MemoryCipher
from finpilot.memory.definitions import STRUCTURED_MEMORY_FIELDS
from finpilot.memory.models import ChatSessionSummary, ChatTurn
from finpilot.mysql import connect_runtime_mysql


class BaseStore(ABC):
    def _connect(self):
        return connect_runtime_mysql()


class AgentChatMemoryStore(BaseStore):
    def __init__(self, cipher: MemoryCipher | None = None) -> None:
        self.cipher = cipher
        self._ensure_schema()

    def _memory_cipher(self) -> MemoryCipher:
        if self.cipher is None:
            self.cipher = MemoryCipher.from_base64_key(settings.memory_encryption_key)
        return self.cipher

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists agent_chat_message (
                        id bigint not null auto_increment primary key,
                        user_id varchar(64) not null,
                        chat_id varchar(128) not null,
                        request_id varchar(64) not null,
                        role varchar(16) not null,
                        content longtext not null,
                        created_at datetime(6) not null,
                        unique key uq_agent_chat_message_request_role(request_id, role),
                        index idx_agent_chat_message_user_chat_id(user_id, chat_id, id)
                    )
                    """
                )

    def get_messages(self, user_id: str, chat_id: str, limit: int = 20) -> list[ChatTurn]:
        cleaned_limit = max(1, min(limit, 200))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select role, content, created_at
                    from agent_chat_message
                    where user_id = %s and chat_id = %s
                    order by id desc
                    limit %s
                    """,
                    (user_id, chat_id, cleaned_limit),
                )
                rows = cursor.fetchall()
        messages = [
            ChatTurn(role=str(row[0]), content=self._memory_cipher().decrypt_text(str(row[1])), created_at=row[2])
            for row in reversed(rows)
        ]
        return messages

    def append_interaction(
        self,
        *,
        user_id: str,
        chat_id: str,
        request_id: str,
        user_content: str,
        assistant_content: str,
    ) -> None:
        params = (
            user_id,
            chat_id,
            request_id,
            "user",
            self._memory_cipher().encrypt_text(user_content),
            user_id,
            chat_id,
            request_id,
            "assistant",
            self._memory_cipher().encrypt_text(assistant_content),
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert ignore into agent_chat_message(user_id, chat_id, request_id, role, content, created_at)
                    values (%s, %s, %s, %s, %s, current_timestamp(6)),
                           (%s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    params,
                )

    def list_sessions(self, user_id: str, limit: int = 20) -> list[ChatSessionSummary]:
        cleaned_limit = max(1, min(limit, 100))
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    select chat_id, count(*), max(created_at)
                    from agent_chat_message
                    where user_id = %s
                    group by chat_id
                    order by max(created_at) desc
                    limit %s
                    """,
                    (user_id, cleaned_limit),
                )
                rows = cursor.fetchall()
        return [self._session_summary(user_id, row) for row in rows]

    def delete_messages(self, user_id: str, chat_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "delete from agent_chat_message where user_id = %s and chat_id = %s",
                    (user_id, chat_id),
                )

    def _session_summary(self, user_id: str, row) -> ChatSessionSummary:
        chat_id = str(row[0])
        messages = self.get_messages(user_id, chat_id, 20)
        return ChatSessionSummary(
            chat_id=chat_id,
            memory_id=f"chat:{user_id}:{chat_id}",
            message_count=int(row[1]),
            last_user_message=self._last_message(messages, "user"),
            last_assistant_message=self._last_message(messages, "assistant"),
            updated_at=row[2] if isinstance(row[2], datetime) else None,
        )

    def _last_message(self, messages: list[ChatTurn], role: str) -> str | None:
        for message in reversed(messages):
            if message.role == role:
                return message.content
        return None


class UserProfileMemoryStore(BaseStore):
    fields = tuple(STRUCTURED_MEMORY_FIELDS.keys())

    def __init__(self, cipher: MemoryCipher | None = None) -> None:
        self.cipher = cipher
        self._ensure_schema()

    def _memory_cipher(self) -> MemoryCipher:
        if self.cipher is None:
            self.cipher = MemoryCipher.from_base64_key(settings.memory_encryption_key)
        return self.cipher

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists user_profile (
                        user_id varchar(64) not null primary key,
                        profile_ciphertext longtext null,
                        age int null,
                        occupation varchar(128) null,
                        education varchar(64) null,
                        income_range varchar(64) null,
                        gender varchar(32) null,
                        city varchar(64) null,
                        marital_status varchar(32) null,
                        notes varchar(255) null,
                        created_at datetime(6) not null,
                        updated_at datetime(6) not null
                    )
                    """
                )
                cursor.execute(
                    """
                    select count(*)
                    from information_schema.columns
                    where table_schema = database()
                      and table_name = 'user_profile'
                      and column_name = 'profile_ciphertext'
                    """
                )
                row = cursor.fetchone()
                if not row or int(row[0]) == 0:
                    cursor.execute("alter table user_profile add column profile_ciphertext longtext null after user_id")

    def get_profile(self, user_id: str) -> dict[str, Any]:
        selected_fields = ", ".join(self.fields)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"select profile_ciphertext, {selected_fields} from user_profile where user_id = %s",
                    (user_id,),
                )
                row = cursor.fetchone()
        if not row:
            return {}
        if row[0]:
            return self._clean_values(self._memory_cipher().decrypt_json(str(row[0])))
        legacy = {field: row[index + 1] for index, field in enumerate(self.fields) if row[index + 1] is not None}
        cleaned = self._clean_values(legacy)
        if cleaned:
            self._write_encrypted_profile(user_id, cleaned)
        return cleaned

    def upsert_profile(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._clean_values(values)
        if not cleaned:
            return {}
        merged = {**self.get_profile(user_id), **cleaned}
        self._write_encrypted_profile(user_id, merged)
        return cleaned

    def clear_profile_fields(self, user_id: str, fields: list[str]) -> list[str]:
        cleaned = self._clean_fields(fields)
        if not cleaned:
            return []
        profile = self.get_profile(user_id)
        for field in cleaned:
            profile.pop(field, None)
        self._write_encrypted_profile(user_id, profile)
        return cleaned

    def _write_encrypted_profile(self, user_id: str, profile: dict[str, Any]) -> None:
        ciphertext = self._memory_cipher().encrypt_json(self._clean_values(profile))
        clear_legacy = ", ".join(f"{field} = null" for field in self.fields)
        sql = f"""
            insert into user_profile(user_id, profile_ciphertext, created_at, updated_at)
            values (%s, %s, current_timestamp(6), current_timestamp(6))
            on duplicate key update profile_ciphertext = values(profile_ciphertext),
                {clear_legacy}, updated_at = current_timestamp(6)
        """
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (user_id, ciphertext))

    def _clean_values(self, values: dict[str, Any]) -> dict[str, Any]:
        if "job" in values and "occupation" not in values:
            values = {**values, "occupation": values["job"]}
        cleaned: dict[str, Any] = {}
        for field in self.fields:
            value = values.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            if field == "age":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    continue
            cleaned[field] = value
        return cleaned

    def _clean_fields(self, fields: list[str]) -> list[str]:
        cleaned: list[str] = []
        for field in fields:
            field = "occupation" if field == "job" else field
            if field in self.fields and field not in cleaned:
                cleaned.append(field)
        return cleaned

