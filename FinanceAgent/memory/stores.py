from __future__ import annotations

import json
from abc import ABC
from typing import Any

from FinanceAgent.memory.definitions import STRUCTURED_MEMORY_FIELDS
from FinanceAgent.memory.models import ChatTurn
from FinanceAgent.mysql import connect_runtime_mysql


class BaseStore(ABC):
    def _connect(self):
        return connect_runtime_mysql()


class AgentChatMemoryStore(BaseStore):
    def __init__(self) -> None:
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists agent_chat_memory (
                        memory_id varchar(255) not null primary key,
                        messages_json longtext not null,
                        updated_at datetime(6) not null
                    )
                    """
                )

    def get_messages(self, memory_id: str) -> list[ChatTurn]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select messages_json from agent_chat_memory where memory_id = %s", (memory_id,))
                row = cursor.fetchone()
        if not row or not row[0]:
            return []
        payload = json.loads(row[0])
        return [ChatTurn.model_validate(item) for item in payload]

    def update_messages(self, memory_id: str, messages: list[ChatTurn]) -> None:
        payload = json.dumps([message.model_dump(mode="json") for message in messages], ensure_ascii=False)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_chat_memory(memory_id, messages_json, updated_at)
                    values (%s, %s, current_timestamp(6))
                    on duplicate key update messages_json = values(messages_json), updated_at = current_timestamp(6)
                    """,
                    (memory_id, payload),
                )

    def delete_messages(self, memory_id: str) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("delete from agent_chat_memory where memory_id = %s", (memory_id,))


class UserProfileMemoryStore(BaseStore):
    fields = tuple(STRUCTURED_MEMORY_FIELDS.keys())

    def __init__(self) -> None:
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists user_profile (
                        user_id varchar(64) not null primary key,
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

    def get_profile(self, user_id: str) -> dict[str, Any]:
        selected_fields = ", ".join(self.fields)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(f"select {selected_fields} from user_profile where user_id = %s", (user_id,))
                row = cursor.fetchone()
        if not row:
            return {}
        return {field: row[index] for index, field in enumerate(self.fields) if row[index] is not None}

    def upsert_profile(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        cleaned = self._clean_values(values)
        if not cleaned:
            return {}
        columns = ["user_id", *cleaned.keys(), "created_at", "updated_at"]
        values_sql = ", ".join(["%s"] * (len(cleaned) + 1) + ["current_timestamp(6)", "current_timestamp(6)"])
        update_clause = ", ".join([f"{field} = values({field})" for field in cleaned])
        sql = f"""
            insert into user_profile({", ".join(columns)})
            values ({values_sql})
            on duplicate key update {update_clause}, updated_at = current_timestamp(6)
        """
        params = [user_id, *cleaned.values()]
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, params)
        return cleaned

    def clear_profile_fields(self, user_id: str, fields: list[str]) -> list[str]:
        cleaned = self._clean_fields(fields)
        if not cleaned:
            return []
        update_clause = ", ".join(f"{field} = null" for field in cleaned)
        sql = f"update user_profile set {update_clause}, updated_at = current_timestamp(6) where user_id = %s"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql, (user_id,))
        return cleaned

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
