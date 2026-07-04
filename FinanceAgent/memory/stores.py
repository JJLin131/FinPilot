from __future__ import annotations

import json
from abc import ABC
from typing import Any

import pymysql

from FinanceAgent.config import settings
from FinanceAgent.memory.definitions import STRUCTURED_MEMORY_FIELDS
from FinanceAgent.memory.models import ChatTurn


class BaseStore(ABC):
    def _connect(self):
        return pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            autocommit=True,
        )


class AgentChatMemoryStore(BaseStore):
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

    def _clean_values(self, values: dict[str, Any]) -> dict[str, Any]:
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
