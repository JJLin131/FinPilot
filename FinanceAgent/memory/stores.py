from __future__ import annotations

import json
from abc import ABC

import pymysql

from FinanceAgent.config import settings
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
