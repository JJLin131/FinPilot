from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class MemoryContext(BaseModel):
    recent_messages: list[ChatTurn] = Field(default_factory=list)
