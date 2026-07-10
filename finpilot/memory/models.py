from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatTurn(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ChatSessionSummary(BaseModel):
    chat_id: str
    memory_id: str
    message_count: int = 0
    last_user_message: str | None = None
    last_assistant_message: str | None = None
    updated_at: datetime | None = None


class SemanticMemoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    memory_key: str = Field(alias="memoryKey")
    memory_value: str = Field(default="", alias="memoryValue")
    confidence: float = 0.0
    evidence: str | None = None

    @field_validator("memory_value", mode="before")
    @classmethod
    def _none_memory_value_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class ExtractedMemory(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    is_forget_intent: bool = Field(default=False, alias="isForgetIntent")
    structured_memories: dict[str, Any] = Field(default_factory=dict, alias="structuredMemories")
    semantic_memories: list[SemanticMemoryItem] = Field(default_factory=list, alias="semanticMemories")


class SemanticMemoryRecord(BaseModel):
    memory_key: str
    memory_value: str
    confidence: float = 0.0
    evidence: str | None = None
    updated_at: datetime | None = None
    score: float | None = None
    distance: float | None = None


class MemoryContext(BaseModel):
    recent_messages: list[ChatTurn] = Field(default_factory=list)
    structured_memory: dict[str, Any] = Field(default_factory=dict)
    semantic_memory: list[SemanticMemoryRecord] = Field(default_factory=list)
