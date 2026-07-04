from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatTurn(BaseModel):
    role: str
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SemanticMemoryItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    memory_key: str = Field(alias="memoryKey")
    memory_value: str = Field(alias="memoryValue")
    confidence: float = 0.0
    evidence: str | None = None


class ExtractedMemory(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    structured_memories: dict[str, Any] = Field(default_factory=dict, alias="structuredMemories")
    semantic_memories: list[SemanticMemoryItem] = Field(default_factory=list, alias="semanticMemories")


class SemanticMemoryRecord(BaseModel):
    memory_key: str
    memory_value: str
    confidence: float = 0.0
    evidence: str | None = None
    updated_at: datetime | None = None


class MemoryContext(BaseModel):
    recent_messages: list[ChatTurn] = Field(default_factory=list)
    structured_memory: dict[str, Any] = Field(default_factory=dict)
    semantic_memory: list[SemanticMemoryRecord] = Field(default_factory=list)
    long_term_memory: list[str] = Field(default_factory=list)
