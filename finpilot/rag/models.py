from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from finpilot.models import RagMatch


class KnowledgeDocumentRequest(BaseModel):
    document_id: str = Field(min_length=1, max_length=191, pattern=r"^[A-Za-z0-9_.:@-]+$")
    domain: Literal["FINANCE"] = "FINANCE"
    title: str = Field(min_length=1, max_length=256)
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=200000)
    tags: list[str] = Field(default_factory=list, max_length=32)
    valid_from: date | None = None
    valid_to: date | None = None


class KnowledgeDocumentResult(BaseModel):
    document_id: str
    domain: str
    status: str
    chunk_count: int
    valid_from: date | None = None
    valid_to: date | None = None


class CandidateBundle(BaseModel):
    lexical: list[RagMatch] = Field(default_factory=list)
    title: list[RagMatch] = Field(default_factory=list)


class KnowledgeDocumentRecord(BaseModel):
    document_id: str
    domain: str
    title: str
    status: str
    valid_from: date | None = None
    valid_to: date | None = None

