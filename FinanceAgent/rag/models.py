from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from FinanceAgent.models import RagMatch


class KnowledgeDocumentRequest(BaseModel):
    document_id: str
    domain: str = "FINANCE"
    tenant_id: str = "__GLOBAL__"
    title: str
    source: str
    content: str
    tags: list[str] = Field(default_factory=list)
    valid_from: date | None = None
    valid_to: date | None = None


class KnowledgeDocumentResult(BaseModel):
    document_id: str
    domain: str
    tenant_id: str
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
    tenant_id: str | None = None
    title: str
    status: str
    valid_from: date | None = None
    valid_to: date | None = None
