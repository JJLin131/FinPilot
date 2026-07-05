from __future__ import annotations

import json
import logging
from datetime import date

from pydantic import BaseModel, Field

from finpilot.config import settings
from finpilot.llm import DeepSeekChatClient, OllamaClient
from finpilot.rag.models import KnowledgeDocumentRecord, KnowledgeDocumentRequest

logger = logging.getLogger(__name__)


class CuratedKnowledgeDocument(BaseModel):
    title: str | None = None
    normalized_markdown: str | None = None
    tags: list[str] = Field(default_factory=list)


class RagCurationAgent:
    SYSTEM_PROMPT = """
You are a knowledge-document structure curation agent.
Your task is to normalize documents into well-structured Markdown so a downstream
heading-aware chunker can produce semantically independent chunks.

Preserve every source rule, amount, date, role, condition, result, exception, table row,
workflow step, status code, and legal meaning verbatim.
Never summarize, invent, remove, supersede, reinterpret, or resolve contradictions.
Do not generate chunks.

For unstructured documents, separate mixed topics using meaningful Markdown headings.
Keep each condition together with its result and exceptions under the same heading.
Preserve Markdown tables, ordered workflows, lists, and question-answer blocks.
For already well-structured Markdown, keep its structure and wording whenever possible.
Assign concise retrieval tags that reflect the document's actual topics.
Return JSON only with this schema:
{"title":"...", "normalized_markdown":"...", "tags":["..."]}
"""

    def __init__(self, client=None):
        self.client = client or self._build_client()

    def curate(self, source: KnowledgeDocumentRequest) -> KnowledgeDocumentRequest:
        if not settings.rag_curation_enabled:
            return source
        try:
            curated = self._curate_content(source.content)
        except Exception as exc:
            logger.warning("RAG document curation unavailable for %s; preserving original document: %s", source.document_id, exc)
            return source

        normalized = curated.normalized_markdown or source.content
        content = normalized if self._preserves_all_source_paragraphs(source.content, normalized) else source.content
        return KnowledgeDocumentRequest(
            document_id=source.document_id,
            domain=source.domain,
            title=curated.title or source.title,
            source=source.source,
            content=content,
            tags=curated.tags or source.tags,
            valid_from=source.valid_from,
            valid_to=source.valid_to,
        )

    def older_conflict_document_ids(
        self,
        source: KnowledgeDocumentRequest,
        candidates: list[KnowledgeDocumentRecord],
    ) -> list[str]:
        incoming_date = source.valid_from or date.today()
        expired_ids: list[str] = []
        for candidate in candidates:
            if candidate.document_id == source.document_id:
                continue
            if candidate.status != "ACTIVE" or candidate.domain != source.domain:
                continue
            candidate_date = candidate.valid_from or date.min
            if candidate_date <= incoming_date:
                expired_ids.append(candidate.document_id)
        return expired_ids

    def _build_client(self):
        if settings.rag_curation_provider.lower() == "deepseek":
            return DeepSeekChatClient(
                model_name=settings.rag_curation_model_name,
                timeout_seconds=settings.rag_curation_timeout_seconds,
            )
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.rag_curation_model_name,
            timeout_seconds=settings.rag_curation_timeout_seconds,
        )

    def _curate_content(self, content: str) -> CuratedKnowledgeDocument:
        raw = self.client.generate(content, model_name=settings.rag_curation_model_name, system_prompt=self.SYSTEM_PROMPT)
        return CuratedKnowledgeDocument.model_validate(self._extract_json(raw))

    def _extract_json(self, raw: str) -> dict:
        text = raw.strip()
        if "```" in text:
            text = text.replace("```json", "```")
            parts = [part.strip() for part in text.split("```") if part.strip()]
            for part in parts:
                if part.startswith("{") and part.endswith("}"):
                    return json.loads(part)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)

    def _preserves_all_source_paragraphs(self, source: str, curated: str) -> bool:
        if not source or not curated:
            return False
        paragraphs = [line.strip() for line in source.splitlines() if line.strip()]
        return all(paragraph in curated for paragraph in paragraphs)

