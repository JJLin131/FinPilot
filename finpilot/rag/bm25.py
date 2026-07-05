from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finpilot.config import settings
from finpilot.models import RagMatch
from finpilot.rag.models import KnowledgeDocumentRequest


def _tokens(text: str) -> list[str]:
    normalized = (text or "").lower()
    ascii_tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{1,2}", normalized)
    return ascii_tokens + cjk_tokens


@dataclass
class Bm25Document:
    chunk_id: str
    document_id: str
    domain: str
    title: str
    source: str
    text: str
    tags: str


class Bm25ChunkIndex:
    def __init__(self, index_path: Path | None = None, k1: float = 1.5, b: float = 0.75):
        self.index_path = index_path or settings.bm25_index_path
        self.k1 = k1
        self.b = b
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def replace_document(self, request: KnowledgeDocumentRequest, chunk_ids: list[str], chunks: list[str]) -> None:
        docs = [doc for doc in self._load() if doc.document_id != request.document_id]
        tags = " ".join(request.tags or [])
        for index, text in enumerate(chunks):
            docs.append(
                Bm25Document(
                    chunk_id=chunk_ids[index],
                    document_id=request.document_id,
                    domain=request.domain,
                    title=request.title,
                    source=request.source,
                    text=text,
                    tags=tags,
                )
            )
        self._save(docs)

    def delete_document(self, document_id: str) -> None:
        self._save([doc for doc in self._load() if doc.document_id != document_id])

    def search(self, domain: str, query: str, limit: int) -> list[RagMatch]:
        docs = [doc for doc in self._load() if doc.domain == domain]
        if not docs:
            return []

        tokenized_docs = [self._weighted_tokens(doc) for doc in docs]
        doc_count = len(tokenized_docs)
        avg_len = sum(len(tokens) for tokens in tokenized_docs) / max(doc_count, 1)
        doc_freq: dict[str, int] = {}
        for tokens in tokenized_docs:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        query_tokens = _tokens(query)
        scored: list[tuple[Bm25Document, float]] = []
        for doc, tokens in zip(docs, tokenized_docs, strict=False):
            score = self._score(query_tokens, tokens, doc_freq, doc_count, avg_len)
            if score > 0:
                scored.append((doc, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            RagMatch(
                document_id=doc.document_id,
                title=doc.title,
                source=doc.source,
                text=doc.text,
                score=round(score, 4),
            )
            for doc, score in scored[:limit]
        ]

    def _weighted_tokens(self, doc: Bm25Document) -> list[str]:
        return _tokens(doc.text) + _tokens(doc.title) * 2 + _tokens(doc.tags)

    def _score(self, query_tokens: list[str], doc_tokens: list[str], doc_freq: dict[str, int], doc_count: int, avg_len: float) -> float:
        if not query_tokens or not doc_tokens:
            return 0.0
        score = 0.0
        doc_len = len(doc_tokens)
        term_freq: dict[str, int] = {}
        for token in doc_tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        for token in query_tokens:
            freq = term_freq.get(token, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(token, 0)
            idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / max(avg_len, 1.0))
            score += idf * (freq * (self.k1 + 1)) / denominator
        return score

    def _load(self) -> list[Bm25Document]:
        if not self.index_path.exists():
            return []
        payload = json.loads(self.index_path.read_text(encoding="utf-8") or "[]")
        return [Bm25Document(**{key: value for key, value in item.items() if key != "tenant_id"}) for item in payload]

    def _save(self, docs: list[Bm25Document]) -> None:
        payload: list[dict[str, Any]] = [doc.__dict__ for doc in docs]
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

