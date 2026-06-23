from __future__ import annotations

import re
from pathlib import Path

from python_agent_service.config import settings
from python_agent_service.llm import QueryRewriteService
from python_agent_service.models import RagMatch
from python_agent_service.reranker import RemoteReranker


def _tokenize(text: str) -> set[str]:
    return {token for token in re.split(r"[\W_]+", text.lower()) if token}


class RagKnowledgeService:
    def __init__(self, knowledge_dir: Path | None = None):
        self.knowledge_dir = knowledge_dir or settings.knowledge_dir
        self.documents = self._load_documents()
        self.query_rewriter = QueryRewriteService()
        self.reranker = RemoteReranker()

    def search(self, query: str, limit: int = 5) -> list[RagMatch]:
        rewrites = self.query_rewriter.rewrite(query)
        merged: dict[str, RagMatch] = {}
        for rewritten_query in rewrites:
            query_tokens = _tokenize(rewritten_query)
            for document_id, title, source, content in self.documents:
                content_tokens = _tokenize(content)
                overlap = len(query_tokens & content_tokens)
                if overlap == 0:
                    continue
                score = overlap / max(len(query_tokens), 1)
                match = RagMatch(
                    document_id=document_id,
                    title=title,
                    source=source,
                    text=content[:300],
                    score=round(score, 4),
                )
                previous = merged.get(document_id)
                if previous is None or match.score > previous.score:
                    merged[document_id] = match
        return self.reranker.rerank(query, list(merged.values()), limit)

    def _load_documents(self) -> list[tuple[str, str, str, str]]:
        documents: list[tuple[str, str, str, str]] = []
        if not self.knowledge_dir.exists():
            return documents
        for index, path in enumerate(sorted(self.knowledge_dir.glob("*.md")), start=1):
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="utf-8", errors="ignore")
            documents.append((f"doc-{index}", path.stem, str(path), content))
        return documents
