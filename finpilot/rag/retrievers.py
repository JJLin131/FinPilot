from __future__ import annotations

import logging

from finpilot.config import settings
from finpilot.models import RagMatch
from finpilot.rag.bm25 import Bm25ChunkIndex
from finpilot.rag.embeddings import OllamaEmbeddingClient
from finpilot.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class VectorCandidateRetriever:
    def __init__(self, embedding_client: OllamaEmbeddingClient, vector_store: ChromaVectorStore):
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(self, domain: str, query: str, limit: int) -> list[RagMatch]:
        if domain != "FINANCE":
            return []
        query_embedding = self.embedding_client.embed(query)
        return self.vector_store.query(query_embedding, limit)


class Bm25CandidateRetriever:
    def __init__(self, index: Bm25ChunkIndex):
        self.index = index

    def retrieve(self, domain: str, query: str, limit: int) -> list[RagMatch]:
        return self.index.search(domain, query, limit)


class HybridCandidateRetriever:
    def __init__(
        self,
        vector_retriever: VectorCandidateRetriever,
        bm25_retriever: Bm25CandidateRetriever,
        rrf_k: int | None = None,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = max(1, rrf_k or settings.rrf_k)

    def retrieve(self, domain: str, query: str, limit: int) -> list[RagMatch]:
        fusion: dict[str, tuple[RagMatch, float]] = {}
        self._add_safely(fusion, lambda: self.vector_retriever.retrieve(domain, query, limit), "vector")
        self._add_safely(fusion, lambda: self.bm25_retriever.retrieve(domain, query, limit), "bm25")
        return [
            RagMatch(
                document_id=match.document_id,
                title=match.title,
                source=match.source,
                text=match.text,
                score=round(score, 6),
            )
            for match, score in sorted(fusion.values(), key=lambda item: item[1], reverse=True)[:limit]
        ]

    def _add_safely(self, fusion: dict[str, tuple[RagMatch, float]], call, source: str) -> None:
        try:
            self._add(fusion, call())
        except Exception as exc:
            logger.warning("%s candidate retrieval failed; continuing with remaining retrievers: %s", source, exc)

    def _add(self, fusion: dict[str, tuple[RagMatch, float]], results: list[RagMatch]) -> None:
        for rank, match in enumerate(results):
            key = f"{match.document_id}:{hash(match.text)}"
            score = 1.0 / (self.rrf_k + rank + 1)
            if key in fusion:
                fusion[key] = (fusion[key][0], fusion[key][1] + score)
            else:
                fusion[key] = (match, score)

