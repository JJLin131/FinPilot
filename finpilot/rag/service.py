from __future__ import annotations

import logging
import uuid
from datetime import date

from finpilot.config import settings
from finpilot.llm import QueryRewriteService
from finpilot.models import AgentIssue, RagMatch
from finpilot.rag.bm25 import Bm25ChunkIndex
from finpilot.rag.chunker import KnowledgeChunker
from finpilot.rag.curation import RagCurationAgent
from finpilot.rag.embeddings import OllamaEmbeddingClient
from finpilot.rag.models import KnowledgeDocumentRequest, KnowledgeDocumentResult
from finpilot.rag.registry import KnowledgeDocumentRegistry
from finpilot.rag.retrievers import Bm25CandidateRetriever, HybridCandidateRetriever, VectorCandidateRetriever
from finpilot.rag.vector_store import ChromaVectorStore
from finpilot.reranker import RemoteReranker

logger = logging.getLogger(__name__)


class RagKnowledgeService:
    def __init__(
        self,
        registry: KnowledgeDocumentRegistry | None = None,
        chunker: KnowledgeChunker | None = None,
        query_rewriter: QueryRewriteService | None = None,
        retriever: HybridCandidateRetriever | None = None,
        reranker: RemoteReranker | None = None,
        curation_agent: RagCurationAgent | None = None,
        embedding_client: OllamaEmbeddingClient | None = None,
        vector_store: ChromaVectorStore | None = None,
        bm25_index: Bm25ChunkIndex | None = None,
    ):
        self.registry = registry or KnowledgeDocumentRegistry()
        self.chunker = chunker or KnowledgeChunker()
        self.query_rewriter = query_rewriter or QueryRewriteService()
        self.reranker = reranker or RemoteReranker()
        self.curation_agent = curation_agent or RagCurationAgent()
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.vector_store = vector_store or ChromaVectorStore()
        self.bm25_index = bm25_index or Bm25ChunkIndex()
        self.retriever = retriever or HybridCandidateRetriever(
            VectorCandidateRetriever(self.embedding_client, self.vector_store),
            Bm25CandidateRetriever(self.bm25_index),
        )
        self._issues: list[AgentIssue] = []

    def bootstrap_resources(self) -> list[KnowledgeDocumentResult]:
        results: list[KnowledgeDocumentResult] = []
        for path in sorted(settings.knowledge_dir.glob("*.md")):
            if not path.name[:2].isdigit():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            request = KnowledgeDocumentRequest(
                document_id=self._resource_document_id(path.name),
                domain="FINANCE",
                title=path.stem,
                source=str(path),
                content=content,
                tags=self._resource_tags(path.stem),
            )
            results.append(self.ingest(request))
        return results

    def ingest(self, request: KnowledgeDocumentRequest) -> KnowledgeDocumentResult:
        self._validate(request)
        curated = self.curation_agent.curate(request)
        self._expire_older_title_conflicts_best_effort(curated)
        old_chunk_ids = self.registry.chunk_ids(curated.document_id)
        if old_chunk_ids:
            self._delete_document_vectors_best_effort(curated.document_id, old_chunk_ids)
        chunks = self.chunker.split(curated.content)
        chunk_ids = self.registry.chunk_ids_for(curated.document_id, len(chunks))
        self._index_vectors_best_effort(curated, chunk_ids, chunks)
        self.bm25_index.replace_document(curated, chunk_ids, chunks)
        self.registry.save_document(curated, len(chunks))
        self.registry.replace_chunks(curated, chunks, chunk_ids)
        return KnowledgeDocumentResult(
            document_id=curated.document_id,
            domain=curated.domain,
            status="ACTIVE",
            chunk_count=len(chunks),
            valid_from=curated.valid_from,
            valid_to=curated.valid_to,
        )

    def search(self, query: str, limit: int = 5) -> list[RagMatch]:
        matches, issues = self.search_with_issues(query, limit=limit)
        self._issues = issues
        return matches

    def search_with_issues(self, query: str, limit: int = 5) -> tuple[list[RagMatch], list[AgentIssue]]:
        """返回本次检索的诊断信息，避免并发请求竞争实例级 issue 缓存。"""
        issues: list[AgentIssue] = []
        candidates: dict[str, RagMatch] = {}
        for rewritten in self.query_rewriter.rewrite(query):
            retrieved = self.retriever.retrieve("FINANCE", rewritten, max(limit * 4, 12))
            issues.extend(self._consume_component_issues(self.retriever))
            for match in retrieved:
                if not self.registry.is_active(match.document_id, "FINANCE", date.today()):
                    continue
                key = f"{match.document_id}:{hash(match.text)}"
                current = candidates.get(key)
                if current is None or match.score > current.score:
                    candidates[key] = match
        reranked = self.reranker.rerank(query, list(candidates.values()), limit)
        issues.extend(self._consume_component_issues(self.reranker))
        return reranked, issues

    def consume_issues(self) -> list[AgentIssue]:
        issues = list(self._issues)
        self._issues = []
        return issues

    def delete_expired_documents(self) -> int:
        expired = self.registry.expired_document_ids(date.today())
        for document_id in expired:
            chunk_ids = self.registry.chunk_ids(document_id)
            self._delete_document_vectors_best_effort(document_id, chunk_ids)
            self.bm25_index.delete_document(document_id)
            self.registry.mark_expired(document_id)
        return len(expired)

    def _index_vectors_best_effort(self, request: KnowledgeDocumentRequest, chunk_ids: list[str], chunks: list[str]) -> None:
        if not chunks:
            return
        try:
            embeddings = self.embedding_client.embed_many([request.title, *chunks])
            title_embedding = embeddings[0]
            chunk_embeddings = embeddings[1:]
            metadatas = [
                {
                    "documentId": request.document_id,
                    "recordType": "chunk",
                    "status": "ACTIVE",
                }
                for _ in range(len(chunks))
            ]
            self.vector_store.upsert_document_info(
                document_id=request.document_id,
                embedding=title_embedding,
                title=request.title,
            )
            self.vector_store.upsert_chunks(ids=chunk_ids, embeddings=chunk_embeddings, texts=chunks, metadatas=metadatas)
        except Exception as exc:
            logger.warning("Vector indexing unavailable for document %s; BM25 remains active: %s", request.document_id, exc)

    def _expire_older_title_conflicts_best_effort(self, request: KnowledgeDocumentRequest) -> None:
        if not settings.title_conflict_enabled or not request.title.strip():
            return
        try:
            title_embedding = self.embedding_client.embed(request.title)
            similar_titles = self.vector_store.query_similar_document_info(
                title_embedding,
                limit=settings.title_conflict_search_limit,
            )
            similar_ids = [
                item["document_id"]
                for item in similar_titles
                if item["document_id"] != request.document_id and item["score"] >= settings.title_conflict_min_score
            ]
            if not similar_ids:
                return
            records = self.registry.documents_by_ids(similar_ids)
            for document_id in self.curation_agent.older_conflict_document_ids(request, records):
                self._expire_document(document_id)
                logger.info(
                    "Expired older similar knowledge document %s while ingesting %s.",
                    document_id,
                    request.document_id,
                )
        except Exception as exc:
            logger.warning("Title conflict detection unavailable for document %s; continuing ingest: %s", request.document_id, exc)

    def _expire_document(self, document_id: str) -> None:
        chunk_ids = self.registry.chunk_ids(document_id)
        self._delete_document_vectors_best_effort(document_id, chunk_ids)
        self.bm25_index.delete_document(document_id)
        self.registry.mark_expired(document_id)

    def _delete_document_vectors_best_effort(self, document_id: str, chunk_ids: list[str]) -> None:
        try:
            self.vector_store.delete_document_vectors(document_id, chunk_ids)
        except Exception as exc:
            logger.warning("Vector store unavailable while deleting vectors for document %s; continuing cleanup: %s", document_id, exc)

    def _validate(self, request: KnowledgeDocumentRequest) -> None:
        if not request.document_id.strip():
            raise ValueError("Knowledge document requires document_id.")
        if not request.domain.strip():
            raise ValueError("Knowledge document requires domain.")

    def _consume_component_issues(self, component) -> list[AgentIssue]:
        consume = getattr(component, "consume_issues", None)
        if callable(consume):
            return consume()
        return []

    def _resource_document_id(self, file_name: str) -> str:
        return f"resource-finance-{uuid.uuid5(uuid.NAMESPACE_URL, file_name)}"

    def _resource_tags(self, stem: str) -> list[str]:
        return [part for part in stem.replace("_", " ").split() if part]

