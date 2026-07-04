from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from FinanceAgent.config import settings
from FinanceAgent.memory.models import SemanticMemoryItem, SemanticMemoryRecord
from FinanceAgent.rag.embeddings import OllamaEmbeddingClient
from FinanceAgent.rag.vector_store import ChromaVectorStore


class ChromaSemanticMemoryStore:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingClient | None = None,
        vector_store: ChromaVectorStore | None = None,
    ):
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=settings.chroma_memory_collection,
            timeout_seconds=settings.memory_chroma_timeout_seconds,
        )

    def search(self, user_id: str, query: str, limit: int) -> list[SemanticMemoryRecord]:
        if not settings.vector_enabled or not settings.memory_semantic_search_enabled or not query.strip():
            return []
        query_embedding = self.embedding_client.embed(query)
        collection_id = self.vector_store._ensure_collection()
        raw = self.vector_store._post_collection_action(
            collection_id,
            "query",
            {
                "query_embeddings": [query_embedding],
                "n_results": limit,
                "include": ["documents", "metadatas", "distances"],
                "where": self._where_user(user_id),
            },
        )
        return self._query_payload_to_records(raw)

    def get(self, user_id: str, memory_key: str) -> SemanticMemoryRecord | None:
        if not settings.vector_enabled:
            return None
        collection_id = self.vector_store._ensure_collection()
        raw = self.vector_store._post_collection_action(
            collection_id,
            "get",
            {
                "ids": [self.memory_vector_id(user_id, memory_key)],
                "include": ["documents", "metadatas"],
            },
        )
        records = self._get_payload_to_records(raw)
        return records[0] if records else None

    def upsert_summary(self, user_id: str, item: SemanticMemoryItem) -> SemanticMemoryRecord:
        existing = self.get(user_id, item.memory_key)
        merged_value = self._merge_memory(existing.memory_value if existing else "", item.memory_value)
        now = datetime.now(UTC)
        metadata = {
            "recordType": "user_memory",
            "userId": user_id,
            "memoryKey": item.memory_key,
            "memoryType": "semantic",
            "status": "ACTIVE",
            "confidence": float(item.confidence),
            "evidence": item.evidence or "",
            "createdAt": existing.updated_at.isoformat() if existing and existing.updated_at else now.isoformat(),
            "updatedAt": now.isoformat(),
            "observedAt": now.isoformat(),
        }
        embedding = self.embedding_client.embed(merged_value)
        self.vector_store.upsert_chunks(
            ids=[self.memory_vector_id(user_id, item.memory_key)],
            embeddings=[embedding],
            texts=[merged_value],
            metadatas=[metadata],
        )
        return SemanticMemoryRecord(
            memory_key=item.memory_key,
            memory_value=merged_value,
            confidence=item.confidence,
            evidence=item.evidence,
            updated_at=now,
        )

    def memory_vector_id(self, user_id: str, memory_key: str) -> str:
        return f"user-memory:{user_id}:{memory_key}"

    def _merge_memory(self, old_value: str, new_value: str) -> str:
        old_clean = " ".join(old_value.split())
        new_clean = " ".join(new_value.split())
        if not old_clean:
            return new_clean
        if not new_clean or new_clean in old_clean:
            return old_clean
        return f"{old_clean}\n新记忆：{new_clean}"

    def _where_user(self, user_id: str) -> dict[str, Any]:
        return {
            "$and": [
                {"recordType": "user_memory"},
                {"memoryType": "semantic"},
                {"status": "ACTIVE"},
                {"userId": user_id},
            ]
        }

    def _query_payload_to_records(self, payload: dict[str, Any]) -> list[SemanticMemoryRecord]:
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        records: list[SemanticMemoryRecord] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            if metadata.get("recordType") != "user_memory" or metadata.get("status") != "ACTIVE":
                continue
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            if score < settings.memory_semantic_min_score:
                continue
            records.append(self._record_from_payload(text, metadata))
        return records

    def _get_payload_to_records(self, payload: dict[str, Any]) -> list[SemanticMemoryRecord]:
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []
        records: list[SemanticMemoryRecord] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            if metadata.get("recordType") == "user_memory" and metadata.get("status") == "ACTIVE":
                records.append(self._record_from_payload(text, metadata))
        return records

    def _record_from_payload(self, text: Any, metadata: dict[str, Any]) -> SemanticMemoryRecord:
        updated_at = None
        updated_raw = metadata.get("updatedAt")
        if isinstance(updated_raw, str) and updated_raw:
            try:
                updated_at = datetime.fromisoformat(updated_raw)
            except ValueError:
                updated_at = None
        return SemanticMemoryRecord(
            memory_key=str(metadata.get("memoryKey") or ""),
            memory_value=str(text or ""),
            confidence=float(metadata.get("confidence") or 0.0),
            evidence=str(metadata.get("evidence") or "") or None,
            updated_at=updated_at,
        )
