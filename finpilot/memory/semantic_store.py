from __future__ import annotations

import re
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

from finpilot.config import settings
from finpilot.memory.crypto import MemoryCipher
from finpilot.memory.models import SemanticMemoryItem, SemanticMemoryRecord
from finpilot.rag.embeddings import OllamaEmbeddingClient
from finpilot.rag.vector_store import ChromaVectorStore, cosine_score

MEMORY_FRAGMENT_SPLIT_RE = re.compile(r"[\r\n]+|[;.]+")
MEMORY_FRAGMENT_PREFIX_RE = re.compile(r"^\s*(?:[-*]\s*)?(?:memory|remember)\s*[:]?\s*")


class ChromaSemanticMemoryStore:
    def __init__(
        self,
        embedding_client: OllamaEmbeddingClient | None = None,
        vector_store: ChromaVectorStore | None = None,
        cipher: MemoryCipher | None = None,
    ):
        self.embedding_client = embedding_client or OllamaEmbeddingClient()
        self.cipher = cipher
        self.vector_store = vector_store or ChromaVectorStore(
            collection_name=settings.chroma_memory_collection,
            timeout_seconds=settings.memory_chroma_timeout_seconds,
        )

    def _memory_cipher(self) -> MemoryCipher:
        if self.cipher is None:
            self.cipher = MemoryCipher.from_base64_key(settings.memory_encryption_key)
        return self.cipher

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
        confidence = max(float(existing.confidence) if existing else 0.0, float(item.confidence))
        metadata = {
            "recordType": "user_memory",
            "userId": user_id,
            "memoryKey": item.memory_key,
            "memoryType": "semantic",
            "status": "ACTIVE",
            "confidence": confidence,
            "evidence": self._memory_cipher().encrypt_text(item.evidence) if item.evidence else "",
            "createdAt": existing.updated_at.isoformat() if existing and existing.updated_at else now.isoformat(),
            "updatedAt": now.isoformat(),
            "observedAt": now.isoformat(),
        }
        embedding = self.embedding_client.embed(merged_value)
        self.vector_store.upsert_chunks(
            ids=[self.memory_vector_id(user_id, item.memory_key)],
            embeddings=[embedding],
            texts=[self._memory_cipher().encrypt_text(merged_value)],
            metadatas=[metadata],
        )
        return SemanticMemoryRecord(
            memory_key=item.memory_key,
            memory_value=merged_value,
            confidence=confidence,
            evidence=item.evidence,
            updated_at=now,
        )

    def delete_summary(self, user_id: str, memory_key: str) -> None:
        self.vector_store.delete_chunks([self.memory_vector_id(user_id, memory_key)])

    def memory_vector_id(self, user_id: str, memory_key: str) -> str:
        return f"user-memory:{user_id}:{memory_key}"

    def _merge_memory(self, old_value: str, new_value: str) -> str:
        old_fragments = self._memory_fragments(old_value)
        new_fragments = self._memory_fragments(new_value)
        if not old_fragments:
            return self._render_memory_fragments(new_fragments)
        if not new_fragments:
            return self._render_memory_fragments(old_fragments)

        merged = old_fragments[:]
        for fragment in new_fragments:
            self._merge_fragment(merged, fragment)
        return self._render_memory_fragments(merged)

    def _merge_fragment(self, fragments: list[str], candidate: str) -> None:
        candidate_norm = self._normalize_for_compare(candidate)
        if not candidate_norm:
            return
        for index, existing in enumerate(fragments):
            existing_norm = self._normalize_for_compare(existing)
            if not existing_norm:
                continue
            if candidate_norm == existing_norm or candidate_norm in existing_norm:
                return
            if existing_norm in candidate_norm:
                fragments[index] = candidate
                return
            if self._memory_similarity(existing_norm, candidate_norm) >= settings.memory_semantic_merge_similarity_threshold:
                if len(candidate) > len(existing):
                    fragments[index] = candidate
                return
        fragments.append(candidate)

    def _memory_fragments(self, value: str) -> list[str]:
        fragments: list[str] = []
        for raw in MEMORY_FRAGMENT_SPLIT_RE.split(value or ""):
            fragment = self._clean_memory_fragment(raw)
            if fragment:
                fragments.append(fragment)
        return fragments

    def _clean_memory_fragment(self, value: str) -> str:
        cleaned = MEMORY_FRAGMENT_PREFIX_RE.sub("", value.strip())
        cleaned = " ".join(cleaned.split())
        return cleaned.strip(".;, ")

    def _render_memory_fragments(self, fragments: list[str]) -> str:
        unique: list[str] = []
        seen: set[str] = set()
        for fragment in fragments:
            cleaned = self._clean_memory_fragment(fragment)
            key = self._normalize_for_compare(cleaned)
            if cleaned and key not in seen:
                unique.append(cleaned)
                seen.add(key)

        max_items = max(1, settings.memory_semantic_merge_max_items)
        max_chars = max(1, settings.memory_semantic_merge_max_chars)
        retained = unique[-max_items:]
        while retained:
            rendered = "; ".join(retained)
            if len(rendered) <= max_chars or len(retained) == 1:
                return rendered[:max_chars].rstrip(".;, ")
            retained = retained[1:]
        return ""

    def _normalize_for_compare(self, value: str) -> str:
        return re.sub(r"\s+", "", value).lower()

    def _memory_similarity(self, left: str, right: str) -> float:
        return SequenceMatcher(None, left, right).ratio()

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
            score = cosine_score(distance)
            if score < settings.memory_semantic_min_score:
                continue
            records.append(self._record_from_payload(text, metadata, distance=distance))
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

    def _record_from_payload(
        self,
        text: Any,
        metadata: dict[str, Any],
        *,
        distance: float | None = None,
    ) -> SemanticMemoryRecord:
        updated_at = None
        updated_raw = metadata.get("updatedAt")
        if isinstance(updated_raw, str) and updated_raw:
            try:
                updated_at = datetime.fromisoformat(updated_raw)
            except ValueError:
                updated_at = None
        return SemanticMemoryRecord(
            memory_key=str(metadata.get("memoryKey") or ""),
            memory_value=self._memory_cipher().decrypt_text(str(text or "")),
            confidence=float(metadata.get("confidence") or 0.0),
            evidence=(
                self._memory_cipher().decrypt_text(str(metadata.get("evidence")))
                if metadata.get("evidence")
                else None
            ),
            updated_at=updated_at,
            score=cosine_score(distance) if distance is not None else None,
            distance=distance,
        )

