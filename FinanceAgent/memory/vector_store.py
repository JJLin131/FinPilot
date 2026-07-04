from __future__ import annotations

import logging
from typing import Any

import httpx

from FinanceAgent.config import settings
from FinanceAgent.memory.models import SemanticMemoryMatch

logger = logging.getLogger(__name__)


class MemoryVectorStore:
    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        timeout_seconds: int = 10,
    ):
        self.base_url = (base_url or settings.chroma_base_url).rstrip("/")
        self.collection_name = collection_name or settings.memory_chroma_collection
        self.tenant = tenant or settings.chroma_tenant
        self.database = database or settings.chroma_database
        self.timeout_seconds = timeout_seconds
        self._collection_id: str | None = None

    def upsert_memory(
        self,
        *,
        memory_id: str,
        embedding: list[float],
        text: str,
        metadata: dict[str, Any],
    ) -> None:
        if not settings.vector_enabled or not embedding or not text.strip():
            return
        collection_id = self._ensure_collection()
        payload = {
            "ids": [memory_id],
            "embeddings": [embedding],
            "documents": [text],
            "metadatas": [{**metadata, "recordType": "memory", "status": metadata.get("status", "ACTIVE")}],
        }
        self._post_collection_action(collection_id, "upsert", payload)

    def delete_memory(self, memory_id: str) -> None:
        if not settings.vector_enabled:
            return
        collection_id = self._ensure_collection()
        self._post_collection_action(collection_id, "delete", {"ids": [memory_id]})

    def query(self, *, query_embedding: list[float], user_id: str, limit: int) -> list[SemanticMemoryMatch]:
        if not settings.vector_enabled or not query_embedding:
            return []
        collection_id = self._ensure_collection()
        payload = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
            "where": {
                "$and": [
                    {"recordType": "memory"},
                    {"status": "ACTIVE"},
                    {"user_id": user_id},
                ]
            },
        }
        raw = self._post_collection_action(collection_id, "query", payload)
        return self._to_matches(raw)

    def _ensure_collection(self) -> str:
        if self._collection_id:
            return self._collection_id
        with httpx.Client(timeout=self.timeout_seconds) as client:
            collection_id = self._ensure_collection_v2(client) or self._ensure_collection_v1(client)
        if not collection_id:
            raise RuntimeError("Unable to create or resolve Chroma memory collection.")
        self._collection_id = collection_id
        return collection_id

    def _ensure_collection_v2(self, client: httpx.Client) -> str | None:
        base = f"{self.base_url}/api/v2/tenants/{self.tenant}/databases/{self.database}/collections"
        try:
            response = client.get(base)
            if response.status_code == 200:
                for item in response.json():
                    if item.get("name") == self.collection_name:
                        return item.get("id") or item.get("name")
            response = client.post(base, json={"name": self.collection_name})
            if response.status_code in {200, 201}:
                payload = response.json()
                return payload.get("id") or payload.get("name") or self.collection_name
        except Exception as exc:
            logger.debug("Chroma v2 memory collection resolution failed: %s", exc)
        return None

    def _ensure_collection_v1(self, client: httpx.Client) -> str | None:
        base = f"{self.base_url}/api/v1/collections"
        try:
            response = client.get(base)
            if response.status_code == 200:
                for item in response.json():
                    if item.get("name") == self.collection_name:
                        return item.get("id") or item.get("name")
            response = client.post(base, json={"name": self.collection_name})
            if response.status_code in {200, 201}:
                payload = response.json()
                return payload.get("id") or payload.get("name") or self.collection_name
        except Exception as exc:
            logger.debug("Chroma v1 memory collection resolution failed: %s", exc)
        return None

    def _post_collection_action(self, collection_id: str, action: str, payload: dict[str, Any]) -> dict[str, Any]:
        paths = [
            f"{self.base_url}/api/v2/tenants/{self.tenant}/databases/{self.database}/collections/{collection_id}/{action}",
            f"{self.base_url}/api/v1/collections/{collection_id}/{action}",
        ]
        last_error: Exception | None = None
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for path in paths:
                try:
                    response = client.post(path, json=payload)
                    if response.status_code < 400:
                        return response.json() if response.content else {}
                    last_error = RuntimeError(f"Chroma memory {action} failed with {response.status_code}: {response.text[:300]}")
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"Chroma memory {action} failed.") from last_error

    def _to_matches(self, payload: dict[str, Any]) -> list[SemanticMemoryMatch]:
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        matches: list[SemanticMemoryMatch] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            if metadata.get("status") != "ACTIVE" or metadata.get("recordType") != "memory":
                continue
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            memory_id = str(ids[index]) if index < len(ids) else str(metadata.get("mysql_memory_id") or "")
            matches.append(
                SemanticMemoryMatch(
                    id=memory_id,
                    content=str(text or ""),
                    score=round(score, 4),
                    metadata=metadata,
                )
            )
        return matches
