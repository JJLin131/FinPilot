from __future__ import annotations

import logging
from typing import Any

import httpx

from FinanceAgent.config import settings
from FinanceAgent.models import RagMatch

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    def __init__(
        self,
        base_url: str | None = None,
        collection_name: str | None = None,
        tenant: str | None = None,
        database: str | None = None,
        timeout_seconds: int = 10,
    ):
        self.base_url = (base_url or settings.chroma_base_url).rstrip("/")
        self.collection_name = collection_name or settings.chroma_finance_collection
        self.tenant = tenant or settings.chroma_tenant
        self.database = database or settings.chroma_database
        self.timeout_seconds = timeout_seconds
        self._collection_id: str | None = None

    def upsert_chunks(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not settings.vector_enabled or not ids:
            return
        collection_id = self._ensure_collection()
        payload = {
            "ids": ids,
            "embeddings": embeddings,
            "documents": texts,
            "metadatas": metadatas,
        }
        self._post_collection_action(collection_id, "upsert", payload)

    def upsert_document_info(self, *, document_id: str, embedding: list[float], title: str) -> str:
        vector_id = self.document_info_vector_id(document_id)
        self.upsert_chunks(
            ids=[vector_id],
            embeddings=[embedding],
            texts=[title],
            metadatas=[{"documentId": document_id, "recordType": "doc_info", "status": "ACTIVE"}],
        )
        return vector_id

    def delete_chunks(self, ids: list[str]) -> None:
        if not settings.vector_enabled or not ids:
            return
        collection_id = self._ensure_collection()
        self._post_collection_action(collection_id, "delete", {"ids": ids})

    def delete_document_vectors(self, document_id: str, chunk_ids: list[str]) -> None:
        self.delete_chunks([self.document_info_vector_id(document_id), *chunk_ids])

    def query(self, query_embedding: list[float], limit: int, tenant_id: str) -> list[RagMatch]:
        if not settings.vector_enabled or not query_embedding:
            return []
        collection_id = self._ensure_collection()
        payload = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
            "where": self._where("chunk"),
        }
        raw = self._post_collection_action(collection_id, "query", payload)
        return self._to_matches(raw, tenant_id)

    def query_similar_document_info(self, title_embedding: list[float], *, limit: int, tenant_id: str = "__GLOBAL__") -> list[dict[str, Any]]:
        if not settings.vector_enabled or not title_embedding:
            return []
        collection_id = self._ensure_collection()
        payload = {
            "query_embeddings": [title_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances"],
            "where": self._where("doc_info"),
        }
        raw = self._post_collection_action(collection_id, "query", payload)
        return self._to_title_results(raw, tenant_id)

    def _to_title_results(self, raw: dict[str, Any], tenant_id: str) -> list[dict[str, Any]]:
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        by_document: dict[str, dict[str, Any]] = {}
        for index, title in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            if metadata.get("status") != "ACTIVE" or metadata.get("recordType") != "doc_info":
                continue
            document_id = str(metadata.get("documentId") or metadata.get("document_id") or "")
            if not document_id:
                continue
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            current = by_document.get(document_id)
            if current is None or score > current["score"]:
                by_document[document_id] = {
                    "document_id": document_id,
                    "title": str(title or metadata.get("title") or ""),
                    "score": round(score, 4),
                    "metadata": metadata,
                }
        return sorted(by_document.values(), key=lambda item: item["score"], reverse=True)

    def document_info_vector_id(self, document_id: str) -> str:
        return f"{document_id}:doc_info"

    def _ensure_collection(self) -> str:
        if self._collection_id:
            return self._collection_id
        with httpx.Client(timeout=self.timeout_seconds) as client:
            collection_id = self._ensure_collection_v2(client) or self._ensure_collection_v1(client)
        if not collection_id:
            raise RuntimeError("Unable to create or resolve Chroma collection.")
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
            logger.debug("Chroma v2 collection resolution failed: %s", exc)
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
            logger.debug("Chroma v1 collection resolution failed: %s", exc)
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
                    last_error = RuntimeError(f"Chroma {action} failed with {response.status_code}: {response.text[:300]}")
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"Chroma {action} failed.") from last_error

    def _to_matches(self, payload: dict[str, Any], tenant_id: str) -> list[RagMatch]:
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        matches: list[RagMatch] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and isinstance(metadatas[index], dict) else {}
            if metadata.get("status") != "ACTIVE" or metadata.get("recordType") != "chunk":
                continue
            distance = float(distances[index]) if index < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)
            if score < settings.vector_min_score:
                continue
            matches.append(
                RagMatch(
                    document_id=str(metadata.get("documentId") or metadata.get("document_id") or ""),
                    title="",
                    source="",
                    text=str(text or ""),
                    score=round(score, 4),
                )
            )
        return matches

    def _where(self, record_type: str) -> dict[str, Any]:
        return {"$and": [{"recordType": record_type}, {"status": "ACTIVE"}]}
