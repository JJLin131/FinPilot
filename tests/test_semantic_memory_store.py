from __future__ import annotations

from finpilot.memory.models import SemanticMemoryItem
from finpilot.memory.semantic_store import ChromaSemanticMemoryStore


class FakeCipher:
    def encrypt_text(self, value: str) -> str:
        return f"enc:{value}"

    def decrypt_text(self, value: str) -> str:
        return value.removeprefix("enc:")


class RecordingEmbeddingClient:
    def __init__(self) -> None:
        self.values: list[str] = []

    def embed(self, value: str) -> list[float]:
        self.values.append(value)
        return [0.1, 0.2]


class RecordingVectorStore:
    def __init__(self, query_payload=None) -> None:
        self.upserts: list[dict] = []
        self.query_payload = query_payload
        self.deleted: list[list[str]] = []

    def _ensure_collection(self):
        return "collection-1"

    def _post_collection_action(self, collection_id, action, payload):
        if action == "query":
            return self.query_payload
        assert action == "get"
        return {"documents": [], "metadatas": []}

    def upsert_chunks(self, **kwargs) -> None:
        self.upserts.append(kwargs)

    def delete_chunks(self, ids):
        self.deleted.append(ids)


def test_semantic_memory_embeds_plaintext_but_stores_encrypted_document_and_evidence():
    embedding = RecordingEmbeddingClient()
    vector_store = RecordingVectorStore()
    store = ChromaSemanticMemoryStore(
        embedding_client=embedding,
        vector_store=vector_store,
        cipher=FakeCipher(),
    )

    store.upsert_summary(
        "user-1",
        SemanticMemoryItem(
            memoryKey="userBank",
            memoryValue="用户工资卡是招商银行卡 6222020202020202020。",
            confidence=0.9,
            evidence="用户明确提供了完整卡号。",
        ),
    )

    assert embedding.values == ["用户工资卡是招商银行卡 6222020202020202020。"]
    assert vector_store.upserts[0]["texts"] == ["enc:用户工资卡是招商银行卡 6222020202020202020。"]
    metadata = vector_store.upserts[0]["metadatas"][0]
    assert metadata["evidence"] == "enc:用户明确提供了完整卡号。"


def test_semantic_search_decrypts_memory_and_keeps_cosine_distance_for_debug():
    vector_store = RecordingVectorStore(
        {
            "documents": [["enc:用户常用招商银行。"]],
            "metadatas": [[{"recordType": "user_memory", "status": "ACTIVE", "memoryKey": "userBank"}]],
            "distances": [[0.25]],
        }
    )
    store = ChromaSemanticMemoryStore(
        embedding_client=RecordingEmbeddingClient(),
        vector_store=vector_store,
        cipher=FakeCipher(),
    )

    records = store.search("user-1", "常用银行", 5)

    assert records[0].memory_value == "用户常用招商银行。"
    assert records[0].score == 0.75
    assert records[0].distance == 0.25


def test_semantic_delete_targets_only_user_and_memory_key_vector():
    vector_store = RecordingVectorStore()
    store = ChromaSemanticMemoryStore(
        embedding_client=RecordingEmbeddingClient(),
        vector_store=vector_store,
        cipher=FakeCipher(),
    )

    store.delete_summary("user-1", "userBank")

    assert vector_store.deleted == [["user-memory:user-1:userBank"]]
