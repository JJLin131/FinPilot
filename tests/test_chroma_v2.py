from __future__ import annotations

import pytest

from finpilot.rag.chroma_migration import ChromaCollectionMigrator, memory_encryption_transform
from finpilot.rag.vector_store import ChromaVectorStore, cosine_score


class FakeResponse:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"
        self.text = str(payload)

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, get_payload, post_payload=None) -> None:
        self.get_payload = get_payload
        self.post_payload = post_payload
        self.posts: list[dict] = []

    def get(self, url):
        return FakeResponse(200, self.get_payload)

    def post(self, url, json):
        self.posts.append(json)
        return FakeResponse(201, self.post_payload)


def test_chroma_v2_collection_creation_explicitly_requests_cosine():
    store = ChromaVectorStore(collection_name="finance-knowledge-bge-m3-v2")
    client = FakeClient(
        [],
        {
            "id": "collection-1",
            "name": "finance-knowledge-bge-m3-v2",
            "configuration_json": {"hnsw": {"space": "cosine"}},
        },
    )

    collection_id = store._ensure_collection_v2(client)

    assert collection_id == "collection-1"
    assert client.posts == [
        {
            "name": "finance-knowledge-bge-m3-v2",
            "configuration": {"hnsw": {"space": "cosine"}},
        }
    ]


def test_existing_non_cosine_collection_is_rejected():
    store = ChromaVectorStore(collection_name="finance-knowledge-bge-m3-v2")
    client = FakeClient(
        [
            {
                "id": "collection-1",
                "name": "finance-knowledge-bge-m3-v2",
                "configuration_json": {"hnsw": {"space": "l2"}},
            }
        ]
    )

    with pytest.raises(RuntimeError, match="cosine"):
        store._ensure_collection_v2(client)


@pytest.mark.parametrize(
    ("distance", "expected"),
    [(-0.2, 1.0), (0.0, 1.0), (0.35, 0.65), (1.4, 0.0)],
)
def test_cosine_score_is_clamped(distance: float, expected: float):
    assert cosine_score(distance) == expected


class SourceStore:
    def iter_records(self, batch_size: int):
        assert batch_size == 2
        yield [
            {"id": "id-1", "document": "first", "metadata": {"kind": "one"}},
            {"id": "id-2", "document": "second", "metadata": {"kind": "two"}},
        ]


class TargetStore:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def upsert_chunks(self, **kwargs) -> int:
        self.calls.append(kwargs)
        return len(kwargs["ids"])


class Embeddings:
    def embed_many(self, texts):
        assert texts == ["first", "second"]
        return [[1.0], [2.0]]


def test_collection_migration_preserves_ids_documents_and_metadata_while_reembedding():
    target = TargetStore()
    migrator = ChromaCollectionMigrator(embedding_client=Embeddings())

    result = migrator.migrate(SourceStore(), target, batch_size=2)

    assert result.read_count == 2
    assert result.write_count == 2
    assert target.calls == [
        {
            "ids": ["id-1", "id-2"],
            "embeddings": [[1.0], [2.0]],
            "texts": ["first", "second"],
            "metadatas": [{"kind": "one"}, {"kind": "two"}],
        }
    ]


class PrefixCipher:
    def decrypt_text(self, value: str) -> str:
        return value.removeprefix("old:").removeprefix("new:")

    def decrypt_legacy_text(self, value: str) -> str:
        return self.decrypt_text(value)

    def encrypt_text(self, value: str) -> str:
        return f"new:{value}"


def test_memory_migration_reencrypts_document_and_evidence_but_embeds_plaintext():
    transform = memory_encryption_transform(PrefixCipher())

    embedding_text, target_text, metadata = transform(
        "old:用户常用招商银行。",
        {"memoryKey": "userBank", "evidence": "old:用户明确说明。"},
    )

    assert embedding_text == "用户常用招商银行。"
    assert target_text == "new:用户常用招商银行。"
    assert metadata == {"memoryKey": "userBank", "evidence": "new:用户明确说明。"}


def test_collection_migration_fails_when_target_reports_no_writes():
    class DisabledTarget(TargetStore):
        def upsert_chunks(self, **kwargs) -> int:
            self.calls.append(kwargs)
            return 0

    with pytest.raises(RuntimeError, match="wrote 0"):
        ChromaCollectionMigrator(embedding_client=Embeddings()).migrate(SourceStore(), DisabledTarget(), batch_size=2)
