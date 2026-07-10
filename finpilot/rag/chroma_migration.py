from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from finpilot.rag.embeddings import OllamaEmbeddingClient


@dataclass(frozen=True)
class ChromaMigrationResult:
    read_count: int
    write_count: int


class ChromaCollectionMigrator:
    def __init__(self, embedding_client: OllamaEmbeddingClient | None = None) -> None:
        self.embedding_client = embedding_client or OllamaEmbeddingClient()

    def migrate(
        self,
        source_store,
        target_store,
        *,
        batch_size: int = 100,
        transform: Callable[[str, dict], tuple[str, str, dict]] | None = None,
    ) -> ChromaMigrationResult:
        read_count = 0
        write_count = 0
        for batch in source_store.iter_records(batch_size):
            ids: list[str] = []
            embedding_texts: list[str] = []
            target_texts: list[str] = []
            metadatas: list[dict] = []
            for record in batch:
                document = str(record["document"])
                metadata = dict(record["metadata"])
                embedding_text, target_text, target_metadata = (
                    transform(document, metadata) if transform else (document, document, metadata)
                )
                ids.append(str(record["id"]))
                embedding_texts.append(embedding_text)
                target_texts.append(target_text)
                metadatas.append(target_metadata)
            embeddings = self.embedding_client.embed_many(embedding_texts)
            if len(embeddings) != len(ids):
                raise RuntimeError("Embedding count did not match Chroma migration record count.")
            written = target_store.upsert_chunks(
                ids=ids,
                embeddings=embeddings,
                texts=target_texts,
                metadatas=metadatas,
            )
            if written != len(ids):
                raise RuntimeError(f"Chroma migration read {len(ids)} records but target wrote {written}.")
            read_count += len(ids)
            write_count += written
        return ChromaMigrationResult(read_count=read_count, write_count=write_count)


def memory_encryption_transform(cipher):
    def transform(document: str, metadata: dict) -> tuple[str, str, dict]:
        plaintext = cipher.decrypt_legacy_text(document)
        target_metadata = dict(metadata)
        evidence = target_metadata.get("evidence")
        if evidence:
            target_metadata["evidence"] = cipher.encrypt_text(cipher.decrypt_legacy_text(str(evidence)))
        return plaintext, cipher.encrypt_text(plaintext), target_metadata

    return transform
