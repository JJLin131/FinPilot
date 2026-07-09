from __future__ import annotations

from pathlib import Path


def test_compose_keeps_local_only_langfuse_defaults_explicit():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert "${LANGFUSE_SECRET_KEY:-sk-lf-local}" in compose
    assert "${LANGFUSE_POSTGRES_PASSWORD:-postgres}" in compose
    assert "${LANGFUSE_REDIS_AUTH:-myredissecret}" in compose
    assert "${MINIO_ROOT_PASSWORD:-miniosecret}" in compose


def test_compose_defaults_to_dependency_stack_without_finpilot_service():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    assert 'finpilot-service:\n    profiles: ["app"]' in compose
    assert 'chroma:\n    image: chromadb/chroma:latest\n    ports:' in compose
    assert 'ollama:\n    image: ollama/ollama:latest\n    dns:' in compose
    assert 'reranker:\n    image: ghcr.io/huggingface/text-embeddings-inference:cpu-latest\n    command:' in compose
    assert "ollama-rewrite-model-pull:" in compose
