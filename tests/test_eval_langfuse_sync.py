from __future__ import annotations

import json

from finpilot.observability import langfuse_support


def test_langfuse_sync_uses_new_suite_contract_without_legacy_chat_fields(tmp_path, monkeypatch):
    dataset = tmp_path / "rag_retrieval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "suite": "rag_retrieval",
                "case_id": "rag-1",
                "name": "工资召回",
                "execution_mode": "live",
                "query": "工资规则",
                "relevant_document_ids": ["doc-1"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}
    client = object()
    monkeypatch.setattr(langfuse_support, "get_langfuse_client", lambda: client)
    monkeypatch.setattr(langfuse_support, "ensure_dataset", lambda *args, **kwargs: None)

    def upsert(dataset_name, **kwargs):
        captured.update(dataset_name=dataset_name, **kwargs)
        return "item-1"

    monkeypatch.setattr(langfuse_support, "upsert_dataset_item", upsert)

    langfuse_support.sync_local_datasets(tmp_path)

    assert captured["payload_input"] == {"case_id": "rag-1", "query": "工资规则"}
    assert captured["expected_output"] == {"relevant_document_ids": ["doc-1"]}
    assert captured["metadata"]["schema_version"] == 2
