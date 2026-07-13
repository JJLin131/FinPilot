from __future__ import annotations

import json

from finpilot.evals.models import EvalStatus, EvalSuiteResult
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


def test_publish_eval_suite_scores_creates_trace_and_numeric_scores(monkeypatch):
    captured: dict[str, object] = {"scores": []}

    class SpanContext:
        def __enter__(self):
            return object()

        def __exit__(self, exc_type, exc, traceback):
            return False

    class Client:
        def create_trace_id(self, *, seed):
            captured["seed"] = seed
            return "trace-eval-1"

        def start_as_current_span(self, **kwargs):
            captured["span"] = kwargs
            return SpanContext()

        def create_score(self, **kwargs):
            captured["scores"].append(kwargs)

    monkeypatch.setattr(langfuse_support, "get_langfuse_client", lambda: Client())
    suite = EvalSuiteResult(
        suite="tool_calling",
        mode="release",
        status=EvalStatus.FAILED,
        total_cases=4,
        passed_cases=3,
        metrics={"tool_sequence_accuracy": 0.75, "parameter_error_rate": 0.25},
    )

    langfuse_support.publish_eval_suite_scores(suite, run_id="run-1")

    assert captured["seed"] == "finpilot-eval:run-1"
    assert captured["span"]["name"] == "eval.tool_calling"
    scores = {item["name"]: item for item in captured["scores"]}
    assert scores["eval.tool_calling.pass_rate"]["value"] == 0.75
    assert scores["eval.tool_calling.tool_sequence_accuracy"]["value"] == 0.75
    assert scores["eval.tool_calling.parameter_error_rate"]["value"] == 0.25
    assert all(item["trace_id"] == "trace-eval-1" for item in scores.values())
