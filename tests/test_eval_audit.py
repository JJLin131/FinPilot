from __future__ import annotations

import json

from finpilot.evals.models import EvalStatus, EvalSuiteResult
from finpilot.observability.audit import FileAuditStore


def test_file_audit_store_purges_legacy_eval_results_and_writes_schema_v2(tmp_path):
    audit_file = tmp_path / "agent_eval_run.jsonl"
    audit_file.write_text(
        "\n".join(
            [
                json.dumps({"suite": "routing", "score": 1.0}),
                json.dumps({"schema_version": 2, "suite": "rag_retrieval", "status": "PASSED"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    store = FileAuditStore(tmp_path)
    store.record_eval_run(
        EvalSuiteResult(
            suite="tool_calling",
            mode="smoke",
            status=EvalStatus.PASSED,
            total_cases=1,
            passed_cases=1,
        )
    )

    records = [json.loads(line) for line in audit_file.read_text(encoding="utf-8").splitlines()]
    assert [record["suite"] for record in records] == ["rag_retrieval", "tool_calling"]
    assert {record["schema_version"] for record in records} == {2}
