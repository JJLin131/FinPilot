from __future__ import annotations

import json

import pytest

from finpilot.evals.coverage import CoverageInventory, validate_coverage
from finpilot.evals.loader import EvalDatasetError, load_eval_cases
from finpilot.evals.registry import DEFAULT_SUITE_REGISTRY, EVALUATION_SUITES


def test_default_registry_contains_only_the_twelve_executable_suites():
    assert set(DEFAULT_SUITE_REGISTRY.names()) == set(EVALUATION_SUITES)
    assert len(EVALUATION_SUITES) == 12
    assert "routing" not in EVALUATION_SUITES
    assert "regression_release_gate" not in EVALUATION_SUITES


def test_loader_strictly_parses_jsonl_and_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "rag_retrieval.jsonl"
    payload = {
        "suite": "rag_retrieval",
        "case_id": "rag-001",
        "name": "工资规则",
        "query": "工资审批规则是什么？",
        "relevant_document_ids": ["doc-salary"],
    }
    path.write_text(
        "\n".join([json.dumps(payload, ensure_ascii=False), json.dumps(payload, ensure_ascii=False)]),
        encoding="utf-8",
    )

    with pytest.raises(EvalDatasetError, match="duplicate case_id rag-001"):
        load_eval_cases(path, DEFAULT_SUITE_REGISTRY)


def test_loader_rejects_suite_that_does_not_match_file_name(tmp_path):
    path = tmp_path / "rag_generation.jsonl"
    path.write_text(
        json.dumps(
            {
                "suite": "rag_retrieval",
                "case_id": "rag-001",
                "name": "wrong file",
                "query": "工资审批规则是什么？",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(EvalDatasetError, match="does not match dataset file rag_generation"):
        load_eval_cases(path, DEFAULT_SUITE_REGISTRY)


def test_coverage_requires_every_runtime_agent_tool_document_and_fault_component():
    cases = [
        DEFAULT_SUITE_REGISTRY.parse(
            {
                "suite": "planning_orchestration",
                "case_id": "plan-001",
                "name": "知识问答计划",
                "user_message": "工资规则",
                "expected_agents": ["QueryAgent"],
                "required_nodes": ["knowledge"],
            }
        ),
        DEFAULT_SUITE_REGISTRY.parse(
            {
                "suite": "tool_selection",
                "case_id": "tool-001",
                "name": "余额工具",
                "message": "查询余额",
                "expected": {
                    "agents": ["TreasuryDataAgent"],
                    "tool_calls": [{"tool_name": "query_account_balance", "arguments": {}}],
                },
            }
        ),
        DEFAULT_SUITE_REGISTRY.parse(
            {
                "suite": "rag_retrieval",
                "case_id": "rag-001",
                "name": "工资文档",
                "query": "工资规则",
                "relevant_document_ids": ["doc-salary"],
            }
        ),
        DEFAULT_SUITE_REGISTRY.parse(
            {
                "suite": "resilience_degradation",
                "case_id": "fault-001",
                "name": "规划器故障",
                "prompt": "查询余额",
                "fault": {"component": "planner", "behavior": "timeout"},
                "expected_status": "FAILED",
            }
        ),
    ]
    inventory = CoverageInventory(
        agents={"QueryAgent", "TreasuryDataAgent"},
        tools={"query_account_balance", "query_transactions"},
        document_ids={"doc-salary", "doc-receipt"},
        fault_components={"planner", "chroma"},
    )

    report = validate_coverage(cases, inventory)

    assert report.complete is False
    assert report.missing == {
        "tools": ["query_transactions"],
        "document_ids": ["doc-receipt"],
        "fault_components": ["chroma"],
    }
