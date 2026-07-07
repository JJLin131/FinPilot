from __future__ import annotations

from pathlib import Path

from finpilot.evals.runner import EvalRunner


EXPECTED_SUITES = {
    "routing",
    "tool_use",
    "rag_retrieval",
    "grounded_answer",
    "safety",
}


def test_minimum_eval_datasets_exist_and_load():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)

    assert {path.stem for path in root.glob("*.jsonl")} >= EXPECTED_SUITES

    for suite in EXPECTED_SUITES:
        cases = list(runner._load_suite(suite))
        assert len(cases) >= 2
        assert all(case.suite == suite for case in cases)
        assert all(case.name for case in cases)
        assert all(case.chat_id for case in cases)
        assert all(case.content for case in cases)


def test_eval_datasets_cover_core_assertion_types():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)
    cases = [case for suite in EXPECTED_SUITES for case in runner._load_suite(suite)]

    assert any(case.expected_intent for case in cases)
    assert any(case.expected_tool for case in cases)
    assert any(case.relevant_document_ids for case in cases)
    assert any(case.expected_answer_contains for case in cases)
    assert any(case.threat for case in cases)
