from __future__ import annotations

import uuid
from pathlib import Path

from finpilot.agent.tools import ToolRegistry
from finpilot.evals.runner import EvalRunner
from finpilot.intents import INTENT_ORDER


EXPECTED_SUITES = {
    "routing",
    "tool_use",
    "rag_retrieval",
    "grounded_answer",
    "safety",
    "query_rewrite",
}

MIN_CASES_BY_SUITE = {
    "routing": 20,
    "tool_use": 20,
    "rag_retrieval": 18,
    "grounded_answer": 18,
    "safety": 18,
    "query_rewrite": 10,
}

KNOWN_AGENTS = {
    "QueryAgent",
    "TreasuryDataAgent",
    "TreasuryOperationAgent",
    "UNSUPPORTED",
}


class EmptyRagService:
    def search(self, query: str, limit: int = 3):
        return []


def test_minimum_eval_datasets_exist_and_load():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)

    assert {path.stem for path in root.glob("*.jsonl")} >= EXPECTED_SUITES

    for suite in EXPECTED_SUITES:
        cases = list(runner._load_suite(suite))
        assert len(cases) >= MIN_CASES_BY_SUITE[suite]
        assert all(case.suite == suite for case in cases)
        assert all(case.name for case in cases)
        assert all(case.chat_id for case in cases)
        assert all(case.content for case in cases)
        assert len({case.name for case in cases}) == len(cases)
        assert len({case.chat_id for case in cases}) == len(cases)


def test_eval_datasets_cover_core_assertion_types():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)
    cases = [case for suite in EXPECTED_SUITES for case in runner._load_suite(suite)]

    assert any(case.expected_intent for case in cases)
    assert any(case.expected_tool for case in cases)
    assert any(case.relevant_document_ids for case in cases)
    assert any(case.expected_answer_contains for case in cases)
    assert any(case.threat for case in cases)
    assert any(case.expected_safety_action for case in cases)
    assert any(case.expected_safety_code for case in cases)
    assert any(case.expected_agent for case in cases)
    assert any(case.expected_status for case in cases)
    assert any(case.expected_tool_status for case in cases)
    assert any(case.expected_tool_args for case in cases)
    assert any(case.expected_evidence_tool for case in cases)
    assert any(case.requires_approval for case in cases)
    assert any(case.privacy_forbidden_fields for case in cases)
    assert any(case.metric_tags for case in cases)


def test_safety_threat_cases_declare_explicit_expected_safety_outcome():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)

    for case in runner._load_suite("safety"):
        if case.threat:
            assert case.expected_safety_action
            assert case.expected_safety_code


def test_eval_dataset_expected_intents_tools_and_documents_match_runtime_contracts():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)
    cases = [case for suite in EXPECTED_SUITES for case in runner._load_suite(suite)]

    known_intents = set(INTENT_ORDER)
    registry = ToolRegistry(EmptyRagService())
    known_tools = {tool.name for tool in registry.list_tools()}
    tool_arg_keys = {
        name: set(spec.args_model.model_fields) if spec.args_model else set()
        for name, spec in registry._tools.items()
    }
    known_resource_document_ids = {
        f"resource-finance-{uuid.uuid5(uuid.NAMESPACE_URL, path.name)}"
        for path in Path("src/main/resources").glob("*.md")
        if path.name[:2].isdigit()
    }

    for case in cases:
        if case.expected_intent:
            assert case.expected_intent in known_intents
        if case.expected_agent:
            assert case.expected_agent in KNOWN_AGENTS
        if case.expected_tool:
            assert case.expected_tool in known_tools
        if case.expected_evidence_tool:
            assert case.expected_evidence_tool in known_tools
        if case.expected_tool_status:
            assert case.expected_tool_status in {"SUCCEEDED", "FAILED", "BLOCKED"}
        if case.expected_tool_args:
            assert case.expected_tool
            assert set(case.expected_tool_args) <= tool_arg_keys[case.expected_tool]
        assert set(case.relevant_document_ids) <= known_resource_document_ids


def test_eval_dataset_new_fields_have_expected_types():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)
    cases = [case for suite in EXPECTED_SUITES for case in runner._load_suite(suite)]

    for case in cases:
        assert isinstance(case.expected_tool_args, dict)
        assert isinstance(case.expected_reranked_document_ids, list)
        assert isinstance(case.requires_approval, bool)
        assert isinstance(case.privacy_forbidden_fields, list)
        assert all(isinstance(field, str) for field in case.privacy_forbidden_fields)
        assert isinstance(case.metric_tags, list)
        assert all(isinstance(tag, str) for tag in case.metric_tags)


def test_query_rewrite_dataset_exists_and_is_loadable():
    root = Path("evals/datasets")
    runner = EvalRunner(agent_service=object(), audit_store=object(), root=root)

    cases = list(runner._load_suite("query_rewrite"))

    assert len(cases) >= MIN_CASES_BY_SUITE["query_rewrite"]
    assert any("中文短问" in case.metric_tags for case in cases)
    assert any("english" in case.metric_tags for case in cases)
    assert any("歧义问法" in case.metric_tags for case in cases)
    assert any("资金池" in case.metric_tags for case in cases)
