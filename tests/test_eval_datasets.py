from __future__ import annotations

import json
import uuid
from pathlib import Path

from finpilot.agent.agents.finance_qa_subagent import FinanceQaSubAgent
from finpilot.agent.agents.treasury_data_agent import TreasuryDataAgent
from finpilot.agent.agents.treasury_operation_agent import TreasuryOperationAgent
from finpilot.agent.tools import ToolRegistry
from finpilot.evals.coverage import CoverageInventory, validate_coverage
from finpilot.evals.loader import load_eval_cases
from finpilot.evals.registry import DEFAULT_SUITE_REGISTRY, EVALUATION_SUITES
from finpilot.evals.models import ToolSelectionCase


class EmptyRagService:
    def search(self, query: str, limit: int = 3):
        del query, limit
        return []


def test_dataset_directory_contains_only_new_evaluation_suites():
    root = Path("evals/datasets")
    actual = {path.stem for path in root.glob("*.jsonl")}

    assert actual == set(EVALUATION_SUITES)
    assert actual.isdisjoint({"routing", "tool_use", "grounded_answer", "safety", "query_rewrite"})


def test_every_new_dataset_is_strictly_loadable_and_has_smoke_case():
    root = Path("evals/datasets")
    for suite in EVALUATION_SUITES:
        cases = load_eval_cases(root / f"{suite}.jsonl", DEFAULT_SUITE_REGISTRY)
        assert cases
        assert any("smoke" in case.tags for case in cases)
        assert len({case.case_id for case in cases}) == len(cases)


def test_datasets_cover_runtime_agents_tools_documents_and_fault_components():
    root = Path("evals/datasets")
    cases = [
        case
        for suite in EVALUATION_SUITES
        for case in load_eval_cases(root / f"{suite}.jsonl", DEFAULT_SUITE_REGISTRY)
    ]
    registry = ToolRegistry(EmptyRagService())
    agents = [FinanceQaSubAgent(), TreasuryDataAgent(), TreasuryOperationAgent()]
    inventory = CoverageInventory(
        agents={"QueryAgent", "TreasuryDataAgent", "TreasuryOperationAgent"},
        tools={tool.name for agent in agents for tool in agent.build_context(registry).allowed_tools},
        document_ids={
            f"resource-finance-{uuid.uuid5(uuid.NAMESPACE_URL, path.name)}"
            for path in Path("src/main/resources").glob("*.md")
            if path.name[:2].isdigit()
        },
        fault_components={"planner", "llm", "chroma", "reranker", "mysql", "langfuse", "otel", "audit"},
    )

    report = validate_coverage(cases, inventory)

    assert report.complete, report.missing


def test_datasets_include_language_risk_and_execution_slices():
    root = Path("evals/datasets")
    cases = [
        case
        for suite in EVALUATION_SUITES
        for case in load_eval_cases(root / f"{suite}.jsonl", DEFAULT_SUITE_REGISTRY)
    ]
    tags = {tag for case in cases for tag in case.tags}

    assert {"中文标准表达", "口语表达", "歧义表达", "smoke", "release_only"} <= tags
    assert {case.execution_mode for case in cases} == {"live", "controlled"}
    assert "critical" in {case.severity for case in cases}


def test_release_gate_has_metric_thresholds_for_every_executable_suite():
    config = json.loads(Path("evals/release_gate.json").read_text(encoding="utf-8"))

    assert set(config["metric_thresholds"]) == set(EVALUATION_SUITES)
    assert {"ENV_UNAVAILABLE", "FIXTURE_UNAVAILABLE", "EVALUATOR_ERROR"} <= set(config["blocking_statuses"])


def test_tool_selection_expectations_match_real_subagent_allowlists_and_tool_schemas():
    registry = ToolRegistry(EmptyRagService())
    cases = load_eval_cases(Path("evals/datasets/tool_selection.jsonl"), DEFAULT_SUITE_REGISTRY)
    agents = {
        agent.name: {tool.name for tool in agent.build_context(registry).allowed_tools}
        for agent in [FinanceQaSubAgent(), TreasuryDataAgent(), TreasuryOperationAgent()]
    }

    for case in cases:
        assert isinstance(case, ToolSelectionCase)
        assert set(case.expected.agents) <= set(agents)
        for expected in case.expected.tool_calls:
            assert expected.agent_name in case.expected.agents
            assert expected.tool_name in agents[expected.agent_name]
            assert registry.arguments_are_valid(expected.tool_name, expected.arguments)
