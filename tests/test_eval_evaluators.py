from __future__ import annotations

from finpilot.evals.evaluators import DEFAULT_EVALUATORS, build_default_evaluators
from finpilot.evals.evaluators.planning_orchestration import PlanningOrchestrationEvaluator
from finpilot.evals.evaluators.rag_generation import RagGenerationEvaluator
from finpilot.evals.evaluators.rag_retrieval import RagRetrievalEvaluator
from finpilot.evals.evaluators.tool_execution import ToolExecutionEvaluator
from finpilot.evals.evaluators.tool_selection import ToolSelectionEvaluator
from finpilot.evals.models import (
    EvalObservation,
    PlanningOrchestrationCase,
    RagGenerationCase,
    RagRetrievalCase,
    ToolExecutionCase,
    ToolSelectionCase,
)
from finpilot.evals.registry import EVALUATION_SUITES


def test_rag_retrieval_computes_rank_aware_metrics():
    case = RagRetrievalCase(
        suite="rag_retrieval",
        case_id="rag-001",
        name="multi relevant",
        query="规则",
        relevant_document_ids=["doc-a", "doc-c"],
        k_values=[1, 3],
    )
    observation = EvalObservation(
        status="SUCCEEDED",
        retrieved_documents=[
            {"document_id": "doc-a"},
            {"document_id": "doc-b"},
            {"document_id": "doc-c"},
        ],
    )

    result = RagRetrievalEvaluator().evaluate(case, observation)

    assert result.passed is True
    assert result.metrics["recall_at_1"] == 0.5
    assert result.metrics["precision_at_3"] == 0.6667
    assert result.metrics["mrr"] == 1.0
    assert result.metrics["ndcg_at_3"] > 0.9


def test_rag_generation_uses_judge_scores_instead_of_local_heuristics():
    class Judge:
        def score(self, case, observation):
            return {
                "faithfulness": 0.92,
                "answer_correctness": 0.88,
                "answer_relevance": 0.95,
            }

    case = RagGenerationCase(
        suite="rag_generation",
        case_id="generation-001",
        name="answer",
        question="工资规则",
        reference_answer="需要审批",
        contexts=[{"document_id": "doc-a", "text": "需要审批"}],
        required_claims=["审批"],
        expected_citations=["doc-a"],
    )
    observation = EvalObservation(
        status="SUCCEEDED",
        response={"answer": "需要审批", "evidence": [{"summary": {"document_id": "doc-a"}}]},
    )

    result = RagGenerationEvaluator(Judge(), minimum_score=0.8).evaluate(case, observation)

    assert result.passed is True
    assert result.metrics["faithfulness"] == 0.92
    assert result.metrics["citation_recall"] == 1.0


def test_planning_evaluator_checks_agents_nodes_dependencies_and_parallel_groups():
    case = PlanningOrchestrationCase(
        suite="planning_orchestration",
        case_id="plan-001",
        name="query then pay",
        user_message="查询后付款",
        expected_agents=["TreasuryDataAgent", "TreasuryOperationAgent"],
        required_nodes=["balance", "payment"],
        required_dependencies=[["payment", "balance"]],
        expected_parallel_groups=[],
    )
    observation = EvalObservation(
        status="SUCCEEDED",
        plan={
            "status": "READY",
            "nodes": [
                {
                    "node_id": "balance",
                    "agent_name": "TreasuryDataAgent",
                    "task": "查询账户余额",
                    "depends_on": [],
                },
                {
                    "node_id": "payment",
                    "agent_name": "TreasuryOperationAgent",
                    "task": "余额充足时发起付款",
                    "depends_on": ["balance"],
                },
            ],
        },
    )

    result = PlanningOrchestrationEvaluator().evaluate(case, observation)

    assert result.passed is True
    assert result.metrics["agent_set_accuracy"] == 1.0
    assert result.metrics["dependency_f1"] == 1.0
    assert result.metrics["plan_executable"] == 1.0


def test_tool_selection_evaluator_scores_agent_tool_and_semantic_arguments_separately():
    case = ToolSelectionCase(
        suite="tool_selection",
        case_id="tool-001",
        name="balance",
        message="查询余额",
        expected={
            "agents": ["TreasuryDataAgent"],
            "tool_calls": [
                {
                    "tool_name": "query_account_balance",
                    "arguments": {"accountId": "ACC-001", "optional": None},
                }
            ],
        },
    )
    observation = EvalObservation(
        status="READY",
        plan={"nodes": [{"agent_name": "TreasuryDataAgent"}]},
        tool_calls=[
            {
                "agent_name": "TreasuryDataAgent",
                "tool_name": "query_account_balance",
                "parameters": {"accountId": "ACC-001"},
                "schema_valid": True,
            }
        ],
    )

    result = ToolSelectionEvaluator().evaluate(case, observation)

    assert result.passed is True
    assert result.metrics["agent_selection_accuracy"] == 1.0
    assert result.metrics["tool_selection_accuracy"] == 1.0
    assert result.metrics["tool_misjudgment_rate"] == 0.0
    assert result.metrics["argument_accuracy"] == 1.0
    assert result.metrics["parameter_error_rate"] == 0.0
    assert result.metrics["argument_schema_validity"] == 1.0


def test_tool_selection_keeps_parameter_accuracy_when_expected_call_is_correct_but_extra_tool_exists():
    case = ToolSelectionCase(
        suite="tool_selection",
        case_id="tool-extra-001",
        name="payment with over-planning",
        message="创建付款单",
        expected={
            "agents": ["TreasuryOperationAgent"],
            "tool_calls": [
                {
                    "agent_name": "TreasuryOperationAgent",
                    "tool_name": "create_payment_order",
                    "arguments": {
                        "accountId": "ACC-001",
                        "payeeAccountId": "ACC-002",
                        "amount": 100,
                        "currency": "CNY",
                        "purpose": "工资",
                    },
                }
            ],
        },
    )
    observation = EvalObservation(
        status="READY",
        plan={
            "nodes": [
                {"agent_name": "TreasuryDataAgent"},
                {"agent_name": "TreasuryOperationAgent"},
            ]
        },
        tool_calls=[
            {
                "agent_name": "TreasuryDataAgent",
                "tool_name": "query_treasury_account",
                "parameters": {"accountId": "ACC-001"},
                "schema_valid": True,
            },
            {
                "agent_name": "TreasuryOperationAgent",
                "tool_name": "create_payment_order",
                "parameters": {
                    "accountId": "ACC-001",
                    "payeeAccountId": "ACC-002",
                    "amount": 100,
                    "currency": "CNY",
                    "purpose": "工资",
                },
                "schema_valid": True,
            },
        ],
    )

    result = ToolSelectionEvaluator().evaluate(case, observation)

    assert result.metrics["tool_selection_accuracy"] == 0.0
    assert result.metrics["extra_tool_count"] == 1.0
    assert result.metrics["argument_accuracy"] == 1.0
    assert result.metrics["parameter_error_rate"] == 0.0


def test_tool_execution_evaluator_scores_runtime_status_and_actual_invocation():
    case = ToolExecutionCase(
        suite="tool_execution",
        case_id="tool-exec-001",
        name="balance execution",
        user_message="查询余额",
        expected_calls=[{"tool_name": "query_account_balance", "arguments": {"accountId": "ACC-001"}}],
        expected_status="SUCCEEDED",
    )
    observation = EvalObservation(
        status="SUCCEEDED",
        tool_calls=[
            {"tool_name": "query_account_balance", "parameters": {"accountId": "ACC-001"}, "status": "SUCCEEDED"}
        ],
    )

    result = ToolExecutionEvaluator().evaluate(case, observation)

    assert result.passed is True
    assert result.metrics["execution_status_match"] == 1.0
    assert result.metrics["tool_execution_accuracy"] == 1.0


def test_default_evaluator_registry_covers_every_executable_suite():
    assert set(DEFAULT_EVALUATORS) == set(EVALUATION_SUITES)
    assert set(build_default_evaluators()) == set(EVALUATION_SUITES)
