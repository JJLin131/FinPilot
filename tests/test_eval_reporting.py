from __future__ import annotations

from finpilot.evals.models import EvalCaseResult, EvalRunResult, EvalStatus, EvalSuiteResult
from finpilot.evals.reporting import write_capability_report


def _case(suite: str, case_id: str, metrics: dict[str, float], *, passed: bool = True) -> EvalCaseResult:
    return EvalCaseResult(
        suite=suite,
        case_id=case_id,
        status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
        passed=passed,
        metrics=metrics,
    )


def test_report_summarizes_capability_metrics_instead_of_case_log(tmp_path):
    rag_cases = [
        _case("rag_retrieval", "rag-1", {"recall_at_5": 1.0, "precision_at_5": 0.8, "mrr": 1.0}),
        _case("rag_retrieval", "rag-2", {"recall_at_5": 0.8, "precision_at_5": 0.6, "mrr": 0.5}),
    ]
    planning_cases = [
        _case("planning_orchestration", "plan-1", {"plan_executable": 1.0, "dependency_f1": 0.8}),
        _case("planning_orchestration", "plan-2", {"plan_executable": 0.0, "dependency_f1": 1.0}, passed=False),
    ]
    tool_cases = [
        _case("tool_calling", "tool-1", {"tool_sequence_accuracy": 1.0, "argument_exact_match": 0.0}),
        _case("tool_calling", "tool-2", {"tool_sequence_accuracy": 0.0, "argument_exact_match": 1.0}, passed=False),
    ]
    suites = [
        EvalSuiteResult(
            suite="rag_retrieval",
            mode="release",
            status=EvalStatus.PASSED,
            total_cases=2,
            passed_cases=2,
            results=rag_cases,
            metrics={"recall_at_5": 0.9, "precision_at_5": 0.7, "mrr": 0.75},
        ),
        EvalSuiteResult(
            suite="planning_orchestration",
            mode="release",
            status=EvalStatus.FAILED,
            total_cases=2,
            passed_cases=1,
            results=planning_cases,
            metrics={"plan_executable": 0.5, "dependency_f1": 0.9},
        ),
        EvalSuiteResult(
            suite="tool_calling",
            mode="release",
            status=EvalStatus.FAILED,
            total_cases=2,
            passed_cases=1,
            results=tool_cases,
            metrics={"tool_sequence_accuracy": 0.5, "argument_exact_match": 0.5},
        ),
    ]
    run = EvalRunResult(
        mode="release",
        status=EvalStatus.BLOCKED,
        suites=suites,
        release_gate={"status": "BLOCKED", "reasons": ["planning_orchestration.plan_executable below minimum"]},
    )
    config = {
        "metric_thresholds": {
            "rag_retrieval": {"recall_at_5": {"min": 0.9}, "mrr": {"min": 0.7}},
            "planning_orchestration": {"plan_executable": {"min": 1.0}},
            "tool_calling": {
                "tool_sequence_accuracy": {"min": 0.95},
                "argument_exact_match": {"min": 0.95},
            },
        }
    }

    report_path = write_capability_report(run, output_dir=tmp_path, gate_config=config)
    report = report_path.read_text(encoding="utf-8")

    assert "# FinPilot 能力测评报告" in report
    assert "RAG 检索" in report
    assert "Recall@5" in report
    assert "90.00%" in report
    assert "计划可行率" in report
    assert "50.00%" in report
    assert "工具误判率" in report
    assert "参数错误率" in report
    assert "≤ 5.00%" in report
    assert "发布门禁：**未通过**" in report
    assert "有效样本数" in report
    assert "rag-1" not in report
    assert "tool-2" not in report


def test_report_marks_metrics_unavailable_when_environment_is_missing(tmp_path):
    suite = EvalSuiteResult(
        suite="rag_retrieval",
        mode="smoke",
        status=EvalStatus.ENV_UNAVAILABLE,
        total_cases=1,
        passed_cases=0,
        environment={
            "available": False,
            "blocking_failures": [
                {"case_id": "rag-live", "status": "ENV_UNAVAILABLE", "code": "ENV_UNAVAILABLE", "message": "Missing: chroma"}
            ],
        },
    )

    report_path = write_capability_report(suite, output_dir=tmp_path)
    report = report_path.read_text(encoding="utf-8")

    assert "环境状态：**不可用**" in report
    assert "Missing: chroma" in report
    assert "MRR" in report
    assert "不可用" in report
