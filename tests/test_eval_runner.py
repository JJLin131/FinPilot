from __future__ import annotations

import json

from finpilot.evals.backends import ControlledBackend, SequenceFixture
from finpilot.evals.gates.release import ReleaseGate
from finpilot.evals.models import EvalObservation, EvalStatus, EvalSuiteResult
from finpilot.evals.runner import EvalRunner


class RecordingAuditStore:
    def __init__(self):
        self.results = []

    def record_eval_run(self, result):
        self.results.append(result)


def _write_case(root, suite: str, payload: dict):
    path = root / f"{suite}.jsonl"
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def test_runner_executes_controlled_suite_and_aggregates_metrics(tmp_path):
    _write_case(
        tmp_path,
        "tool_execution",
        {
            "suite": "tool_execution",
            "case_id": "tool-001",
            "name": "余额查询",
            "tags": ["smoke"],
            "execution_mode": "controlled",
            "fixtures": {"fixture": "balance"},
            "user_message": "查询 ACC-001 余额",
            "expected_calls": [
                {"tool_name": "query_account_balance", "arguments": {"accountId": "ACC-001"}}
            ],
        },
    )
    fixture = SequenceFixture(
        [
            EvalObservation(
                status="SUCCEEDED",
                tool_calls=[
                    {
                        "tool_name": "query_account_balance",
                        "parameters": {"accountId": "ACC-001"},
                        "status": "SUCCEEDED",
                    }
                ],
            )
        ]
    )
    audit = RecordingAuditStore()
    runner = EvalRunner(
        agent_service=object(),
        audit_store=audit,
        root=tmp_path,
        controlled_backend=ControlledBackend({"balance": fixture}),
        environment_checker=lambda requirements: [],
    )

    suite = runner.run_suite("tool_execution", mode="smoke")

    assert suite.status is EvalStatus.PASSED
    assert suite.total_cases == 1
    assert suite.metrics["tool_execution_accuracy"] == 1.0
    assert audit.results == [suite]


def test_runner_does_not_turn_missing_fixture_into_zero_or_pass(tmp_path):
    _write_case(
        tmp_path,
        "resilience_degradation",
        {
            "suite": "resilience_degradation",
            "case_id": "fault-001",
            "name": "规划超时",
            "execution_mode": "controlled",
            "fault": {"component": "planner", "behavior": "timeout"},
            "prompt": "查询余额",
            "expected_status": "FAILED",
        },
    )
    runner = EvalRunner(
        agent_service=object(),
        audit_store=RecordingAuditStore(),
        root=tmp_path,
        controlled_backend=ControlledBackend({}),
        environment_checker=lambda requirements: [],
    )

    suite = runner.run_suite("resilience_degradation", mode="release")

    assert suite.status is EvalStatus.FIXTURE_UNAVAILABLE
    assert suite.results[0].failure.code == "FIXTURE_UNAVAILABLE"
    assert suite.passed_cases == 0


def test_runner_marks_missing_required_environment_as_blocking_status(tmp_path):
    _write_case(
        tmp_path,
        "rag_retrieval",
        {
            "suite": "rag_retrieval",
            "case_id": "rag-001",
            "name": "工资规则",
            "execution_mode": "live",
            "query": "工资审批规则",
            "relevant_document_ids": ["doc-salary"],
        },
    )
    runner = EvalRunner(
        agent_service=object(),
        audit_store=RecordingAuditStore(),
        root=tmp_path,
        environment_checker=lambda requirements: ["chroma"],
    )

    suite = runner.run_suite("rag_retrieval", mode="release")

    assert suite.status is EvalStatus.ENV_UNAVAILABLE
    assert suite.results[0].failure.code == "ENV_UNAVAILABLE"


def test_runner_marks_live_case_unavailable_when_service_cannot_initialize(tmp_path):
    _write_case(
        tmp_path,
        "rag_retrieval",
        {
            "suite": "rag_retrieval",
            "case_id": "rag-runtime",
            "name": "运行时未启动",
            "execution_mode": "live",
            "query": "工资规则",
            "relevant_document_ids": ["doc-1"],
        },
    )
    runner = EvalRunner(
        agent_service=None,
        audit_store=RecordingAuditStore(),
        root=tmp_path,
        environment_checker=lambda requirements: [],
    )

    suite = runner.run_suite("rag_retrieval", mode="release")

    assert suite.status is EvalStatus.ENV_UNAVAILABLE
    assert suite.results[0].failure.code == "RUNTIME_SERVICE_UNAVAILABLE"


def test_controlled_ragas_case_still_checks_judge_package_dependency(tmp_path):
    _write_case(
        tmp_path,
        "rag_generation",
        {
            "suite": "rag_generation",
            "case_id": "generation-001",
            "name": "可控回答真实评分",
            "execution_mode": "controlled",
            "fixtures": {"observation": {"status": "COMPLETED", "response": {"answer": "需要审批"}}},
            "question": "工资规则？",
            "reference_answer": "需要审批",
            "contexts": [{"document_id": "doc-1", "text": "工资需要审批"}],
        },
    )
    runner = EvalRunner(
        agent_service=object(),
        audit_store=RecordingAuditStore(),
        root=tmp_path,
        environment_checker=lambda requirements: [item for item in requirements if item == "ragas"],
    )

    suite = runner.run_suite("rag_generation", mode="release")

    assert suite.status is EvalStatus.ENV_UNAVAILABLE
    assert suite.results[0].failure.message == "Missing: ragas"


def test_release_gate_blocks_critical_failure_and_infrastructure_errors():
    gate = ReleaseGate(
        {
            "critical_requires_all_passed": True,
            "blocking_statuses": ["ENV_UNAVAILABLE", "FIXTURE_UNAVAILABLE", "EVALUATOR_ERROR"],
        }
    )

    decision = gate.evaluate(
        suite_results=[],
        case_results=[
            {"case_id": "critical-1", "severity": "critical", "status": "FAILED", "passed": False},
            {"case_id": "env-1", "severity": "high", "status": "ENV_UNAVAILABLE", "passed": False},
        ],
    )

    assert decision["status"] == "BLOCKED"
    assert len(decision["reasons"]) == 2


def test_runner_warms_up_and_repeats_performance_case_before_computing_p95(tmp_path):
    _write_case(
        tmp_path,
        "performance_cost",
        {
            "suite": "performance_cost",
            "case_id": "perf-001",
            "name": "重复性能测试",
            "tags": ["smoke"],
            "execution_mode": "controlled",
            "prompt": "查询工资规则",
            "warmups": 1,
            "repetitions": 3,
            "max_p95_ms": 30,
            "max_total_tokens": 300,
            "max_cost": 0.3,
        },
    )

    class PerformanceBackend:
        def __init__(self):
            self.calls = 0

        def execute(self, case):
            del case
            durations = [100, 10, 20, 30]
            duration = durations[self.calls]
            self.calls += 1
            return EvalObservation(
                status="COMPLETED",
                duration_ms=duration,
                token_usage={"total_tokens": 100},
                cost=0.1,
            )

    backend = PerformanceBackend()
    runner = EvalRunner(
        agent_service=object(),
        audit_store=RecordingAuditStore(),
        root=tmp_path,
        controlled_backend=backend,
        environment_checker=lambda requirements: [],
    )

    suite = runner.run_suite("performance_cost", mode="smoke")

    assert backend.calls == 4
    assert suite.status is EvalStatus.PASSED
    assert suite.results[0].metrics["p95_latency_ms"] == 30
    assert suite.results[0].metrics["total_tokens"] == 300
    assert suite.results[0].metrics["total_cost"] == 0.3


def test_release_gate_blocks_when_suite_metric_is_below_threshold():
    gate = ReleaseGate(
        {
            "critical_requires_all_passed": True,
            "blocking_statuses": ["ENV_UNAVAILABLE"],
            "metric_thresholds": {"rag_retrieval": {"mrr": {"min": 0.8}}},
        }
    )
    suite = EvalSuiteResult(
        suite="rag_retrieval",
        mode="release",
        status=EvalStatus.PASSED,
        total_cases=1,
        passed_cases=1,
        metrics={"mrr": 0.5},
    )

    decision = gate.evaluate(suite_results=[suite], case_results=[])

    assert decision["status"] == "BLOCKED"
    assert decision["reasons"] == ["rag_retrieval.mrr: 0.5 is below minimum 0.8"]


def test_runner_run_executes_all_registered_suites_and_applies_release_gate(monkeypatch):
    runner = EvalRunner(
        agent_service=object(),
        audit_store=RecordingAuditStore(),
        environment_checker=lambda requirements: [],
        release_gate=ReleaseGate({"blocking_statuses": ["ENV_UNAVAILABLE"]}),
    )
    called = []

    def run_suite(suite, *, mode):
        called.append((suite, mode))
        return EvalSuiteResult(
            suite=suite,
            mode=mode,
            status=EvalStatus.PASSED,
            total_cases=1,
            passed_cases=1,
        )

    monkeypatch.setattr(runner, "run_suite", run_suite)

    result = runner.run(mode="release")

    assert called == [(suite, "release") for suite in runner.registry.names()]
    assert result.status is EvalStatus.PASSED
    assert result.release_gate["status"] == "PASSED"
