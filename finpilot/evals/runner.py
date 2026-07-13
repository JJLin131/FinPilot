from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

from finpilot.evals.backends import ControlledBackend, EvalBackendError, LiveBackend
from finpilot.evals.evaluators import build_default_evaluators
from finpilot.evals.environment import RuntimeEnvironmentChecker
from finpilot.evals.gates.release import ReleaseGate
from finpilot.evals.judges.ragas_judge import JudgeUnavailable
from finpilot.evals.loader import load_eval_cases
from finpilot.evals.models import (
    BaseEvalCase,
    EvalCaseResult,
    EvalFailure,
    EvalRunResult,
    EvalStatus,
    EvalSuiteResult,
    PerformanceCostCase,
)
from finpilot.evals.registry import DEFAULT_SUITE_REGISTRY, SuiteRegistry
EnvironmentChecker = Callable[[tuple[str, ...]], list[str]]
ScorePublisher = Callable[..., None]


class EvalRunner:
    def __init__(
        self,
        agent_service,
        audit_store,
        root: Path | None = None,
        *,
        registry: SuiteRegistry = DEFAULT_SUITE_REGISTRY,
        controlled_backend: ControlledBackend | None = None,
        evaluators: dict[str, object] | None = None,
        environment_checker: EnvironmentChecker | None = None,
        run_id: str = "local",
        release_gate: ReleaseGate | None = None,
        score_publisher: ScorePublisher | None = None,
    ):
        self.agent_service = agent_service
        self.audit_store = audit_store
        self.root = root or Path("evals/datasets")
        self.registry = registry
        self.live_backend = LiveBackend(agent_service, user_id="eval-user", run_id=run_id)
        self.controlled_backend = controlled_backend or ControlledBackend({})
        self.evaluators = evaluators or build_default_evaluators()
        self.environment_checker = environment_checker or RuntimeEnvironmentChecker()
        self.run_id = run_id
        self.score_publisher = score_publisher
        gate_path = self.root.parent / "release_gate.json"
        self.release_gate = release_gate or (
            ReleaseGate.from_path(gate_path)
            if gate_path.exists()
            else ReleaseGate({
                "critical_requires_all_passed": True,
                "blocking_statuses": ["ENV_UNAVAILABLE", "FIXTURE_UNAVAILABLE", "EVALUATOR_ERROR"],
            })
        )

    def run(self, *, mode: str = "smoke") -> EvalRunResult:
        suites = [self.run_suite(suite, mode=mode) for suite in self.registry.names()]
        case_results = [item.model_dump(mode="json") for suite in suites for item in suite.results]
        gate = self.release_gate.evaluate(suite_results=suites, case_results=case_results)
        if mode == "release" and gate["status"] == "BLOCKED":
            status = EvalStatus.BLOCKED
        else:
            status = self._run_status(suites)
        return EvalRunResult(mode=mode, status=status, suites=suites, release_gate=gate)

    def run_suite(self, suite: str, *, mode: str = "smoke") -> EvalSuiteResult:
        definition = self.registry.definition(suite)
        cases = self._select_cases(load_eval_cases(self.root / f"{suite}.jsonl", self.registry), mode)
        results = [self._run_case(case, definition.required_environment) for case in cases]
        suite_result = EvalSuiteResult(
            suite=suite,
            mode=mode,
            status=self._suite_status(results),
            total_cases=len(results),
            passed_cases=sum(result.passed for result in results),
            results=results,
            metrics=self._aggregate_metrics(results),
            slices=self._aggregate_slices(cases, results),
            environment=self._environment_summary(results),
        )
        self.audit_store.record_eval_run(suite_result)
        if self.score_publisher is not None:
            self.score_publisher(suite_result, run_id=self.run_id)
        return suite_result

    def _run_case(self, case: BaseEvalCase, requirements: tuple[str, ...]) -> EvalCaseResult:
        if case.execution_mode == "live" and self.agent_service is None:
            return self._failure(
                case,
                EvalStatus.ENV_UNAVAILABLE,
                "RUNTIME_SERVICE_UNAVAILABLE",
                "FinPilot runtime service could not be initialized.",
            )
        checked_requirements = (
            requirements
            if case.execution_mode == "live"
            else tuple(requirement for requirement in requirements if requirement in {"ragas", "llm", "embedding"})
        )
        missing = self.environment_checker(checked_requirements) if checked_requirements else []
        if missing:
            return self._failure(case, EvalStatus.ENV_UNAVAILABLE, "ENV_UNAVAILABLE", f"Missing: {', '.join(missing)}")
        backend = self.live_backend if case.execution_mode == "live" else self.controlled_backend
        try:
            observation = self._execute_case(backend, case)
            evaluator = self.evaluators[case.suite]
            return evaluator.evaluate(case, observation)
        except EvalBackendError as exc:
            return self._failure(case, exc.status, exc.code, str(exc))
        except JudgeUnavailable as exc:
            return self._failure(case, EvalStatus.ENV_UNAVAILABLE, "JUDGE_UNAVAILABLE", str(exc))
        except Exception as exc:
            return self._failure(case, EvalStatus.EVALUATOR_ERROR, "EVALUATOR_ERROR", str(exc))

    @staticmethod
    def _execute_case(backend, case: BaseEvalCase):
        if not isinstance(case, PerformanceCostCase):
            return backend.execute(case)
        for index in range(case.warmups):
            backend.execute(case.model_copy(update={"case_id": f"{case.case_id}-warmup-{index + 1}"}))
        samples = [
            backend.execute(case.model_copy(update={"case_id": f"{case.case_id}-sample-{index + 1}"}))
            for index in range(case.repetitions)
        ]
        token_usage: dict[str, int] = defaultdict(int)
        for sample in samples:
            for name, value in sample.token_usage.items():
                token_usage[name] += value
        return samples[-1].model_copy(
            update={
                "duration_samples_ms": [sample.duration_ms for sample in samples],
                "token_usage": dict(token_usage),
                "cost": round(sum(sample.cost for sample in samples), 8),
                "attempts": len(samples),
            }
        )

    @staticmethod
    def _select_cases(cases: list[BaseEvalCase], mode: str) -> list[BaseEvalCase]:
        if mode not in {"smoke", "regression", "release"}:
            raise ValueError(f"Unknown evaluation mode: {mode}")
        if mode == "smoke":
            selected = [case for case in cases if "smoke" in case.tags]
            return selected or cases[:1]
        if mode == "regression":
            return [case for case in cases if "release_only" not in case.tags]
        return cases

    @staticmethod
    def _failure(case: BaseEvalCase, status: EvalStatus, code: str, message: str) -> EvalCaseResult:
        return EvalCaseResult(
            suite=case.suite,
            case_id=case.case_id,
            severity=case.severity,
            status=status,
            passed=False,
            failure=EvalFailure(code=code, message=message),
        )

    @staticmethod
    def _suite_status(results: list[EvalCaseResult]) -> EvalStatus:
        priority = (EvalStatus.EVALUATOR_ERROR, EvalStatus.FIXTURE_UNAVAILABLE, EvalStatus.ENV_UNAVAILABLE)
        for status in priority:
            if any(result.status is status for result in results):
                return status
        return EvalStatus.PASSED if results and all(result.passed for result in results) else EvalStatus.FAILED

    @staticmethod
    def _run_status(suites: list[EvalSuiteResult]) -> EvalStatus:
        priority = (
            EvalStatus.EVALUATOR_ERROR,
            EvalStatus.FIXTURE_UNAVAILABLE,
            EvalStatus.ENV_UNAVAILABLE,
            EvalStatus.FAILED,
        )
        for status in priority:
            if any(suite.status is status for suite in suites):
                return status
        return EvalStatus.PASSED if suites else EvalStatus.FAILED

    @staticmethod
    def _aggregate_metrics(results: list[EvalCaseResult]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for item in results:
            for name, value in item.metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values[name].append(float(value))
        return {name: round(sum(items) / len(items), 4) for name, items in values.items() if items}

    @staticmethod
    def _aggregate_slices(cases: list[BaseEvalCase], results: list[EvalCaseResult]) -> dict[str, dict[str, int]]:
        slices: dict[str, dict[str, int]] = {}
        for case, item in zip(cases, results, strict=True):
            for tag in case.tags:
                summary = slices.setdefault(tag, {"total": 0, "passed": 0})
                summary["total"] += 1
                summary["passed"] += int(item.passed)
        return slices

    @staticmethod
    def _environment_summary(results: list[EvalCaseResult]) -> dict[str, object]:
        failures = [
            {
                "case_id": item.case_id,
                "status": item.status.value,
                "code": item.failure.code,
                "message": item.failure.message,
            }
            for item in results
            if item.failure is not None
            and item.status in {EvalStatus.ENV_UNAVAILABLE, EvalStatus.FIXTURE_UNAVAILABLE}
        ]
        return {"available": not failures, "blocking_failures": failures}
