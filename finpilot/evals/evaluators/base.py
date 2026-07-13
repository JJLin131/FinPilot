from __future__ import annotations

from typing import Protocol

from finpilot.evals.models import BaseEvalCase, EvalCaseResult, EvalObservation, EvalStatus


class Evaluator(Protocol):
    def evaluate(self, case: BaseEvalCase, observation: EvalObservation) -> EvalCaseResult: ...


def result(case: BaseEvalCase, observation: EvalObservation, metrics: dict[str, object], passed: bool) -> EvalCaseResult:
    return EvalCaseResult(
        suite=case.suite,
        case_id=case.case_id,
        severity=case.severity,
        status=EvalStatus.PASSED if passed else EvalStatus.FAILED,
        passed=passed,
        metrics=metrics,
        raw_output=observation.model_dump(mode="json"),
        duration_ms=observation.duration_ms,
        token_usage=observation.token_usage,
    )


def f1(expected: set, actual: set) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    overlap = len(expected & actual)
    precision = overlap / len(actual)
    recall = overlap / len(expected)
    return round(2 * precision * recall / (precision + recall), 4) if overlap else 0.0


def contains_mapping(actual: dict, expected: dict) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())
