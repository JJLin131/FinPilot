from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from finpilot.evals.models import BaseEvalCase
from finpilot.evals.registry import SuiteRegistry


class EvalDatasetError(ValueError):
    pass


def load_eval_cases(path: Path, registry: SuiteRegistry) -> list[BaseEvalCase]:
    if not path.exists():
        raise EvalDatasetError(f"dataset file does not exist: {path}")
    expected_suite = path.stem
    registry.definition(expected_suite)
    cases: list[BaseEvalCase] = []
    seen_case_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise EvalDatasetError(f"{path.name}:{line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise EvalDatasetError(f"{path.name}:{line_number}: case must be a JSON object")
        actual_suite = payload.get("suite")
        if actual_suite != expected_suite:
            raise EvalDatasetError(
                f"{path.name}:{line_number}: suite {actual_suite} does not match dataset file {expected_suite}"
            )
        try:
            case = registry.parse(payload)
        except (ValueError, ValidationError) as exc:
            raise EvalDatasetError(f"{path.name}:{line_number}: {exc}") from exc
        if case.case_id in seen_case_ids:
            raise EvalDatasetError(f"{path.name}:{line_number}: duplicate case_id {case.case_id}")
        seen_case_ids.add(case.case_id)
        cases.append(case)
    if not cases:
        raise EvalDatasetError(f"dataset contains no cases: {path}")
    return cases
