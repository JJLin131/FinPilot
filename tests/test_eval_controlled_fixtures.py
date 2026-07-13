from __future__ import annotations

from pathlib import Path

from finpilot.evals.backends import ControlledBackend
from finpilot.evals.evaluators import build_default_evaluators
from finpilot.evals.evaluators.rag_generation import RagGenerationEvaluator
from finpilot.evals.loader import load_eval_cases
from finpilot.evals.registry import DEFAULT_SUITE_REGISTRY, EVALUATION_SUITES


class PassingRagasJudge:
    def score(self, case, observation):
        del case, observation
        return {"faithfulness": 0.95, "answer_correctness": 0.95, "answer_relevance": 0.95}


def test_all_controlled_dataset_fixtures_are_executable_and_meet_expectations():
    backend = ControlledBackend({})
    evaluators = build_default_evaluators()
    evaluators["rag_generation"] = RagGenerationEvaluator(PassingRagasJudge())
    controlled_cases = [
        case
        for suite in EVALUATION_SUITES
        for case in load_eval_cases(Path(f"evals/datasets/{suite}.jsonl"), DEFAULT_SUITE_REGISTRY)
        if case.execution_mode == "controlled"
    ]

    results = [evaluators[case.suite].evaluate(case, backend.execute(case)) for case in controlled_cases]

    assert controlled_cases
    assert all(result.passed for result in results), [
        {"case_id": result.case_id, "metrics": result.metrics} for result in results if not result.passed
    ]
