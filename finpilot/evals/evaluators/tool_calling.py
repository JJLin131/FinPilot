from __future__ import annotations

from finpilot.evals.evaluators.base import contains_mapping, result
from finpilot.evals.models import EvalObservation, ToolCallingCase


class ToolCallingEvaluator:
    def evaluate(self, case: ToolCallingCase, observation: EvalObservation):
        actual_names = [str(call.get("tool_name")) for call in observation.tool_calls]
        expected_names = [call.tool_name for call in case.expected_calls]
        args_ok = len(observation.tool_calls) == len(case.expected_calls) and all(
            dict(actual.get("parameters") or {}) == expected.arguments
            for actual, expected in zip(observation.tool_calls, case.expected_calls, strict=True)
        )
        metrics = {
            "tool_sequence_accuracy": float(actual_names == expected_names),
            "argument_exact_match": float(args_ok),
            "forbidden_tool_count": float(len(set(actual_names) & set(case.forbidden_tools))),
            "final_state_match": float(contains_mapping(observation.final_state, case.expected_final_state)),
        }
        passed = metrics["tool_sequence_accuracy"] == 1.0 and metrics["argument_exact_match"] == 1.0 and metrics["forbidden_tool_count"] == 0 and metrics["final_state_match"] == 1.0
        return result(case, observation, metrics, passed)
