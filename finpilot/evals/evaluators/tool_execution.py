from __future__ import annotations

from finpilot.evals.evaluators.base import contains_mapping, result
from finpilot.evals.evaluators.tool_selection import calls_match
from finpilot.evals.models import EvalObservation, ToolExecutionCase


class ToolExecutionEvaluator:
    def evaluate(self, case: ToolExecutionCase, observation: EvalObservation):
        tool_match, argument_match = calls_match(
            observation.tool_calls,
            case.expected_calls,
            order_matters=True,
        )
        actual_names = {str(call.get("tool_name") or "") for call in observation.tool_calls}
        status_match = case.expected_status is None or observation.status == case.expected_status
        metrics = {
            "execution_status_match": float(status_match),
            "tool_execution_accuracy": float(tool_match),
            "execution_argument_accuracy": float(argument_match),
            "forbidden_tool_count": float(len(actual_names & set(case.forbidden_tools))),
            "final_state_match": float(contains_mapping(observation.final_state, case.expected_final_state)),
            "side_effect_match": float(
                not case.expected_final_state
                or contains_mapping(observation.final_state, case.expected_final_state)
            ),
            "side_effect_count": float(observation.side_effect_count),
        }
        passed = (
            metrics["execution_status_match"] == 1.0
            and metrics["tool_execution_accuracy"] == 1.0
            and metrics["execution_argument_accuracy"] == 1.0
            and metrics["forbidden_tool_count"] == 0
            and metrics["final_state_match"] == 1.0
        )
        return result(case, observation, metrics, passed)
