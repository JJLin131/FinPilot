from __future__ import annotations

from collections import Counter
from typing import Any

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, ExpectedToolCall, ToolSelectionCase


def normalize_arguments(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): normalize_arguments(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [normalize_arguments(item) for item in value]
    return value


def calls_match(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[ExpectedToolCall],
    *,
    order_matters: bool,
) -> tuple[bool, bool]:
    actual_names = [str(call.get("tool_name") or "") for call in actual_calls]
    expected_names = [call.tool_name for call in expected_calls]
    names_match = (
        actual_names == expected_names
        if order_matters
        else Counter(actual_names) == Counter(expected_names)
    )
    if not names_match:
        return False, argument_match_rate(actual_calls, expected_calls) == 1.0
    remaining = list(actual_calls)
    for expected in expected_calls:
        index = next(
            (position for position, call in enumerate(remaining) if call.get("tool_name") == expected.tool_name),
            None,
        )
        if index is None:
            return True, False
        actual = remaining.pop(index)
        if expected.agent_name is not None and actual.get("agent_name") != expected.agent_name:
            return False, False
        if normalize_arguments(actual.get("parameters") or {}) != normalize_arguments(expected.arguments):
            return True, False
    return True, True


def argument_match_rate(
    actual_calls: list[dict[str, Any]],
    expected_calls: list[ExpectedToolCall],
) -> float:
    if not expected_calls:
        return 1.0
    remaining = list(actual_calls)
    matched = 0
    for expected in expected_calls:
        index = next(
            (
                position
                for position, call in enumerate(remaining)
                if call.get("tool_name") == expected.tool_name
                and (expected.agent_name is None or call.get("agent_name") == expected.agent_name)
            ),
            None,
        )
        if index is None:
            continue
        actual = remaining.pop(index)
        if normalize_arguments(actual.get("parameters") or {}) == normalize_arguments(expected.arguments):
            matched += 1
    return round(matched / len(expected_calls), 4)


class ToolSelectionEvaluator:
    def evaluate(self, case: ToolSelectionCase, observation: EvalObservation):
        actual_agents = {
            str(node.get("agent_name"))
            for node in observation.plan.get("nodes") or []
            if node.get("agent_name")
        }
        expected_agents = set(case.expected.agents)
        tool_match, _ = calls_match(
            observation.tool_calls,
            case.expected.tool_calls,
            order_matters=case.expected.order_matters,
        )
        argument_accuracy = argument_match_rate(observation.tool_calls, case.expected.tool_calls)
        expected_names = Counter(call.tool_name for call in case.expected.tool_calls)
        actual_names = Counter(str(call.get("tool_name") or "") for call in observation.tool_calls)
        missing_count = sum((expected_names - actual_names).values())
        extra_count = sum((actual_names - expected_names).values())
        schema_values = [float(call.get("schema_valid") is True) for call in observation.tool_calls]
        schema_validity = sum(schema_values) / len(schema_values) if schema_values else 1.0
        metrics = {
            "decision_status_match": float(observation.status == case.expected.status),
            "agent_selection_accuracy": float(actual_agents == expected_agents),
            "tool_selection_accuracy": float(tool_match),
            "tool_misjudgment_rate": float(not tool_match),
            "argument_accuracy": argument_accuracy,
            "parameter_error_rate": round(1.0 - argument_accuracy, 4),
            "argument_schema_validity": round(schema_validity, 4),
            "missing_tool_count": float(missing_count),
            "extra_tool_count": float(extra_count),
        }
        passed = all(
            metrics[name] == 1.0
            for name in (
                "decision_status_match",
                "agent_selection_accuracy",
                "tool_selection_accuracy",
                "argument_accuracy",
                "argument_schema_validity",
            )
        )
        return result(case, observation, metrics, passed)
