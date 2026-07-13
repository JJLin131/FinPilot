from __future__ import annotations

from finpilot.evals.evaluators.base import contains_mapping, result
from finpilot.evals.models import EndToEndTaskCase, EvalObservation


class EndToEndTaskEvaluator:
    def evaluate(self, case: EndToEndTaskCase, observation: EvalObservation):
        answer = str(observation.response.get("answer") or "")
        state_ok = contains_mapping(observation.final_state, case.expected_final_state)
        constraints_ok = all(constraint in answer for constraint in case.required_constraints)
        forbidden_ok = not any(contains_mapping(observation.final_state, state) for state in case.forbidden_states)
        metrics = {
            "final_state_accuracy": float(state_ok),
            "constraint_satisfaction": float(constraints_ok),
            "forbidden_state_avoidance": float(forbidden_ok),
            "task_success": float(
                state_ok
                and constraints_ok
                and forbidden_ok
                and observation.status in {"SUCCEEDED", "COMPLETED"}
            ),
        }
        return result(case, observation, metrics, metrics["task_success"] == 1.0)
