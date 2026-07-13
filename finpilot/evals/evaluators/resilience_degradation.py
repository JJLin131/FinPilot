from __future__ import annotations

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, ResilienceDegradationCase


class ResilienceDegradationEvaluator:
    def evaluate(self, case: ResilienceDegradationCase, observation: EvalObservation):
        metrics = {
            "status_match": float(observation.status == case.expected_status),
            "attempt_budget_respected": float(observation.attempts <= case.max_attempts),
            "side_effect_match": float(observation.side_effect_count == case.expected_side_effect_count),
        }
        return result(case, observation, metrics, all(value == 1.0 for value in metrics.values()))
