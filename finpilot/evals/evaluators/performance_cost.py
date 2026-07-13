from __future__ import annotations

import math

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, PerformanceCostCase


class PerformanceCostEvaluator:
    def evaluate(self, case: PerformanceCostCase, observation: EvalObservation):
        total_tokens = sum(observation.token_usage.values())
        durations = sorted(observation.duration_samples_ms or [observation.duration_ms])
        p95_index = max(0, math.ceil(0.95 * len(durations)) - 1)
        p95_latency_ms = durations[p95_index]
        metrics = {
            "p95_latency_ms": p95_latency_ms,
            "latency_within_limit": float(p95_latency_ms <= case.max_p95_ms),
            "total_tokens": float(total_tokens),
            "tokens_within_limit": float(total_tokens <= case.max_total_tokens),
            "total_cost": observation.cost,
            "cost_within_limit": float(case.max_cost is None or observation.cost <= case.max_cost),
            "usage_available": float(total_tokens > 0),
        }
        passed = all(
            metrics[key] == 1.0
            for key in ("latency_within_limit", "tokens_within_limit", "cost_within_limit", "usage_available")
        )
        return result(case, observation, metrics, passed)
