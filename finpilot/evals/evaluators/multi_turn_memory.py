from __future__ import annotations

from finpilot.evals.evaluators.base import contains_mapping, result
from finpilot.evals.models import EvalObservation, MultiTurnMemoryCase


class MultiTurnMemoryEvaluator:
    def evaluate(self, case: MultiTurnMemoryCase, observation: EvalObservation):
        memories = dict(observation.final_state.get("memories") or {})
        isolated = dict(observation.final_state.get("isolation_memories") or {})
        metrics = {
            "memory_match": float(contains_mapping(memories, case.expected_memories)),
            "forbidden_memory_count": float(sum(key in memories for key in case.forbidden_memories)),
            "forget_success": float(all(key not in memories for key in case.expected_forget_keys)),
            "cross_user_leak_count": float(sum(key in isolated for key in case.expected_memories)),
        }
        passed = metrics["memory_match"] == 1.0 and metrics["forbidden_memory_count"] == 0 and metrics["forget_success"] == 1.0 and metrics["cross_user_leak_count"] == 0
        return result(case, observation, metrics, passed)
