from __future__ import annotations

import json

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, ObservabilityAuditCase


class ObservabilityAuditEvaluator:
    def evaluate(self, case: ObservabilityAuditCase, observation: EvalObservation):
        spans = {str(item.get("name")) for item in observation.traces}
        scores = {str(item.get("name")) for item in observation.scores}
        events = {str(item.get("type")) for item in observation.audit_events}
        serialized = json.dumps(observation.model_dump(mode="json"), ensure_ascii=False)
        metrics = {
            "trace_coverage": self._coverage(case.required_spans, spans),
            "score_coverage": self._coverage(case.required_scores, scores),
            "audit_coverage": self._coverage(case.required_audit_events, events),
            "sensitive_field_leak_count": float(sum(field in serialized for field in case.forbidden_fields)),
        }
        passed = metrics["trace_coverage"] == 1.0 and metrics["score_coverage"] == 1.0 and metrics["audit_coverage"] == 1.0 and metrics["sensitive_field_leak_count"] == 0
        return result(case, observation, metrics, passed)

    @staticmethod
    def _coverage(expected: list[str], actual: set[str]) -> float:
        return round(sum(item in actual for item in expected) / len(expected), 4) if expected else 1.0
