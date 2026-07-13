from __future__ import annotations

from finpilot.evals.evaluators.base import result
from finpilot.evals.judges.ragas_judge import RagasJudge
from finpilot.evals.models import EvalObservation, RagGenerationCase


class RagGenerationEvaluator:
    def __init__(self, judge=None, *, minimum_score: float = 0.8):
        self.judge = judge or RagasJudge()
        self.minimum_score = minimum_score

    def evaluate(self, case: RagGenerationCase, observation: EvalObservation):
        answer = str(observation.response.get("answer") or "")
        metrics = dict(self.judge.score(case, observation))
        evidence = observation.response.get("evidence") or []
        citations = {
            str(item.get("summary", {}).get("document_id"))
            for item in evidence
            if isinstance(item, dict) and item.get("summary", {}).get("document_id")
        }
        expected = set(case.expected_citations)
        metrics["citation_recall"] = round(len(citations & expected) / len(expected), 4) if expected else 1.0
        metrics["required_claim_recall"] = (
            round(sum(claim in answer for claim in case.required_claims) / len(case.required_claims), 4)
            if case.required_claims
            else 1.0
        )
        metrics["forbidden_claim_count"] = float(sum(claim in answer for claim in case.forbidden_claims))
        judge_ok = all(float(metrics.get(name, 0.0)) >= self.minimum_score for name in ("faithfulness", "answer_correctness", "answer_relevance"))
        passed = judge_ok and metrics["citation_recall"] == 1.0 and metrics["required_claim_recall"] == 1.0 and metrics["forbidden_claim_count"] == 0
        return result(case, observation, metrics, passed)
