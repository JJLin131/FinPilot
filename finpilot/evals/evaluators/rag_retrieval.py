from __future__ import annotations

import math

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, RagRetrievalCase


class RagRetrievalEvaluator:
    def evaluate(self, case: RagRetrievalCase, observation: EvalObservation):
        ids = [str(item.get("document_id")) for item in observation.retrieved_documents if item.get("document_id")]
        relevant = set(case.relevant_document_ids)
        metrics: dict[str, float] = {}
        for k in case.k_values:
            top = ids[:k]
            hits = len(set(top) & relevant)
            metrics[f"recall_at_{k}"] = round(hits / len(relevant), 4) if relevant else float(not top)
            metrics[f"precision_at_{k}"] = round(hits / len(top), 4) if top else float(not relevant)
            metrics[f"hit_at_{k}"] = float(hits > 0)
            metrics[f"ndcg_at_{k}"] = self._ndcg(top, relevant)
        first_rank = next((index for index, item in enumerate(ids, start=1) if item in relevant), None)
        metrics["mrr"] = round(1 / first_rank, 4) if first_rank else 0.0
        metrics["forbidden_hit_count"] = float(len(set(ids) & set(case.forbidden_document_ids)))
        max_k = max(case.k_values)
        passed = (
            metrics[f"recall_at_{max_k}"] == 1.0
            and metrics["forbidden_hit_count"] == 0
            and ((not ids) if case.expected_no_answer else True)
        )
        return result(case, observation, metrics, passed)

    @staticmethod
    def _ndcg(ids: list[str], relevant: set[str]) -> float:
        dcg = sum((1 / math.log2(index + 2)) for index, item in enumerate(ids) if item in relevant)
        ideal_hits = min(len(relevant), len(ids))
        ideal = sum(1 / math.log2(index + 2) for index in range(ideal_hits))
        return round(dcg / ideal, 4) if ideal else float(not ids)
