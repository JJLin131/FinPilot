from __future__ import annotations

from finpilot.evals.evaluators.base import result
from finpilot.evals.models import EvalObservation, QueryRewriteRerankerCase


class QueryRewriteRerankerEvaluator:
    def evaluate(self, case: QueryRewriteRerankerCase, observation: EvalObservation):
        relevant = set(case.relevant_document_ids)
        before = [str(item.get("document_id")) for item in observation.retrieved_documents if item.get("document_id")]
        after = [str(item.get("document_id")) for item in observation.ranked_documents if item.get("document_id")]
        recall_before = len(set(before) & relevant) / len(relevant) if relevant else float(not before)
        recall_after = len(set(after) & relevant) / len(relevant) if relevant else float(not after)
        ranked_text = " ".join(str(item.get("text") or "") for item in observation.ranked_documents)
        metrics = {
            "rewrite_success": float(bool(observation.rewritten_query and observation.rewritten_query.strip())),
            "rewrite_recall_lift": round(recall_after - recall_before, 4),
            "rerank_recall": round(recall_after, 4),
            "false_recall_count": float(len(set(after) & set(case.forbidden_document_ids))),
            "subtopic_recall": round(sum(topic in ranked_text for topic in case.subtopics) / len(case.subtopics), 4) if case.subtopics else 1.0,
        }
        passed = metrics["rewrite_success"] == 1.0 and metrics["rerank_recall"] == 1.0 and metrics["false_recall_count"] == 0 and metrics["subtopic_recall"] == 1.0
        return result(case, observation, metrics, passed)
