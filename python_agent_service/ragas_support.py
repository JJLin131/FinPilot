from __future__ import annotations

from typing import Any

from python_agent_service.models import AgentChatResponse, EvalCase


def evaluate_ragas_case(case: EvalCase, response: AgentChatResponse) -> dict[str, float]:
    try:
        import ragas  # noqa: F401
    except Exception:
        return _heuristic_scores(case, response, available=False)
    return _heuristic_scores(case, response, available=True)


def _heuristic_scores(case: EvalCase, response: AgentChatResponse, *, available: bool) -> dict[str, float]:
    evidence_count = len(response.evidence)
    answer = response.answer or ""
    answer_hit = 1.0 if case.expected_answer_contains and case.expected_answer_contains in answer else 0.0
    retrieval_hit = 1.0 if case.relevant_document_ids and any(
        entry.summary.get("document_id") in case.relevant_document_ids for entry in response.evidence
    ) else 0.0
    tool_hit = 1.0 if case.expected_tool and any(tool.tool_name == case.expected_tool for tool in response.tool_calls or []) else 0.0
    faithfulness = min(1.0, 0.4 + evidence_count * 0.2 + answer_hit * 0.2)
    answer_relevance = min(1.0, 0.5 + answer_hit * 0.3 + tool_hit * 0.2)
    context_precision = retrieval_hit if case.relevant_document_ids else 1.0 if evidence_count else 0.0
    return {
        "faithfulness": round(faithfulness, 4),
        "answer_relevance": round(answer_relevance, 4),
        "context_precision": round(context_precision, 4),
        "ragas_available": 1.0 if available else 0.0,
    }

