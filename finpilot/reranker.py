from __future__ import annotations

import logging

import httpx

from finpilot.config import settings
from finpilot.issues import dependency_degraded_issue
from finpilot.models import AgentIssue, RagMatch

logger = logging.getLogger(__name__)


class RemoteReranker:
    def __init__(self) -> None:
        self._issues: list[AgentIssue] = []

    def rerank(self, query: str, candidates: list[RagMatch], limit: int = 5) -> list[RagMatch]:
        self._issues = []
        if not settings.reranker_enabled or not candidates:
            return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]
        try:
            with httpx.Client(timeout=settings.reranker_timeout_seconds) as client:
                response = client.post(
                    f"{settings.reranker_base_url.rstrip('/')}/rerank",
                    json={
                        "query": query,
                        "texts": [candidate.text for candidate in candidates],
                        "raw_scores": False,
                        "return_text": False,
                        "truncate": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("results") if isinstance(payload, dict) else payload
            if not isinstance(rows, list) or len(rows) != len(candidates):
                raise ValueError("Invalid reranker response")
            scores_by_index = {int(item["index"]): float(item["score"]) for item in rows}
            reranked = [
                RagMatch(
                    document_id=candidate.document_id,
                    title=candidate.title,
                    source=candidate.source,
                    text=candidate.text,
                    score=scores_by_index.get(index, candidate.score),
                )
                for index, candidate in enumerate(candidates)
            ]
            return sorted(reranked, key=lambda item: item.score, reverse=True)[:limit]
        except Exception as exc:
            logger.warning("Remote reranker failed, using local score order: %s", exc)
            self._issues.append(
                dependency_degraded_issue(
                    code="RAG_RERANKER_DEGRADED",
                    component="rag:reranker",
                    message="Remote reranker failed; using local score order.",
                    exc=exc,
                )
            )
            return sorted(candidates, key=lambda item: item.score, reverse=True)[:limit]

    def consume_issues(self) -> list[AgentIssue]:
        issues = list(self._issues)
        self._issues = []
        return issues


