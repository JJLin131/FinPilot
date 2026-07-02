from __future__ import annotations

from FinanceAgent.rag.service import RagKnowledgeService


class KnowledgeLifecycleService:
    def __init__(self, service: RagKnowledgeService):
        self.service = service

    def remove_expired_documents(self) -> int:
        return self.service.delete_expired_documents()
