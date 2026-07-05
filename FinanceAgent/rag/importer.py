from __future__ import annotations

from pathlib import Path

from FinanceAgent.config import settings
from FinanceAgent.rag.chunker import KnowledgeChunker
from FinanceAgent.rag.models import KnowledgeDocumentRequest, KnowledgeDocumentResult
from FinanceAgent.rag.registry import KnowledgeDocumentRegistry


class ResourceKnowledgeImporter:
    GLOBAL_TENANT = "__GLOBAL__"

    def __init__(self, registry: KnowledgeDocumentRegistry, chunker: KnowledgeChunker, knowledge_dir: Path | None = None):
        self.registry = registry
        self.chunker = chunker
        self.knowledge_dir = knowledge_dir or settings.knowledge_dir

    def import_finance_resources(self) -> list[KnowledgeDocumentResult]:
        results: list[KnowledgeDocumentResult] = []
        for path in sorted(self.knowledge_dir.glob("*.md")):
            if not path.name[:2].isdigit():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            request = KnowledgeDocumentRequest(
                document_id=f"resource-finance-{abs(hash(path.name))}",
                domain="FINANCE",
                tenant_id=self.GLOBAL_TENANT,
                title=path.stem,
                source=str(path),
                content=content,
                tags=self._tags(path),
            )
            chunks = self.chunker.split(request.content)
            chunk_ids = self.registry.chunk_ids_for(request.document_id, len(chunks))
            self.registry.save_document(request, len(chunks))
            self.registry.replace_chunks(request, chunks, chunk_ids)
            results.append(
                KnowledgeDocumentResult(
                    document_id=request.document_id,
                    domain=request.domain,
                    tenant_id=request.tenant_id,
                    status="ACTIVE",
                    chunk_count=len(chunks),
                )
            )
        return results

    def _tags(self, path: Path) -> list[str]:
        stem = path.stem
        return [part for part in stem.replace("_", " ").split() if part]
