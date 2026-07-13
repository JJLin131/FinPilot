from __future__ import annotations

import inspect
import json
import uuid

from finpilot.agent.service import FinPilotService
from finpilot.models import FinPilotChatRequest, RagMatch
from finpilot.rag.bm25 import Bm25ChunkIndex
from finpilot.rag.importer import ResourceKnowledgeImporter
from finpilot.rag.models import KnowledgeDocumentRequest
from finpilot.rag.service import RagKnowledgeService


def test_chat_request_has_no_tenant_id():
    request = FinPilotChatRequest.model_validate(
        {
            "user_id": "user-1",
            "chat_id": "chat-1",
            "content": "工资发放审批规则是什么？",
        }
    )

    assert request.user_id == "user-1"
    assert "tenant_id" not in FinPilotChatRequest.model_fields


def test_service_chat_signature_is_user_scoped():
    signature = inspect.signature(FinPilotService.chat)

    assert list(signature.parameters) == ["self", "user_id", "chat_id", "content"]


def test_bm25_search_uses_shared_domain_scope(tmp_path):
    index = Bm25ChunkIndex(index_path=tmp_path / "knowledge.json")
    request = KnowledgeDocumentRequest(
        document_id="doc-1",
        domain="FINANCE",
        title="工资发放审批规则",
        source="manual",
        content="工资发放需要审批。",
        tags=["工资", "审批"],
    )

    index.replace_document(request, ["chunk-1"], ["工资发放需要审批。"])

    matches = index.search("FINANCE", "工资审批", limit=3)

    assert [match.document_id for match in matches] == ["doc-1"]


def test_bm25_reads_legacy_tenant_index(tmp_path):
    index_path = tmp_path / "knowledge.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "domain": "FINANCE",
                    "tenant_id": "__GLOBAL__",
                    "title": "工资发放审批规则",
                    "source": "manual",
                    "text": "工资发放需要审批。",
                    "tags": "工资 审批",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    index = Bm25ChunkIndex(index_path=index_path)

    matches = index.search("FINANCE", "工资审批", limit=3)

    assert [match.document_id for match in matches] == ["doc-1"]


def test_resource_importer_uses_stable_uuid_document_ids():
    expected = f"resource-finance-{uuid.uuid5(uuid.NAMESPACE_URL, '01_demo.md')}"

    assert ResourceKnowledgeImporter._resource_document_id("01_demo.md") == expected


def test_rag_search_trace_exposes_rewrites_candidates_and_reranked_results():
    match = RagMatch(document_id="doc-1", title="工资", source="manual", text="工资需要审批", score=0.8)

    class Rewriter:
        def rewrite(self, query):
            assert query == "工资咋发"
            return ["企业工资发放规则"]

    class Retriever:
        def retrieve(self, domain, query, limit):
            assert (domain, query, limit) == ("FINANCE", "企业工资发放规则", 12)
            return [match]

    class Registry:
        def is_active(self, document_id, domain, today):
            del today
            return (document_id, domain) == ("doc-1", "FINANCE")

    class Reranker:
        def rerank(self, query, candidates, limit):
            assert query == "工资咋发"
            assert candidates == [match]
            assert limit == 3
            return candidates

    service = object.__new__(RagKnowledgeService)
    service.query_rewriter = Rewriter()
    service.retriever = Retriever()
    service.registry = Registry()
    service.reranker = Reranker()

    trace = service.search_trace("工资咋发", limit=3)

    assert trace["rewritten_queries"] == ["企业工资发放规则"]
    assert trace["candidates"] == [match]
    assert trace["reranked"] == [match]
    assert trace["issues"] == []
