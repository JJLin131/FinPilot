from __future__ import annotations

import inspect
import json

from finpilot.agent.service import FinPilotService
from finpilot.models import FinPilotChatRequest
from finpilot.rag.bm25 import Bm25ChunkIndex
from finpilot.rag.models import KnowledgeDocumentRequest


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
