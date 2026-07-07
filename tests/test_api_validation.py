from __future__ import annotations

import pytest
from pydantic import ValidationError

from finpilot.models import FinPilotChatRequest
from finpilot.rag.models import KnowledgeDocumentRequest


def test_chat_request_limits_user_chat_and_content_lengths():
    with pytest.raises(ValidationError):
        FinPilotChatRequest(user_id="u" * 65, chat_id="chat-1", content="hello")

    with pytest.raises(ValidationError):
        FinPilotChatRequest(user_id="user-1", chat_id="c" * 129, content="hello")

    with pytest.raises(ValidationError):
        FinPilotChatRequest(user_id="user-1", chat_id="chat-1", content="x" * 8001)


def test_knowledge_document_request_limits_domain_and_text_fields():
    with pytest.raises(ValidationError):
        KnowledgeDocumentRequest(
            document_id="doc-1",
            domain="HR",
            title="title",
            source="manual",
            content="content",
        )

    with pytest.raises(ValidationError):
        KnowledgeDocumentRequest(
            document_id="doc-1",
            domain="FINANCE",
            title="x" * 257,
            source="manual",
            content="content",
        )

    with pytest.raises(ValidationError):
        KnowledgeDocumentRequest(
            document_id="doc-1",
            domain="FINANCE",
            title="title",
            source="manual",
            content="x" * 200001,
        )
