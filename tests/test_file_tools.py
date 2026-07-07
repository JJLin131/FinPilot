from __future__ import annotations

import json

from finpilot.agent.tools import ToolRegistry
from finpilot.models import GraphState
from finpilot.safety.approval import ApprovalDecision, ApprovalService
from finpilot.safety.service import SafetyReviewService
from finpilot.config import settings


class EmptyRagService:
    def search(self, query: str, limit: int = 3):
        return []


def _state() -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="read local project file",
        normalized_intent="GENERAL_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


def _write_access(path, *, read=None, write=None) -> None:
    path.write_text(
        json.dumps(
            {
                "read": read or [],
                "write": write or [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_read_file_requires_configured_read_access(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    note = allowed / "note.txt"
    note.write_text("finance note", encoding="utf-8")
    denied = tmp_path / "denied.txt"
    denied.write_text("secret", encoding="utf-8")
    access_path = tmp_path / "tool-access.json"
    _write_access(access_path, read=[{"path": str(allowed), "recursive": True}])
    monkeypatch.setattr(settings, "tool_access_path", access_path, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    allowed_invocation = registry.invoke(_state(), "read_file", path=str(note))
    denied_invocation = registry.invoke(_state(), "read_file", path=str(denied))

    assert allowed_invocation.status == "SUCCEEDED"
    assert allowed_invocation.output["content"][0]["source"] == str(note.resolve())
    assert allowed_invocation.output["content"][0]["text"] == "finance note"
    assert denied_invocation.status == "FAILED"
    assert "not authorized" in denied_invocation.output["error"]


def test_list_files_only_lists_authorized_directory(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    nested = allowed / "nested"
    nested.mkdir(parents=True)
    (allowed / "a.txt").write_text("a", encoding="utf-8")
    (nested / "b.txt").write_text("b", encoding="utf-8")
    access_path = tmp_path / "tool-access.json"
    _write_access(access_path, read=[{"path": str(allowed), "recursive": True}])
    monkeypatch.setattr(settings, "tool_access_path", access_path, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "list_files", path=str(allowed), recursive=True)

    assert invocation.status == "SUCCEEDED"
    assert [item["name"] for item in invocation.output["files"]] == ["a.txt", "nested/b.txt"]


def test_write_file_requires_write_access_and_approval(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "result.txt"
    access_path = tmp_path / "tool-access.json"
    _write_access(access_path, write=[{"path": str(allowed), "recursive": True}])
    monkeypatch.setattr(settings, "tool_access_path", access_path, raising=False)
    safety = SafetyReviewService(
        approval_service=ApprovalService(callback=lambda request: ApprovalDecision(scope="once")),
        interactive_approval=True,
    )
    registry = ToolRegistry(EmptyRagService(), safety=safety)

    invocation = registry.invoke(_state(), "write_file", path=str(target), content="approved write")

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["path"] == str(target.resolve())
    assert target.read_text(encoding="utf-8") == "approved write"


def test_write_file_blocks_without_interactive_approval(monkeypatch, tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    target = allowed / "result.txt"
    access_path = tmp_path / "tool-access.json"
    _write_access(access_path, write=[{"path": str(allowed), "recursive": True}])
    monkeypatch.setattr(settings, "tool_access_path", access_path, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "write_file", path=str(target), content="blocked write")

    assert invocation.status == "BLOCKED"
    assert invocation.output["safety"]["findings"][0]["code"] == "TOOL_OPERATION_REQUIRES_APPROVAL"
    assert not target.exists()
