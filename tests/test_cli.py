from __future__ import annotations

import json

from typer.testing import CliRunner

from finpilot import cli
from finpilot.models import AgentChatResponse, AgentEvidence, RouteDecision, ToolInvocation


runner = CliRunner()


class FakeService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.shutdown_called = False

    def chat(self, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        self.calls.append((user_id, chat_id, content))
        return fake_response()

    def shutdown(self) -> None:
        self.shutdown_called = True


def fake_response() -> AgentChatResponse:
    return AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="SUCCEEDED",
        answer="工资发放需要审批。",
        evidence=[
            AgentEvidence(
                tool_name="search_finance_knowledge",
                source="manual",
                summary={"document_id": "doc-1"},
            )
        ],
        route=RouteDecision(
            raw_intent_json='{"intent":"QUERY"}',
            normalized_intent="QUERY",
            reason="finance question",
            confidence=0.91,
            valid=True,
            target_agent="QueryAgent",
            classifier_intent="QUERY",
        ),
        route_debug={"loop_count": 1},
        retrieval_debug={"retrieved_docs": [{"document_id": "doc-1"}]},
        tool_calls=[
            ToolInvocation(
                tool_name="search_finance_knowledge",
                status="SUCCEEDED",
                duration_ms=12,
                observation_summary="retrieved 1 documents",
            )
        ],
    )


def test_ask_passes_user_chat_and_question(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(cli, "service_factory", lambda: fake)

    result = runner.invoke(cli.app, ["ask", "工资发放审批规则是什么？", "--user-id", "user-1", "--chat-id", "chat-1"])

    assert result.exit_code == 0
    assert fake.calls == [("user-1", "chat-1", "工资发放审批规则是什么？")]
    assert fake.shutdown_called
    assert "工资发放需要审批" in result.output


def test_ask_json_hides_debug_by_default(monkeypatch):
    fake = FakeService()
    monkeypatch.setattr(cli, "service_factory", lambda: fake)

    result = runner.invoke(cli.app, ["ask", "hello", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["answer"] == "工资发放需要审批。"
    assert "route_debug" not in result.output
    assert "tool_calls" not in result.output


def test_chat_handles_slash_commands_and_message(monkeypatch, tmp_path):
    fake = FakeService()
    prompts = iter(["/debug on", "hello", "/new chat-2", "/exit"])

    class FakePromptSession:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def prompt(self, prompt: str) -> str:
            return next(prompts)

    monkeypatch.setattr(cli, "service_factory", lambda: fake)
    monkeypatch.setattr(cli, "PromptSession", FakePromptSession)
    monkeypatch.setattr(cli, "_history_path", lambda: tmp_path / "history")

    result = runner.invoke(cli.app, ["chat", "--user-id", "user-1", "--chat-id", "chat-1"])

    assert result.exit_code == 0
    assert fake.calls == [("user-1", "chat-1", "hello")]
    assert "Debug" in result.output
    assert "New chat" in result.output


def test_doctor_renders_checks(monkeypatch):
    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql: str) -> None:
            self.sql = sql

        def fetchone(self):
            return (1,)

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(cli, "connect_runtime_mysql", lambda: FakeConnection())
    monkeypatch.setattr(cli, "_probe_http", lambda base_url, paths: (True, f"{base_url} ok"))
    monkeypatch.setattr(cli.settings, "deepseek_api_key", "sk-test")

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "FinPilot doctor" in result.output
    assert "MySQL" in result.output
