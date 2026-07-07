from __future__ import annotations

import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from finpilot import cli
from finpilot.memory.models import ChatSessionSummary, ChatTurn
from finpilot.models import AgentChatResponse, AgentEvidence, RouteDecision, ToolInvocation
from finpilot.safety.approval import ApprovalDecision
from finpilot.safety.models import SafetyFinding


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
    assert "You" in result.output
    assert "FinPilot" in result.output
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


def test_thinking_status_message_includes_elapsed(monkeypatch):
    ticks = iter([10.0, 72.5])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(ticks))

    started_at = cli.time.perf_counter()
    message = cli._thinking_status_message(cli.THINKING_TEXT, started_at)

    assert "FinPilot is thinking" in message
    assert "已思考 1m 02s" in message
    assert "routing -> retrieving -> composing" in message


def test_run_chat_shows_thinking_status_for_interactive_approval(monkeypatch):
    fake = FakeService()
    statuses = []

    class FakeStatus:
        def __init__(self, message: str, **kwargs) -> None:
            self.message = message
            self.kwargs = kwargs
            self.updates: list[str] = []

        def __enter__(self):
            statuses.append(self)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, message: str) -> None:
            self.updates.append(message)

    def fake_status(message: str, **kwargs):
        return FakeStatus(message, **kwargs)

    monkeypatch.setattr(cli, "service_factory", lambda: fake)
    monkeypatch.setattr(cli.console, "status", fake_status)

    response = cli._run_chat(
        user_id="user-1",
        chat_id="chat-1",
        content="hello",
        debug=False,
        status_message=cli.THINKING_TEXT,
        interactive_approval=True,
    )

    assert response.answer
    assert fake.calls == [("user-1", "chat-1", "hello")]
    assert statuses
    assert "FinPilot is thinking" in statuses[0].message
    assert "已思考" in statuses[0].message
    assert statuses[0].kwargs["spinner"] == "dots"


def test_chat_handles_slash_commands_and_message(monkeypatch, tmp_path):
    fake = FakeService()
    prompts = iter(["/debug on", "hello", "/new chat-2", "/exit"])

    def fake_prompt(*args, **kwargs) -> str:
        return next(prompts)

    monkeypatch.setattr(cli, "service_factory", lambda: fake)
    monkeypatch.setattr(cli, "_prompt_user_input", fake_prompt)
    monkeypatch.setattr(cli, "_history_path", lambda: tmp_path / "history")

    result = runner.invoke(cli.app, ["chat", "--user-id", "user-1", "--chat-id", "chat-1"])

    assert result.exit_code == 0
    assert fake.calls == [("user-1", "chat-1", "hello")]
    assert "FinPilot Chat" in result.output
    assert "Slash commands" in result.output
    assert "Debug" in result.output
    assert "New chat" in result.output


def test_cli_approval_decision_parses_once_session_and_deny():
    assert cli._approval_decision_from_text("o") == ApprovalDecision(scope="once")
    assert cli._approval_decision_from_text("once") == ApprovalDecision(scope="once")
    assert cli._approval_decision_from_text("s") == ApprovalDecision(scope="session")
    assert cli._approval_decision_from_text("session") == ApprovalDecision(scope="session")
    assert cli._approval_decision_from_text("d") == ApprovalDecision(scope="deny")
    assert cli._approval_decision_from_text("unexpected") == ApprovalDecision(scope="deny")


def test_safety_review_service_reuses_chat_session_approval_service():
    approval_service = cli.ApprovalService(callback=lambda request: cli.ApprovalDecision(scope="session"))
    first = cli._build_safety_review_service(interactive_approval=True, approval_service=approval_service)
    second = cli._build_safety_review_service(interactive_approval=True, approval_service=approval_service)
    finding = SafetyFinding(
        code="TOOL_OPERATION_REQUIRES_APPROVAL",
        reviewer="operation_risk",
        action="REQUIRE_APPROVAL",
        message="approval required",
        severity="warning",
    )

    first.approval_service.resolve(finding, tool_name="transfer_mock_funds", parameters={"amount": 1000})
    result = second.approval_service.resolve(finding, tool_name="transfer_mock_funds", parameters={"amount": 1000})

    assert result.approved is True
    assert len(approval_service.session_approvals) == 1


def test_chat_can_list_history_and_resume(monkeypatch, tmp_path):
    fake = FakeService()
    prompts = iter(["/sessions 2", "/history 1", "/resume chat-2", "hello", "/exit"])

    def fake_prompt(*args, **kwargs) -> str:
        return next(prompts)

    monkeypatch.setattr(cli, "service_factory", lambda: fake)
    monkeypatch.setattr(cli, "_prompt_user_input", fake_prompt)
    monkeypatch.setattr(cli, "_history_path", lambda: tmp_path / "history")
    monkeypatch.setattr(
        cli,
        "_list_chat_sessions",
        lambda user_id, limit: [
            ChatSessionSummary(
                chat_id="chat-2",
                memory_id="chat:user-1:chat-2",
                message_count=2,
                last_user_message="previous question",
                updated_at=datetime(2026, 7, 6, 8, 30, tzinfo=UTC),
            )
        ],
    )
    monkeypatch.setattr(
        cli,
        "_load_chat_messages",
        lambda user_id, chat_id: [
            ChatTurn(role="user", content=f"{chat_id} question"),
            ChatTurn(role="assistant", content=f"{chat_id} answer"),
        ],
    )

    result = runner.invoke(cli.app, ["chat", "--user-id", "user-1", "--chat-id", "chat-1"])

    assert result.exit_code == 0
    assert fake.calls == [("user-1", "chat-2", "hello")]
    assert "Recent conversations" in result.output
    assert "History: chat-1" in result.output
    assert "Resume" in result.output


def test_chat_status_shows_context_budget_without_guessing_model_window(monkeypatch, tmp_path):
    fake = FakeService()
    prompts = iter(["/status", "/context", "/exit"])

    def fake_prompt(*args, **kwargs) -> str:
        return next(prompts)

    monkeypatch.setattr(cli, "service_factory", lambda: fake)
    monkeypatch.setattr(cli, "_prompt_user_input", fake_prompt)
    monkeypatch.setattr(cli, "_history_path", lambda: tmp_path / "history")
    monkeypatch.setattr(
        cli,
        "_load_chat_messages",
        lambda user_id, chat_id: [
            ChatTurn(role="user", content="hello"),
            ChatTurn(role="assistant", content="answer"),
        ],
    )
    monkeypatch.setattr(cli.settings, "model_context_windows", {})

    result = runner.invoke(cli.app, ["chat", "--user-id", "user-1", "--chat-id", "chat-1"])

    assert result.exit_code == 0
    assert "Status" in result.output
    assert "Context" in result.output
    assert "model window" in result.output
    assert "window config missing" in result.output
    assert "MODEL_CONTEXT_WINDOWS" in result.output
    assert "not configured" not in result.output
    assert "agent prompt" in result.output
    assert "agent trigger" in result.output
    assert "░" not in result.output
    assert "█" not in result.output


def test_model_context_window_uses_configured_model_value(monkeypatch):
    monkeypatch.setattr(cli.settings, "model_context_windows", {"deepseek-v4-pro": 96000})

    assert cli._model_context_window_tokens("deepseek:deepseek-v4-pro") == 96000


def test_input_toolbar_omits_model_and_rag_details():
    lines = cli._status_toolbar_lines("user-1", "chat-1", debug=False)
    text = "\n".join(lines)

    assert "session" in text
    assert "user=user-1" in text
    assert "chat=chat-1" in text
    assert "debug=off" in text
    assert "models" not in text
    assert "rag" not in text
    assert "query=" not in text
    assert "rewrite=" not in text


def test_chat_list_sessions_exits_without_prompt(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_list_chat_sessions",
        lambda user_id, limit: [
            ChatSessionSummary(chat_id="chat-1", memory_id="chat:user-1:chat-1", message_count=2)
        ],
    )

    result = runner.invoke(cli.app, ["chat", "--user-id", "user-1", "--list-sessions"])

    assert result.exit_code == 0
    assert "Recent conversations" in result.output
    assert "chat-1" in result.output


def test_cli_status_uses_runtime_model_settings(monkeypatch):
    monkeypatch.setattr(cli.settings, "ai_provider", "deepseek")
    monkeypatch.setattr(cli.settings, "ai_model_name", "query-model")
    monkeypatch.setattr(cli.settings, "routing_provider", "deepseek")
    monkeypatch.setattr(cli.settings, "routing_model_name", "route-model")
    monkeypatch.setattr(cli.settings, "embedding_model_name", "embed-model")
    monkeypatch.setattr(cli.settings, "query_rewriter_model_name", "rewrite-model")
    monkeypatch.setattr(cli.settings, "rag_curation_model_name", "curation-model")

    items = dict(cli._session_status_items("user-1", "chat-1", debug=True))

    assert items["user"] == "user-1"
    assert items["chat"] == "chat-1"
    assert items["queryModel"] == "deepseek:query-model"
    assert items["routeModel"] == "deepseek:route-model"
    assert items["embeddingModel"] == "embed-model"
    assert items["rewriteModel"] == "rewrite-model"
    assert items["curationModel"] == "deepseek:curation-model"
    assert items["debug"] == "on"


def test_doctor_renders_checks(monkeypatch):
    from finpilot.readiness import ReadinessCheck, RuntimeReadiness

    monkeypatch.setattr(
        cli,
        "check_runtime_readiness",
        lambda: RuntimeReadiness(
            status="degraded",
            checks=[
                ReadinessCheck(name="mysql", status="ok", detail="select 1 succeeded"),
                ReadinessCheck(name="bm25", status="degraded", detail="index not found"),
            ],
        ),
    )

    result = runner.invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "FinPilot doctor" in result.output
    assert "mysql" in result.output
    assert "bm25" in result.output
    assert "degraded" in result.output


def test_tools_commands_manage_file_access_rules(monkeypatch, tmp_path):
    access_path = tmp_path / "tool-access.json"
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    monkeypatch.setattr(cli.settings, "tool_access_path", access_path)

    read_result = runner.invoke(cli.app, ["tools", "allow-read", str(readable), "--recursive"])
    write_result = runner.invoke(cli.app, ["tools", "allow-write", str(writable), "--recursive"])
    access_result = runner.invoke(cli.app, ["tools", "access"])
    revoke_result = runner.invoke(cli.app, ["tools", "revoke-read", str(readable)])

    assert read_result.exit_code == 0
    assert write_result.exit_code == 0
    assert access_result.exit_code == 0
    assert revoke_result.exit_code == 0
    payload = json.loads(access_path.read_text(encoding="utf-8"))
    assert payload["read"] == []
    assert payload["write"][0]["path"] == str(writable.resolve(strict=False))
    assert str(readable.resolve(strict=False)) in access_result.output
    assert str(writable.resolve(strict=False)) in access_result.output
