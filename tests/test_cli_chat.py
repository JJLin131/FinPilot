from __future__ import annotations

import pytest
import threading
import time
from types import SimpleNamespace
from prompt_toolkit.input import DummyInput
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.utils import get_cwidth

from finpilot import cli
from finpilot.agent.runtime_events import AgentRuntimeEvent
from finpilot.cli_chat import (
    FinPilotChatApplication,
    GlobalContextSnapshotCache,
    GlobalContextViewState,
    format_global_context_fragments,
    global_context_snapshot,
    render_response_text,
)
from finpilot.context.compression import ContextLifecycleEvent
from finpilot.models import AgentChatResponse, RouteDecision


def _snapshot(*, used: int = 140_000, budget: int = 380_000, compressed: bool = True):
    return {
        "effective_policy": "global_default",
        "stage": "global",
        "estimated_tokens_before": 320_000,
        "estimated_tokens_after": used,
        "effective_input_budget": budget,
        "trigger_tokens": 323_000,
        "compressed": compressed,
        "within_budget": True,
        "token_counter": "heuristic",
    }


def test_global_context_state_tracks_compression_and_final_snapshot():
    state = GlobalContextViewState()
    state.apply_event(
        ContextLifecycleEvent(
            kind="compression_started",
            stage="global",
            policy="global_default",
            estimated_tokens=320_000,
            effective_input_budget=380_000,
            trigger_tokens=323_000,
        )
    )
    assert state.status == "compressing"
    assert state.percent == pytest.approx(320_000 / 380_000)

    state.apply_snapshot(_snapshot())

    assert state.status == "compressed"
    assert state.used_tokens == 140_000
    assert state.display_usage == "~140k / 380k"


def test_context_cache_isolated_by_user_and_chat():
    cache = GlobalContextSnapshotCache()
    cache.put("user-1", "chat-a", {"estimated_tokens_after": 100})

    assert cache.get("user-1", "chat-a") == {"estimated_tokens_after": 100}
    assert cache.get("user-1", "chat-b") is None


def test_global_context_fragments_show_bar_usage_and_state():
    state = GlobalContextViewState()
    state.apply_snapshot(_snapshot(compressed=False))

    text = "".join(fragment[1] for fragment in format_global_context_fragments(state, 100))

    assert "global context" in text
    assert "~140k / 380k" in text
    assert "37%" in text
    assert "ready" in text


def test_global_context_snapshot_reads_response_runtime_debug():
    response = AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="SUCCEEDED",
        answer="ok",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="QUERY",
            reason="test",
            confidence=1.0,
            valid=True,
            target_agent="QueryAgent",
            classifier_intent="QUERY",
        ),
        route_debug={"context_usage": {"global": _snapshot()}},
    )

    assert global_context_snapshot(response) == _snapshot()


def test_status_prefers_runtime_global_snapshot(monkeypatch):
    monkeypatch.setattr(cli, "_load_chat_messages", lambda user_id, chat_id: [])

    status = cli._build_status_snapshot("user-1", "chat-1", False, runtime_global=_snapshot())

    assert status["context_source"] == "runtime global_default"
    assert status["estimated_tokens"] == 140_000
    assert status["effective_input_budget"] == 380_000


class BlockingService:
    def __init__(self, gate: threading.Event, response: AgentChatResponse) -> None:
        self.gate = gate
        self.response = response

    def chat(self, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        self.gate.wait(timeout=1)
        return self.response

    def shutdown(self) -> None:
        return None


def _response_with_context() -> AgentChatResponse:
    return AgentChatResponse(
        request_id="req-1",
        trace_id="trace-1",
        domain="FINANCE",
        status="SUCCEEDED",
        answer="上下文分析完成。",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="QUERY",
            reason="test",
            confidence=1.0,
            valid=True,
            target_agent="QueryAgent",
            classifier_intent="QUERY",
        ),
        route_debug={"context_usage": {"global": _snapshot(compressed=False)}},
    )


def test_submit_keeps_input_and_status_present_while_worker_runs():
    gate = threading.Event()
    service = BlockingService(gate, _response_with_context())
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: service,
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.submit("hello")

    assert app.busy is True
    assert app.input_area is not None
    assert app.context_control is not None

    gate.set()
    app.wait_for_worker(timeout=1)
    assert app.busy is False
    assert app.context_state.status == "ready"
    assert "上下文分析完成" in app.output_text


def test_compression_event_updates_thinking_copy():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )
    app.handle_context_event(
        ContextLifecycleEvent(
            kind="compression_started",
            stage="global",
            policy="global_default",
            estimated_tokens=320_000,
            effective_input_budget=380_000,
            trigger_tokens=323_000,
        )
    )

    assert "FinPilot is compressing context" in app.thinking_text()


def test_runtime_events_replace_generic_thinking_with_plan_and_running_agents():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )
    app.busy = True
    app.handle_runtime_event(AgentRuntimeEvent(request_id="req-1", kind="PLANNER_STARTED"))
    app.handle_runtime_event(
        AgentRuntimeEvent(
            request_id="req-1",
            kind="PLAN_READY",
            plan={
                "status": "READY",
                "nodes": [
                    {
                        "node_id": "knowledge",
                        "agent_name": "QueryAgent",
                        "task": "查询财务规则",
                        "depends_on": [],
                    }
                ],
            },
        )
    )
    app.handle_runtime_event(
        AgentRuntimeEvent(
            request_id="req-1",
            kind="NODE_STARTED",
            node_id="knowledge",
            agent_name="QueryAgent",
            task="查询财务规则",
        )
    )

    text = "".join(fragment[1] for fragment in app._output_fragments())

    assert "Execution Plan" in text
    assert "QueryAgent is executing" in text
    assert "FinPilot is thinking" not in text


def test_persistent_chat_passes_runtime_callback_and_keeps_final_plan_summary():
    captured: dict[str, object] = {}

    class EventService:
        def chat(self, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
            callback = captured["runtime_event_callback"]
            callback(AgentRuntimeEvent(request_id="req-1", kind="PLANNER_STARTED"))
            callback(
                AgentRuntimeEvent(
                    request_id="req-1",
                    kind="PLAN_READY",
                    plan={
                        "status": "READY",
                        "nodes": [
                            {
                                "node_id": "knowledge",
                                "agent_name": "QueryAgent",
                                "task": "查询财务规则",
                                "depends_on": [],
                            }
                        ],
                    },
                )
            )
            callback(
                AgentRuntimeEvent(
                    request_id="req-1",
                    kind="NODE_FINISHED",
                    node_id="knowledge",
                    agent_name="QueryAgent",
                    task="查询财务规则",
                    node_status="SUCCEEDED",
                )
            )
            callback(AgentRuntimeEvent(request_id="req-1", kind="WORKFLOW_FINISHED"))
            return _response_with_context()

        def shutdown(self) -> None:
            return None

    def service_factory(**kwargs):
        captured.update(kwargs)
        return EventService()

    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=service_factory,
        input=DummyInput(),
        output=DummyOutput(),
    )

    app._run_request("hello")
    app._drain_events()

    assert callable(captured["runtime_event_callback"])
    assert "Execution Plan" in app.output_text
    assert "QueryAgent" in app.output_text
    assert "SUCCEEDED" in app.output_text


def test_execution_plan_keeps_status_row_within_terminal_width_for_long_task():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )
    app.handle_runtime_event(
        AgentRuntimeEvent(
            request_id="req-1",
            kind="PLAN_READY",
            plan={
                "status": "READY",
                "nodes": [
                    {
                        "node_id": "fetch_user_account",
                        "agent_name": "TreasuryDataAgent",
                        "task": "使用 query_treasury_account 查询当前用户的所有财资账户，返回第一个可用账户的账号。",
                        "depends_on": [],
                    }
                ],
            },
        )
    )
    app.handle_runtime_event(
        AgentRuntimeEvent(
            request_id="req-1",
            kind="NODE_FINISHED",
            node_id="fetch_user_account",
            agent_name="TreasuryDataAgent",
            node_status="SUCCEEDED",
        )
    )

    lines = app._runtime_plan_lines()
    status_line = next(line for line in lines if "SUCCEEDED" in line)

    assert get_cwidth(status_line) <= app.application.output.get_size().columns
    assert any("Task:" in line for line in lines)


def test_switch_chat_restores_snapshot_from_current_process_cache():
    cache = GlobalContextSnapshotCache()
    cache.put("user-1", "chat-a", _snapshot(used=100, budget=1000, compressed=False))
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-b",
        debug=False,
        service_factory=lambda **kwargs: None,
        snapshot_cache=cache,
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.switch_chat("chat-a")
    assert app.context_state.used_tokens == 100

    app.switch_chat("chat-c")
    assert app.context_state.status == "waiting"


def test_status_fragments_keep_session_fields_at_narrow_width():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )

    text = "".join(fragment[1] for fragment in app.status_fragments(width=52))

    assert "global context" in text
    assert "user=user-1" in text
    assert "chat=chat-1" in text
    assert "debug=off" in text


def test_slash_command_updates_chat_debug_and_output():
    def command_handler(raw: str, user_id: str, chat_id: str, debug: bool, runtime_global):
        return SimpleNamespace(
            handled=True,
            exit_requested=False,
            chat_id="chat-2",
            debug=True,
        ), "switched"

    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        command_handler=command_handler,
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.submit("/resume chat-2")
    app.wait_for_command(timeout=1)

    assert app.chat_id == "chat-2"
    assert app.debug is True
    assert "switched" in app.output_text


def test_approval_proxy_uses_callback_before_application_loop_starts():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )
    callback = app.approval_proxy(lambda request: f"approved:{request}")

    assert callback("request") == "approved:request"


def test_response_text_preserves_answer_and_debug_details():
    response = _response_with_context()
    response.route_debug["loop_count"] = 2

    text = render_response_text(response, debug=True)

    assert "上下文分析完成" in text
    assert "loop_count" in text


def test_output_fragments_render_user_and_thinking_with_distinct_styles():
    gate = threading.Event()
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: BlockingService(gate, _response_with_context()),
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.submit("hello")
    fragments = app._output_fragments()
    rendered = "".join(text for _, text in fragments)
    styles = {style for style, _ in fragments}
    gate.set()
    app.wait_for_worker(timeout=1)

    assert "╭─ You" in rendered
    assert "Planner is generating plan" in rendered
    assert "class:user.border" in styles
    assert "class:thinking.label" in styles
    assert "class:thinking.elapsed" in styles


def test_output_cursor_uses_same_fragment_snapshot_when_event_arrives_between_callbacks():
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        input=DummyInput(),
        output=DummyOutput(),
    )
    app.busy = True
    fragments = app._output_fragments()
    rendered_max_y = sum(text.count("\n") for _, text in fragments)

    app._post_event(("response", _response_with_context()))
    cursor = app._output_cursor_position()

    assert cursor.y <= rendered_max_y


def test_response_fragments_restore_finpilot_frame_and_color():
    gate = threading.Event()
    gate.set()
    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: BlockingService(gate, _response_with_context()),
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.submit("hello")
    app.wait_for_worker(timeout=1)
    fragments = app._output_fragments()
    rendered = "".join(text for _, text in fragments)

    assert "╭─ FinPilot" in rendered
    assert "class:assistant.border" in {style for style, _ in fragments}


def test_slash_command_ansi_is_parsed_instead_of_rendered_as_control_text():
    def command_handler(raw, user_id, chat_id, debug, runtime_global):
        return SimpleNamespace(
            handled=True,
            exit_requested=False,
            chat_id=chat_id,
            debug=debug,
        ), "\x1b[31mStatus\x1b[0m"

    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        command_handler=command_handler,
        input=DummyInput(),
        output=DummyOutput(),
    )

    app.submit("/status")
    fragments = app._output_fragments()
    rendered = "".join(text for _, text in fragments)

    assert "\x1b" not in rendered
    assert "Status" in rendered
    assert "ansired" in {style for style, _ in fragments}


def test_persistent_application_accepts_multiple_commands_and_exits():
    calls: list[str] = []

    def command_handler(raw, user_id, chat_id, debug, runtime_global):
        calls.append(raw)
        return SimpleNamespace(
            handled=True,
            exit_requested=raw == "/exit",
            chat_id=chat_id,
            debug=debug,
        ), f"handled {raw}"

    with create_pipe_input() as pipe_input:
        app = FinPilotChatApplication(
            user_id="user-1",
            chat_id="chat-1",
            debug=False,
            service_factory=lambda **kwargs: None,
            command_handler=command_handler,
            input=pipe_input,
            output=DummyOutput(),
        )
        runner = threading.Thread(target=app.run, daemon=True)
        runner.start()

        for command in ("/help", "/status"):
            pipe_input.send_text(f"{command}\n")
            deadline = time.monotonic() + 1
            while (not calls or calls[-1] != command or app.command_busy) and time.monotonic() < deadline:
                time.sleep(0.01)

        pipe_input.send_text("/exit\n")
        runner.join(timeout=1)

    assert runner.is_alive() is False
    assert calls == ["/help", "/status", "/exit"]


def test_blocking_slash_handler_does_not_block_ui_thread():
    gate = threading.Event()

    def command_handler(raw, user_id, chat_id, debug, runtime_global):
        gate.wait(timeout=1)
        return SimpleNamespace(
            handled=True,
            exit_requested=False,
            chat_id=chat_id,
            debug=debug,
        ), "status ready"

    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: None,
        command_handler=command_handler,
        input=DummyInput(),
        output=DummyOutput(),
    )
    submit_thread = threading.Thread(target=app.submit, args=("/status",), daemon=True)
    submit_thread.start()
    submit_thread.join(timeout=0.1)

    assert submit_thread.is_alive() is False
    assert app.command_busy is True

    gate.set()
    app.wait_for_command(timeout=1)
    assert app.command_busy is False
    assert "status ready" in app.output_text


def test_output_cursor_follows_latest_message_after_large_status_table():
    gate = threading.Event()

    def command_handler(raw, user_id, chat_id, debug, runtime_global):
        rendered = "\n".join(f"status row {index} " + "x" * 120 for index in range(40))
        return SimpleNamespace(
            handled=True,
            exit_requested=False,
            chat_id=chat_id,
            debug=debug,
        ), rendered

    app = FinPilotChatApplication(
        user_id="user-1",
        chat_id="chat-1",
        debug=False,
        service_factory=lambda **kwargs: BlockingService(gate, _response_with_context()),
        command_handler=command_handler,
        input=DummyInput(),
        output=DummyOutput(),
    )
    app.submit("/status")
    app.wait_for_command(timeout=1)
    app.submit("new question")

    content = app.output_control.create_content(width=48, height=12)
    gate.set()
    app.wait_for_worker(timeout=1)

    assert content.cursor_position.y == content.line_count - 1
