from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import RLock
import threading
import time
from typing import Any, Callable, Mapping

from prompt_toolkit.application import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, HTML, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from finpilot.agent.runtime_events import AgentRuntimeEvent
from finpilot.context.compression import ContextLifecycleEvent
from finpilot.models import AgentChatResponse

THINKING_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def format_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".") + "m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(value)


@dataclass
class GlobalContextViewState:
    policy: str = "global_default"
    status: str = "waiting"
    used_tokens: int = 0
    effective_input_budget: int = 0
    trigger_tokens: int = 0
    before_tokens: int = 0
    compressed: bool = False
    within_budget: bool = True
    token_counter: str = "heuristic"
    status_started_at: float = field(default_factory=time.perf_counter)

    @property
    def percent(self) -> float:
        if self.effective_input_budget <= 0:
            return 0.0
        return min(1.0, max(0.0, self.used_tokens / self.effective_input_budget))

    @property
    def display_usage(self) -> str:
        if self.effective_input_budget <= 0:
            return "waiting"
        return f"~{format_tokens(self.used_tokens)} / {format_tokens(self.effective_input_budget)}"

    def apply_event(self, event: ContextLifecycleEvent) -> None:
        if event.stage != "global" or event.policy != "global_default":
            return
        self.used_tokens = event.estimated_tokens
        self.effective_input_budget = event.effective_input_budget
        self.trigger_tokens = event.trigger_tokens
        next_status = {
            "build_started": "thinking",
            "compression_started": "compressing",
            "build_failed": "failed",
        }.get(event.kind)
        if next_status:
            self.status = next_status
            self.status_started_at = time.perf_counter()

    def apply_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self.policy = str(snapshot.get("effective_policy", "global_default"))
        self.used_tokens = int(snapshot["estimated_tokens_after"])
        self.before_tokens = int(snapshot["estimated_tokens_before"])
        self.effective_input_budget = int(snapshot["effective_input_budget"])
        self.trigger_tokens = int(snapshot["trigger_tokens"])
        self.compressed = bool(snapshot["compressed"])
        self.within_budget = bool(snapshot["within_budget"])
        self.token_counter = str(snapshot.get("token_counter", "heuristic"))
        if not self.within_budget:
            self.status = "over budget"
        elif self.compressed:
            self.status = "compressed"
        elif self.trigger_tokens and self.used_tokens >= self.trigger_tokens:
            self.status = "near limit"
        else:
            self.status = "ready"
        self.status_started_at = time.perf_counter()


class GlobalContextSnapshotCache:
    def __init__(self) -> None:
        self._snapshots: dict[tuple[str, str], dict[str, Any]] = {}
        self._lock = RLock()

    def get(self, user_id: str, chat_id: str) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._snapshots.get((user_id, chat_id))
            return deepcopy(snapshot) if snapshot is not None else None

    def put(self, user_id: str, chat_id: str, snapshot: Mapping[str, Any]) -> None:
        with self._lock:
            self._snapshots[(user_id, chat_id)] = deepcopy(dict(snapshot))

    def clear_chat(self, user_id: str, chat_id: str) -> None:
        with self._lock:
            self._snapshots.pop((user_id, chat_id), None)


@dataclass(frozen=True)
class _CommandCompleted:
    result: Any
    rendered: str


@dataclass(frozen=True)
class _RuntimeEventReceived:
    event: AgentRuntimeEvent
    occurred_at: float


@dataclass
class RuntimeProgressViewState:
    stage: str = "planner"
    plan: dict[str, Any] = field(default_factory=dict)
    planner_started_at: float = field(default_factory=time.perf_counter)
    planner_elapsed: float | None = None
    stage_started_at: float = field(default_factory=time.perf_counter)
    node_statuses: dict[str, str] = field(default_factory=dict)
    node_details: dict[str, dict[str, Any]] = field(default_factory=dict)
    node_started_at: dict[str, float] = field(default_factory=dict)
    node_elapsed: dict[str, float] = field(default_factory=dict)
    terminal_nodes: set[str] = field(default_factory=set)

    def apply(self, event: AgentRuntimeEvent, occurred_at: float) -> None:
        if event.kind == "PLANNER_STARTED":
            self.stage = "planner"
            self.planner_started_at = occurred_at
            self.stage_started_at = occurred_at
        elif event.kind == "PLAN_READY":
            self.plan = dict(event.plan)
            self.planner_elapsed = occurred_at - self.planner_started_at
            self.stage = "execute"
            self.stage_started_at = occurred_at
            for node in self.plan.get("nodes", []):
                node_id = str(node["node_id"])
                self.node_details.setdefault(node_id, dict(node))
                self.node_statuses.setdefault(node_id, "PENDING")
        elif event.kind == "PLAN_FAILED":
            self.stage = "planner_failed"
            self.stage_started_at = occurred_at
        elif event.kind == "NODE_STARTED" and event.node_id:
            if event.node_id in self.terminal_nodes:
                return
            self.stage = "execute"
            self.node_details.setdefault(
                event.node_id,
                {"node_id": event.node_id, "agent_name": event.agent_name, "task": event.task, "depends_on": []},
            )
            self.node_statuses[event.node_id] = "RUNNING"
            self.node_started_at.setdefault(event.node_id, occurred_at)
        elif event.kind == "NODE_FINISHED" and event.node_id:
            if event.node_id in self.terminal_nodes:
                return
            self.node_details.setdefault(
                event.node_id,
                {"node_id": event.node_id, "agent_name": event.agent_name, "task": event.task, "depends_on": []},
            )
            status = event.node_status or "FAILED"
            self.node_statuses[event.node_id] = status
            self.terminal_nodes.add(event.node_id)
            if event.node_id in self.node_started_at:
                self.node_elapsed[event.node_id] = occurred_at - self.node_started_at[event.node_id]
        elif event.kind == "ANSWER_STARTED":
            self.stage = "answer"
            self.stage_started_at = occurred_at
        elif event.kind == "WORKFLOW_FINISHED":
            self.stage = "finished"

    def elapsed(self, now: float) -> float:
        return max(0.0, now - self.stage_started_at)


def global_context_snapshot(response: AgentChatResponse) -> dict[str, Any] | None:
    debug = response.plan_debug or response.route_debug or {}
    usage = debug.get("context_usage")
    if not isinstance(usage, dict):
        return None
    snapshot = usage.get("global")
    return deepcopy(snapshot) if isinstance(snapshot, dict) else None


def render_response_text(response: AgentChatResponse, *, debug: bool) -> str:
    lines = ["FinPilot", response.answer or "(empty answer)"]
    if response.issues:
        lines.extend(["", "Issues"])
        lines.extend(f"- {issue.code}: {issue.message}" for issue in response.issues)
    if response.evidence:
        lines.extend(["", "Evidence"])
        lines.extend(f"- {item.tool_name}: {item.source}" for item in response.evidence)
    if response.tool_calls:
        lines.extend(["", "Tools"])
        lines.extend(f"- {tool.tool_name}: {tool.status}" for tool in response.tool_calls)
    if debug:
        lines.extend(["", json.dumps({"plan_debug": response.plan_debug}, ensure_ascii=False, default=str)])
        lines.append(json.dumps({"retrieval_debug": response.retrieval_debug}, ensure_ascii=False, default=str))
    return "\n".join(lines)


def format_global_context_fragments(state: GlobalContextViewState, width: int) -> StyleAndTextTuples:
    if state.status in {"failed", "over budget"}:
        style = "class:context.error"
    elif state.status in {"near limit", "compressing"}:
        style = "class:context.warning"
    else:
        style = "class:context.ready"
    cells = max(8, min(28, width // 5))
    filled = round(cells * state.percent)
    bar = "█" * filled + "░" * (cells - filled)
    status = state.status
    if state.status == "compressing":
        status = f"compressing · {time.perf_counter() - state.status_started_at:.1f}s"
    return [
        ("class:context.label", "global context  "),
        (style, bar),
        ("class:context.value", f"  {state.display_usage}  {state.percent:.0%}  "),
        (style, status),
    ]


class FinPilotChatApplication:
    def __init__(
        self,
        *,
        user_id: str,
        chat_id: str,
        debug: bool,
        service_factory: Callable[..., Any],
        command_handler: Callable[[str, str, str, bool, dict[str, object] | None], tuple[Any, str]] | None = None,
        history: History | None = None,
        snapshot_cache: GlobalContextSnapshotCache | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> None:
        self.user_id = user_id
        self.chat_id = chat_id
        self.debug = debug
        self.service_factory = service_factory
        self.command_handler = command_handler
        self.snapshot_cache = snapshot_cache or GlobalContextSnapshotCache()
        self.context_state = GlobalContextViewState()
        self.runtime_state = RuntimeProgressViewState()
        self.busy = False
        self.command_busy = False
        self.output_text = ""
        self._history_fragments: StyleAndTextTuples = []
        self._events: Queue[object] = Queue()
        self._worker: threading.Thread | None = None
        self._command_worker: threading.Thread | None = None
        self._active_command = ""
        self._thinking_started_at = 0.0
        self._runtime_summary_persisted = False
        self._last_output_cursor = Point(x=0, y=0)

        self.output_control = FormattedTextControl(
            self._output_fragments,
            get_cursor_position=self._output_cursor_position,
            focusable=False,
        )
        self.output_window = Window(
            self.output_control,
            wrap_lines=True,
            always_hide_cursor=True,
        )
        self.input_area = TextArea(
            multiline=False,
            history=history or InMemoryHistory(),
            prompt=HTML("<input.user>You</input.user> <input.symbol>›</input.symbol> "),
            accept_handler=self._accept_input,
            read_only=Condition(lambda: self.busy or self.command_busy),
            height=1,
        )
        self.context_control = FormattedTextControl(self._context_fragments)
        self.session_control = FormattedTextControl(self._session_fragments)
        self.container = HSplit(
            [
                self.output_window,
                Frame(self.input_area, title=HTML("<input.title>Input</input.title>")),
                Window(self.context_control, height=Dimension(min=1, max=2), wrap_lines=True),
                Window(self.session_control, height=Dimension(min=1, max=2), wrap_lines=True),
            ]
        )
        bindings = KeyBindings()

        @bindings.add("c-c")
        @bindings.add("c-d")
        def exit_chat(event) -> None:
            if self.busy:
                self._append_panel("System", "request in progress", role="system")
                self.application.invalidate()
            else:
                event.app.exit()

        self.application: Application[None] = Application(
            layout=Layout(self.container, focused_element=self.input_area),
            style=Style.from_dict(
                {
                    "frame.border": "ansiyellow bold",
                    "frame.label": "ansiyellow bold",
                    "input.user": "ansigreen bold",
                    "input.symbol": "ansiwhite bold",
                    "input.title": "ansiyellow bold",
                    "context.label": "ansicyan bold",
                    "context.value": "ansiwhite",
                    "context.ready": "ansicyan bold",
                    "context.warning": "ansiyellow bold",
                    "context.error": "ansired bold",
                    "session.label": "ansicyan bold",
                    "session.value": "ansiwhite",
                    "user.border": "ansigreen bold",
                    "user.text": "ansiwhite",
                    "assistant.border": "ansicyan bold",
                    "assistant.text": "ansiwhite",
                    "system.border": "ansiyellow bold",
                    "error.border": "ansired bold",
                    "thinking.spinner": "ansicyan bold",
                    "thinking.label": "ansicyan bold",
                    "thinking.elapsed": "ansiyellow bold",
                }
            ),
            full_screen=False,
            refresh_interval=0.1,
            input=input,
            output=output,
            key_bindings=bindings,
        )

    def run(self) -> None:
        self.application.run()

    def _accept_input(self, buffer) -> bool:
        text = buffer.text.strip()
        buffer.text = ""
        self.submit(text)
        return False

    def submit(self, text: str) -> None:
        if not text.strip():
            return
        if text.lstrip().startswith("/") and self.command_handler is not None:
            self._handle_command(text.strip())
            return
        if self.busy:
            return
        self.busy = True
        self._append_panel("You", text.strip(), role="user")
        self._thinking_started_at = time.perf_counter()
        self.runtime_state = RuntimeProgressViewState(
            planner_started_at=self._thinking_started_at,
            stage_started_at=self._thinking_started_at,
        )
        self._runtime_summary_persisted = False
        self._worker = threading.Thread(target=self._run_request, args=(text.strip(),), daemon=True)
        self._worker.start()
        self.application.invalidate()

    def _handle_command(self, raw: str) -> None:
        if raw.lower() == "/clear":
            self.output_text = ""
            self._history_fragments.clear()
            self.application.invalidate()
            return
        if raw.lower() in {"/exit", "/quit"} and self.busy:
            self._append_panel("System", "request in progress", role="system")
            self.application.invalidate()
            return
        if self.busy and raw.split(maxsplit=1)[0].lower() not in {"/status", "/context"}:
            self._append_panel("System", "request in progress", role="system")
            self.application.invalidate()
            return
        if self.command_busy:
            self._append_panel("System", "command in progress", role="system")
            self.application.invalidate()
            return
        runtime_global = self.snapshot_cache.get(self.user_id, self.chat_id)
        self.command_busy = True
        self._active_command = raw.split(maxsplit=1)[0]
        self._command_worker = threading.Thread(
            target=self._run_command,
            args=(raw, runtime_global),
            daemon=True,
        )
        self._command_worker.start()
        self.application.invalidate()

    def _run_command(self, raw: str, runtime_global: dict[str, object] | None) -> None:
        try:
            result, rendered = self.command_handler(
                raw,
                self.user_id,
                self.chat_id,
                self.debug,
                runtime_global,
            )
            self._post_event(_CommandCompleted(result=result, rendered=rendered))
        except Exception as exc:
            self._post_event(("command_error", str(exc)))

    def _apply_command_result(self, completed: _CommandCompleted) -> None:
        result = completed.result
        rendered = completed.rendered
        if rendered:
            parsed = list(to_formatted_text(ANSI(rendered.rstrip())))
            self._history_fragments.extend([("", "\n"), *parsed, ("", "\n")])
            self.output_text += "\n" + "".join(text for _, text in parsed) + "\n"
        if getattr(result, "chat_id", self.chat_id) != self.chat_id:
            self.switch_chat(result.chat_id)
        self.debug = bool(getattr(result, "debug", self.debug))
        self.command_busy = False
        self._active_command = ""
        if getattr(result, "exit_requested", False):
            self.application.exit()
        else:
            self.application.invalidate()

    def _run_request(self, text: str) -> None:
        service = None
        try:
            service = self.service_factory(
                context_event_callback=self._post_event,
                runtime_event_callback=self._post_runtime_event,
            )
            response = service.chat(self.user_id, self.chat_id, text)
            self._post_event(("response", response))
        except Exception as exc:
            self._post_event(("error", str(exc)))
        finally:
            if service is not None:
                shutdown = getattr(service, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            self._post_event(("complete", None))

    def _post_event(self, event: object) -> None:
        self._events.put(event)
        self.application.invalidate()

    def _post_runtime_event(self, event: AgentRuntimeEvent) -> None:
        self._post_event(_RuntimeEventReceived(event=event, occurred_at=time.perf_counter()))

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            if isinstance(event, ContextLifecycleEvent):
                self.handle_context_event(event)
                continue
            if isinstance(event, _RuntimeEventReceived):
                self.runtime_state.apply(event.event, event.occurred_at)
                continue
            if isinstance(event, _CommandCompleted):
                self._apply_command_result(event)
                continue
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            kind, payload = event
            if kind == "response" and isinstance(payload, AgentChatResponse):
                self._persist_runtime_summary()
                snapshot = global_context_snapshot(payload)
                if snapshot is not None:
                    self.context_state.apply_snapshot(snapshot)
                    self.snapshot_cache.put(self.user_id, self.chat_id, snapshot)
                self._append_panel(
                    "FinPilot",
                    render_response_text(payload, debug=self.debug).removeprefix("FinPilot\n"),
                    role="assistant",
                )
            elif kind == "error":
                self._persist_runtime_summary()
                self.context_state.status = "failed"
                self._append_panel("FinPilot error", str(payload), role="error")
            elif kind == "command_error":
                self.command_busy = False
                self._active_command = ""
                self._append_panel("Command error", str(payload), role="error")
            elif kind == "complete":
                self.busy = False

    def wait_for_worker(self, timeout: float | None = None) -> None:
        if self._worker is not None:
            self._worker.join(timeout)
        self._drain_events()

    def wait_for_command(self, timeout: float | None = None) -> None:
        if self._command_worker is not None:
            self._command_worker.join(timeout)
        self._drain_events()

    def handle_context_event(self, event: ContextLifecycleEvent) -> None:
        self.context_state.apply_event(event)
        self.application.invalidate()

    def handle_runtime_event(self, event: AgentRuntimeEvent) -> None:
        self.runtime_state.apply(event, time.perf_counter())
        self.application.invalidate()

    def switch_chat(self, chat_id: str) -> None:
        self.chat_id = chat_id
        self.context_state = GlobalContextViewState()
        snapshot = self.snapshot_cache.get(self.user_id, chat_id)
        if snapshot is not None:
            self.context_state.apply_snapshot(snapshot)
        self.application.invalidate()

    def status_fragments(self, *, width: int) -> StyleAndTextTuples:
        fragments = format_global_context_fragments(self.context_state, width)
        fragments.append(("", "\n"))
        fragments.extend(self._session_fragments())
        return fragments

    def approval_proxy(self, callback: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def proxy(request: Any) -> Any:
            loop = self.application.loop
            if loop is None or loop.is_closed():
                return callback(request)

            async def prompt() -> Any:
                return await run_in_terminal(lambda: callback(request), in_executor=True)

            return asyncio.run_coroutine_threadsafe(prompt(), loop).result()

        return proxy

    def thinking_text(self) -> str:
        now = time.perf_counter()
        if self.context_state.status == "compressing":
            return f"FinPilot is compressing context · 已用时 {now - self.context_state.status_started_at:.1f}s"
        labels = self._active_runtime_labels(now)
        return "\n".join(labels)

    def _output_fragments(self) -> StyleAndTextTuples:
        self._drain_events()
        fragments = list(self._history_fragments)
        if self.busy:
            fragments.extend(self._runtime_progress_fragments())
        if self.command_busy:
            fragments.extend(
                [
                    ("", "\n"),
                    ("class:thinking.spinner", "⠋ "),
                    ("class:system.border", f"Running {self._active_command}..."),
                    ("", "\n"),
                ]
            )
        # Prompt Toolkit 会分别请求文本与光标；两者必须引用同一次渲染快照。
        self._last_output_cursor = Point(x=0, y=sum(text.count("\n") for _, text in fragments))
        return fragments

    def _runtime_progress_fragments(self) -> StyleAndTextTuples:
        now = time.perf_counter()
        elapsed = max(0.0, now - self._thinking_started_at)
        frame = THINKING_FRAMES[int(elapsed * 10) % len(THINKING_FRAMES)]
        fragments: StyleAndTextTuples = [("", "\n")]
        if self.runtime_state.plan:
            for line in self._runtime_plan_lines():
                fragments.append(("class:context.value", f"{line}\n"))
        labels = (
            [f"FinPilot is compressing context · 已用时 {now - self.context_state.status_started_at:.1f}s"]
            if self.context_state.status == "compressing"
            else self._active_runtime_labels(now)
        )
        for label in labels:
            operation, separator, elapsed_text = label.partition(" · ")
            fragments.extend(
                [
                    ("class:thinking.spinner", f"{frame} "),
                    ("class:thinking.label", operation),
                    ("class:thinking.elapsed", f"  {elapsed_text}" if separator else ""),
                    ("", "\n"),
                ]
            )
        return fragments

    def _active_runtime_labels(self, now: float) -> list[str]:
        state = self.runtime_state
        if state.stage == "planner":
            return [f"Planner is generating plan... · 已用时 {state.elapsed(now):.1f}s"]
        if state.stage == "planner_failed":
            return ["Planner failed to generate plan"]
        if state.stage == "answer":
            return [f"FinPilot is generating final answer... · 已用时 {state.elapsed(now):.1f}s"]
        running = [node_id for node_id, status in state.node_statuses.items() if status == "RUNNING"]
        if running:
            return [
                f"{state.node_details[node_id].get('agent_name', node_id)} is executing... · 已用时 "
                f"{max(0.0, now - state.node_started_at.get(node_id, now)):.1f}s"
                for node_id in running
            ]
        if state.stage == "execute":
            return [f"FinPilot is scheduling DAG tasks... · 已用时 {state.elapsed(now):.1f}s"]
        return []

    def _runtime_plan_lines(self) -> list[str]:
        lines = ["Execution Plan", "Node | Agent | Task | Depends On | Status | Duration"]
        for node in self.runtime_state.plan.get("nodes", []):
            node_id = str(node["node_id"])
            status = self.runtime_state.node_statuses.get(node_id, "PENDING")
            if status == "RUNNING" and node_id in self.runtime_state.node_started_at:
                duration = time.perf_counter() - self.runtime_state.node_started_at[node_id]
                duration_text = f"{duration:.1f}s"
            elif node_id in self.runtime_state.node_elapsed:
                duration_text = f"{self.runtime_state.node_elapsed[node_id]:.1f}s"
            else:
                duration_text = "—"
            lines.append(
                " | ".join(
                    [
                        node_id,
                        str(node.get("agent_name", "")),
                        str(node.get("task", "")),
                        ", ".join(node.get("depends_on", [])) or "—",
                        status,
                        duration_text,
                    ]
                )
            )
        return lines

    def _persist_runtime_summary(self) -> None:
        if self._runtime_summary_persisted or not self.runtime_state.plan:
            return
        lines = self._runtime_plan_lines()[1:]
        if self.runtime_state.planner_elapsed is not None:
            lines.append(f"Planner generated plan · {self.runtime_state.planner_elapsed:.1f}s")
        self._append_panel("Execution Plan", "\n".join(lines), role="system")
        self._runtime_summary_persisted = True

    def _output_cursor_position(self) -> Point:
        return self._last_output_cursor

    def _append_panel(self, title: str, content: str, *, role: str) -> None:
        border_style = f"class:{role}.border"
        text_style = f"class:{role}.text" if role in {"user", "assistant"} else "class:context.value"
        fragments: StyleAndTextTuples = [(border_style, f"\n╭─ {title}\n")]
        for line in (content.splitlines() or [""]):
            fragments.extend([(border_style, "│ "), (text_style, line), ("", "\n")])
        fragments.append((border_style, "╰─\n"))
        self._history_fragments.extend(fragments)
        self.output_text += f"\n{title}\n{content}\n"

    def _context_fragments(self) -> StyleAndTextTuples:
        self._drain_events()
        width = self.application.output.get_size().columns if self.application.output else 100
        return format_global_context_fragments(self.context_state, width)

    def _session_fragments(self) -> StyleAndTextTuples:
        return [
            ("class:session.label", "session  "),
            ("class:session.value", f"user={self.user_id}  │  chat={self.chat_id}  │  debug={'on' if self.debug else 'off'}"),
        ]
