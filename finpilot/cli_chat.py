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
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML, StyleAndTextTuples
from prompt_toolkit.history import History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from finpilot.context.compression import ContextLifecycleEvent
from finpilot.models import AgentChatResponse


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


def global_context_snapshot(response: AgentChatResponse) -> dict[str, Any] | None:
    debug = response.route_debug or {}
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
        lines.extend(["", json.dumps({"route_debug": response.route_debug}, ensure_ascii=False, default=str)])
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
        self.busy = False
        self.output_text = ""
        self._events: Queue[object] = Queue()
        self._worker: threading.Thread | None = None
        self._thinking_started_at = 0.0

        self.output_control = FormattedTextControl(self._output_fragments, focusable=False)
        self.output_window = Window(
            self.output_control,
            wrap_lines=True,
            always_hide_cursor=True,
            get_vertical_scroll=lambda window: self.output_text.count("\n") + (2 if self.busy else 0),
        )
        self.input_area = TextArea(
            multiline=False,
            history=history or InMemoryHistory(),
            prompt=HTML("<input.user>You</input.user> <input.symbol>›</input.symbol> "),
            accept_handler=self._accept_input,
            read_only=Condition(lambda: self.busy),
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
                self.output_text += "\nrequest in progress\n"
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
        self.output_text += f"\nYou › {text.strip()}\n"
        self._thinking_started_at = time.perf_counter()
        self._worker = threading.Thread(target=self._run_request, args=(text.strip(),), daemon=True)
        self._worker.start()
        self.application.invalidate()

    def _handle_command(self, raw: str) -> None:
        if raw.lower() == "/clear":
            self.output_text = ""
            self.application.invalidate()
            return
        if raw.lower() in {"/exit", "/quit"} and self.busy:
            self.output_text += "\nrequest in progress\n"
            self.application.invalidate()
            return
        if self.busy and raw.split(maxsplit=1)[0].lower() not in {"/status", "/context"}:
            self.output_text += "\nrequest in progress\n"
            self.application.invalidate()
            return
        runtime_global = self.snapshot_cache.get(self.user_id, self.chat_id)
        result, rendered = self.command_handler(raw, self.user_id, self.chat_id, self.debug, runtime_global)
        if rendered:
            self.output_text += f"\n{rendered.rstrip()}\n"
        if getattr(result, "chat_id", self.chat_id) != self.chat_id:
            self.switch_chat(result.chat_id)
        self.debug = bool(getattr(result, "debug", self.debug))
        if getattr(result, "exit_requested", False):
            self.application.exit()
        else:
            self.application.invalidate()

    def _run_request(self, text: str) -> None:
        service = None
        try:
            service = self.service_factory(context_event_callback=self._post_event)
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

    def _drain_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except Empty:
                return
            if isinstance(event, ContextLifecycleEvent):
                self.handle_context_event(event)
                continue
            if not isinstance(event, tuple) or len(event) != 2:
                continue
            kind, payload = event
            if kind == "response" and isinstance(payload, AgentChatResponse):
                snapshot = global_context_snapshot(payload)
                if snapshot is not None:
                    self.context_state.apply_snapshot(snapshot)
                    self.snapshot_cache.put(self.user_id, self.chat_id, snapshot)
                self.output_text += f"\n{render_response_text(payload, debug=self.debug)}\n"
            elif kind == "error":
                self.context_state.status = "failed"
                self.output_text += f"\nFinPilot error: {payload}\n"
            elif kind == "complete":
                self.busy = False

    def wait_for_worker(self, timeout: float | None = None) -> None:
        if self._worker is not None:
            self._worker.join(timeout)
        self._drain_events()

    def handle_context_event(self, event: ContextLifecycleEvent) -> None:
        self.context_state.apply_event(event)
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
        label = "FinPilot is compressing context" if self.context_state.status == "compressing" else "FinPilot is thinking"
        elapsed = max(0.0, time.perf_counter() - self._thinking_started_at)
        return f"{label} · 已思考 {elapsed:.1f}s"

    def _output_fragments(self) -> StyleAndTextTuples:
        self._drain_events()
        text = self.output_text
        if self.busy:
            text += f"\n{self.thinking_text()}\n"
        return [("class:context.value", text)]

    def _context_fragments(self) -> StyleAndTextTuples:
        self._drain_events()
        width = self.application.output.get_size().columns if self.application.output else 100
        return format_global_context_fragments(self.context_state, width)

    def _session_fragments(self) -> StyleAndTextTuples:
        return [
            ("class:session.label", "session  "),
            ("class:session.value", f"user={self.user_id}  │  chat={self.chat_id}  │  debug={'on' if self.debug else 'off'}"),
        ]
