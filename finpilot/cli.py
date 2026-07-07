from __future__ import annotations

import json
import logging
import contextlib
import io
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.shortcuts import clear as clear_terminal
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from finpilot.config import settings
from finpilot.agent.tooling.file_access import FileAccessStore
from finpilot.context.compression import DEFAULT_CONTEXT_POLICIES, estimate_tokens
from finpilot.memory.models import ChatSessionSummary, ChatTurn
from finpilot.memory.service import MAX_STORED_MESSAGES, RECENT_MESSAGE_LIMIT
from finpilot.memory.stores import AgentChatMemoryStore
from finpilot.models import AgentChatResponse
from finpilot.readiness import check_runtime_readiness
from finpilot.responses import prepare_chat_response
from finpilot.safety.approval import ApprovalDecision, ApprovalRequest, ApprovalService
from finpilot.safety.service import SafetyReviewService

if TYPE_CHECKING:
    from finpilot.agent.service import FinPilotService

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change*",
    category=Warning,
)
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s")
logging.getLogger("finpilot").setLevel(logging.ERROR)
logging.getLogger("opentelemetry").setLevel(logging.ERROR)

BRAND = "FinPilot"
AUTHOR = "JJLin131"
PROJECT_ADDRESS = "https://github.com/JJLin131/FinanceAgent"
PACKAGE_NAME = "finpilot"
THINKING_TEXT = "[bold bright_cyan]FinPilot is thinking[/] [bright_yellow]已思考 {elapsed}[/] [dim]routing -> retrieving -> composing[/]"
THINKING_REFRESH_SECONDS = 0.25
ServiceFactory = Callable[..., "FinPilotService"]
ChatStoreFactory = Callable[[], AgentChatMemoryStore]

BRAND_BORDER = "bright_yellow"
ASSISTANT_BORDER = "bright_cyan"
USER_BORDER = "bright_green"
INFO_BORDER = "bright_blue"
DEBUG_BORDER = "magenta"
_ACTIVE_THINKING_STATUSES: list["_ThinkingStatus"] = []

console = Console()
app = typer.Typer(
    name="finpilot",
    help="FinPilot command line assistant.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
knowledge_app = typer.Typer(help="Manage shared FinPilot knowledge resources.", rich_markup_mode=None)
tools_app = typer.Typer(help="Manage FinPilot agent tools.", rich_markup_mode=None)
app.add_typer(knowledge_app, name="knowledge")
app.add_typer(tools_app, name="tools")


def _default_service_factory(
    *, interactive_approval: bool = False, approval_service: ApprovalService | None = None
) -> "FinPilotService":
    _configure_cli_runtime()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from finpilot.agent.service import FinPilotService

        return FinPilotService(
            safety=_build_safety_review_service(
                interactive_approval=interactive_approval,
                approval_service=approval_service,
            )
        )


service_factory: ServiceFactory = _default_service_factory
chat_store_factory: ChatStoreFactory = AgentChatMemoryStore


def _configure_cli_runtime() -> None:
    if not settings.finpilot_cli_otel_enabled:
        settings.otel_enabled = False
        settings.otel_exporter_otlp_endpoint = None


def main() -> None:
    app()


@app.command()
def ask(
    question: list[str] = typer.Argument(..., help="Question to ask FinPilot."),
    user_id: str | None = typer.Option(None, "--user-id", "-u", help="User identity for memory isolation."),
    chat_id: str | None = typer.Option(None, "--chat-id", "-c", help="Conversation id for short-term memory."),
    debug: bool = typer.Option(False, "--debug", help="Show route, retrieval, and tool details."),
    json_output: bool = typer.Option(False, "--json", help="Print the raw response as JSON."),
) -> None:
    content = " ".join(question).strip()
    if not content:
        _fail("Question cannot be empty.")
    response = _run_chat(
        user_id=user_id or settings.finpilot_default_user_id,
        chat_id=chat_id or _new_chat_id(),
        content=content,
        debug=debug,
        status_message=THINKING_TEXT,
        quiet=json_output,
    )
    if json_output:
        typer.echo(response.model_dump_json(exclude_none=True))
        return
    _render_user_message(content)
    _render_response(response, debug=debug)


@app.command(name="chat")
def chat_command(
    user_id: str | None = typer.Option(None, "--user-id", "-u", help="User identity for memory isolation."),
    chat_id: str | None = typer.Option(None, "--chat-id", "-c", help="Conversation id for short-term memory."),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Resume an existing conversation by chat id."),
    list_sessions: bool = typer.Option(False, "--list-sessions", help="List recent conversations and exit."),
    debug: bool = typer.Option(False, "--debug", help="Show route, retrieval, and tool details."),
) -> None:
    resolved_user_id = user_id or settings.finpilot_default_user_id
    if chat_id and resume:
        _fail("Use either --chat-id or --resume, not both.")
    if list_sessions:
        _render_sessions(resolved_user_id)
        return
    current_chat_id = resume or chat_id or _new_chat_id()
    debug_enabled = debug
    history = FileHistory(str(_history_path()))
    approval_service = ApprovalService(callback=_prompt_cli_approval)

    _render_splash(resolved_user_id, current_chat_id, debug_enabled)
    if resume:
        _render_resume_notice(resolved_user_id, current_chat_id)
    _render_help()
    while True:
        try:
            raw = _prompt_user_input(history, resolved_user_id, current_chat_id, debug_enabled).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw:
            continue
        command_result = _handle_slash_command(raw, resolved_user_id, current_chat_id, debug_enabled)
        if command_result.exit_requested:
            break
        if command_result.handled:
            current_chat_id = command_result.chat_id
            debug_enabled = command_result.debug
            continue

        _render_user_message(raw)
        response = _run_chat(
            user_id=resolved_user_id,
            chat_id=current_chat_id,
            content=raw,
            debug=debug_enabled,
            status_message=THINKING_TEXT,
            interactive_approval=True,
            approval_service=approval_service,
        )
        _render_response(response, debug=debug_enabled)


@app.command()
def doctor() -> None:
    table = Table(title="FinPilot doctor", box=box.ROUNDED, border_style=INFO_BORDER)
    table.add_column("Check", style="bright_cyan")
    table.add_column("Status")
    table.add_column("Details")

    readiness = check_runtime_readiness()
    for check in readiness.checks:
        table.add_row(check.name, _status_label(check.status), check.detail)
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host."),
    port: int | None = typer.Option(None, "--port", "-p", help="Bind port."),
) -> None:
    import uvicorn

    from finpilot.api import app as fastapi_app

    uvicorn.run(fastapi_app, host=host, port=port or settings.app_port)


@knowledge_app.command("bootstrap")
def bootstrap_knowledge() -> None:
    service: FinPilotService | None = None
    try:
        service = _create_service()
        with console.status("Bootstrapping shared knowledge resources..."):
            results = service.rag_service.bootstrap_resources()
    except Exception as exc:
        _fail(f"Knowledge bootstrap failed: {exc}")
    finally:
        if service is not None:
            service.shutdown()

    table = Table(title="Knowledge bootstrap", box=box.ROUNDED, border_style=INFO_BORDER)
    table.add_column("Document", style="bright_cyan")
    table.add_column("Domain", style="bright_yellow")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    for result in results:
        table.add_row(result.document_id, result.domain, result.status, str(result.chunk_count))
    console.print(table)


@tools_app.command("allow-read")
def allow_tool_read(
    path: Path = typer.Argument(..., help="File or directory to make readable by file tools."),
    recursive: bool = typer.Option(False, "--recursive", help="Allow files below this directory recursively."),
) -> None:
    _tool_access_store().allow("read", path, recursive=recursive)
    _render_system_notice("Tool access", f"read allowed\npath={path.expanduser().resolve(strict=False)}\nrecursive={recursive}")


@tools_app.command("allow-write")
def allow_tool_write(
    path: Path = typer.Argument(..., help="File or directory to make writable by file tools."),
    recursive: bool = typer.Option(False, "--recursive", help="Allow files below this directory recursively."),
) -> None:
    _tool_access_store().allow("write", path, recursive=recursive)
    _render_system_notice("Tool access", f"write allowed\npath={path.expanduser().resolve(strict=False)}\nrecursive={recursive}")


@tools_app.command("revoke-read")
def revoke_tool_read(path: Path = typer.Argument(..., help="File or directory to remove from readable paths.")) -> None:
    removed = _tool_access_store().revoke("read", path)
    _render_system_notice("Tool access", f"read revoked={removed}\npath={path.expanduser().resolve(strict=False)}")


@tools_app.command("revoke-write")
def revoke_tool_write(path: Path = typer.Argument(..., help="File or directory to remove from writable paths.")) -> None:
    removed = _tool_access_store().revoke("write", path)
    _render_system_notice("Tool access", f"write revoked={removed}\npath={path.expanduser().resolve(strict=False)}")


@tools_app.command("access")
def render_tool_access() -> None:
    snapshot = _tool_access_store().snapshot()
    table = Table(title="Tool file access", box=box.ROUNDED, border_style=INFO_BORDER)
    table.add_column("Mode", style="bright_cyan")
    table.add_column("Path", style="white", no_wrap=True, overflow="ignore")
    table.add_column("Recursive", style="bright_yellow")
    rows = 0
    for mode in ("read", "write"):
        for rule in snapshot[mode]:
            table.add_row(mode, str(rule["path"]), str(rule["recursive"]).lower())
            rows += 1
    if rows == 0:
        table.add_row("-", "No file access rules configured.", "-")
    console.print(table)
    for mode in ("read", "write"):
        for rule in snapshot[mode]:
            typer.echo(f"{mode}: {rule['path']} recursive={str(rule['recursive']).lower()}")
    console.print_json(json.dumps(snapshot, ensure_ascii=False))


@tools_app.command("list")
def render_tools() -> None:
    from finpilot.agent.tools import ToolRegistry

    class EmptyRagService:
        def search(self, query: str, limit: int = 3):
            del query, limit
            return []

    registry = ToolRegistry(EmptyRagService())
    table = Table(title="Registered tools", box=box.ROUNDED, border_style=INFO_BORDER)
    table.add_column("Tool", style="bright_cyan")
    table.add_column("Description", style="white")
    for card in registry.list_tools():
        table.add_row(card.name, card.description)
    console.print(table)


def _tool_access_store() -> FileAccessStore:
    return FileAccessStore(Path(settings.tool_access_path))


class SlashCommandResult:
    def __init__(self, *, handled: bool, exit_requested: bool, chat_id: str, debug: bool):
        self.handled = handled
        self.exit_requested = exit_requested
        self.chat_id = chat_id
        self.debug = debug


def _handle_slash_command(raw: str, user_id: str, chat_id: str, debug: bool) -> SlashCommandResult:
    if not raw.startswith("/"):
        return SlashCommandResult(handled=False, exit_requested=False, chat_id=chat_id, debug=debug)

    parts = raw.split(maxsplit=1)
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    if command in {"/exit", "/quit"}:
        return SlashCommandResult(handled=True, exit_requested=True, chat_id=chat_id, debug=debug)
    if command == "/help":
        _render_help()
    elif command == "/new":
        previous_chat_id = chat_id
        chat_id = argument or _new_chat_id()
        _render_system_notice("New chat", f"chat_id={chat_id}\nprevious_chat_id={previous_chat_id}\nprevious chat is preserved")
    elif command == "/resume":
        if not argument:
            _render_sessions(user_id)
        else:
            chat_id = argument
            _render_resume_notice(user_id, chat_id)
    elif command in {"/sessions", "/chats"}:
        _render_sessions(user_id, limit=_parse_limit(argument, default=10))
    elif command == "/history":
        _render_chat_history(user_id, chat_id, limit=_parse_limit(argument, default=12))
    elif command == "/debug":
        if argument.lower() in {"on", "true", "1"}:
            debug = True
        elif argument.lower() in {"off", "false", "0"}:
            debug = False
        else:
            debug = not debug
        _render_system_notice("Debug", "on" if debug else "off")
    elif command in {"/status", "/context"}:
        _render_status(user_id, chat_id, debug)
    elif command == "/clear":
        clear_terminal()
    else:
        console.print(f"[yellow]Unknown command:[/] {command}")
        _render_help()
    return SlashCommandResult(handled=True, exit_requested=False, chat_id=chat_id, debug=debug)


def _run_chat(
    *,
    user_id: str,
    chat_id: str,
    content: str,
    debug: bool,
    status_message: str,
    quiet: bool = False,
    interactive_approval: bool = False,
    approval_service: ApprovalService | None = None,
) -> AgentChatResponse:
    service: FinPilotService | None = None
    try:
        service = _create_service(interactive_approval=interactive_approval, approval_service=approval_service)
        if quiet:
            response = service.chat(user_id, chat_id, content)
        else:
            with _thinking_status(status_message):
                response = service.chat(user_id, chat_id, content)
    except Exception as exc:
        _fail(f"Chat failed: {exc}")
    finally:
        if service is not None:
            service.shutdown()
    return prepare_chat_response(response, debug_enabled=debug)


def _create_service(
    *, interactive_approval: bool = False, approval_service: ApprovalService | None = None
) -> FinPilotService:
    if service_factory is _default_service_factory:
        return service_factory(interactive_approval=interactive_approval, approval_service=approval_service)
    return service_factory()


def _build_safety_review_service(
    *, interactive_approval: bool, approval_service: ApprovalService | None = None
) -> SafetyReviewService:
    approval = approval_service or ApprovalService(callback=_prompt_cli_approval if interactive_approval else None)
    return SafetyReviewService(approval_service=approval, interactive_approval=interactive_approval)


def _prompt_cli_approval(request: ApprovalRequest) -> ApprovalDecision:
    with _pause_active_thinking_status():
        table = Table(title="Safety approval required", box=box.ROUNDED, border_style=BRAND_BORDER)
        table.add_column("Field", style="bright_cyan")
        table.add_column("Value", style="white")
        table.add_row("Risk", request.finding.message)
        table.add_row("Tool", request.tool_name)
        table.add_row("Parameters", json.dumps(request.parameter_summary, ensure_ascii=False, default=str))
        console.print(table)
        raw = typer.prompt("Approve this operation? [o]nce / [s]ession / [d]eny", default="d")
    return _approval_decision_from_text(raw)


def _approval_decision_from_text(value: str) -> ApprovalDecision:
    normalized = value.strip().lower()
    if normalized in {"o", "once"}:
        return ApprovalDecision(scope="once")
    if normalized in {"s", "session"}:
        return ApprovalDecision(scope="session")
    return ApprovalDecision(scope="deny")


class _ThinkingStatus:
    def __init__(self, status_message: str) -> None:
        self.status_message = status_message
        self.started_at = 0.0
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._status_context = None
        self._status_handle = None
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.started_at = time.perf_counter()
        self._status_context = console.status(
            _thinking_status_message(self.status_message, self.started_at),
            spinner="dots",
            spinner_style="cyan",
        )
        self._status_handle = self._status_context.__enter__()
        _ACTIVE_THINKING_STATUSES.append(self)
        self._thread = threading.Thread(target=self._refresh, daemon=True)
        self._thread.start()
        return self._status_handle

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        if self in _ACTIVE_THINKING_STATUSES:
            _ACTIVE_THINKING_STATUSES.remove(self)
        if self._status_context is not None:
            return self._status_context.__exit__(exc_type, exc, tb)
        return False

    def pause(self) -> None:
        self._paused.set()
        if hasattr(self._status_handle, "stop"):
            self._status_handle.stop()

    def resume(self) -> None:
        if hasattr(self._status_handle, "start"):
            self._status_handle.start()
        self._paused.clear()
        self._update()

    def _refresh(self) -> None:
        while not self._stop.wait(THINKING_REFRESH_SECONDS):
            if not self._paused.is_set():
                self._update()

    def _update(self) -> None:
        if self._status_handle is not None:
            self._status_handle.update(_thinking_status_message(self.status_message, self.started_at))


def _thinking_status(status_message: str) -> _ThinkingStatus:
    return _ThinkingStatus(status_message)


@contextlib.contextmanager
def _pause_active_thinking_status():
    active = _ACTIVE_THINKING_STATUSES[-1] if _ACTIVE_THINKING_STATUSES else None
    if active is not None:
        active.pause()
    try:
        yield
    finally:
        if active is not None:
            active.resume()


def _thinking_status_message(status_message: str, started_at: float) -> str:
    elapsed = _format_elapsed_seconds(time.perf_counter() - started_at)
    if "{elapsed}" in status_message:
        return status_message.format(elapsed=elapsed)
    return f"{status_message} [bright_yellow]已思考 {elapsed}[/]"


def _format_elapsed_seconds(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes == 0:
        return f"{remaining_seconds}s"
    hours, remaining_minutes = divmod(minutes, 60)
    if hours == 0:
        return f"{remaining_minutes}m {remaining_seconds:02d}s"
    return f"{hours}h {remaining_minutes:02d}m {remaining_seconds:02d}s"


def _render_splash(user_id: str, chat_id: str, debug: bool) -> None:
    art = Text(
        "\n".join(
            [
                "______ _       ____  _ _       _",
                "| _____(_)     | __ \\(_) |     | |  ",
                " | |__   _ _ __ | |_) |_| | ___ | |_",
                " |  __| | | '_ \\|  ___/ | |/ _ \\| __|",
                " | |    | | | | | |   | | | (_) | |_",
                " |_|    |_|_| |_|_|   |_|_|\\___/ \\__|",
            ]
        ),
        style="bold bright_cyan",
        justify="center",
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold bright_yellow", justify="right")
    grid.add_column()
    grid.add_row("author", f"[white]{AUTHOR}[/]")
    grid.add_row("version", f"[white]{_app_version()}[/]")
    grid.add_row("projectAddress", f"[white]{PROJECT_ADDRESS}[/]")
#    for label, value in _session_status_items(user_id, chat_id, debug):
#        grid.add_row(label, f"[white]{value}[/]")
    body = Group(
        Align.center(art),
        "[bold white]FinPilot Chat[/] [dim]finance agent CLI / local interactive session[/]",
        grid,
        "[dim]Type /help for commands, /exit to leave.[/]",
    )
    console.print(
        Panel(
            body,
            title=f"[bold bright_yellow]{BRAND}[/]",
            subtitle="[dim]finance copilot shell[/]",
            border_style=BRAND_BORDER,
            box=box.ROUNDED,
            width=_panel_width(88),
            expand=False,
        )
    )


def _render_status(user_id: str, chat_id: str, debug: bool) -> None:
    snapshot = _build_status_snapshot(user_id, chat_id, debug)
    body = Group(
        _status_session_table(snapshot),
        _status_context_table(snapshot),
        _status_models_table(snapshot),
    )
    console.print(
        Panel(
            body,
            title="[bold bright_yellow]Status[/]",
            subtitle="[dim]/context alias supported[/]",
            border_style=BRAND_BORDER,
            box=box.ROUNDED,
            width=_panel_width(100),
            expand=False,
        )
    )


def _build_status_snapshot(user_id: str, chat_id: str, debug: bool) -> dict[str, object]:
    messages, load_error = _load_messages_for_status(user_id, chat_id)
    estimated_tokens = estimate_tokens([message.model_dump(mode="json") for message in messages]) if messages else 0
    policy = DEFAULT_CONTEXT_POLICIES["agent_default"]
    model_label = _model_label()
    model_window = _model_context_window_tokens(model_label)
    return {
        "user_id": user_id,
        "chat_id": chat_id,
        "memory_id": _memory_id(user_id, chat_id),
        "debug": "on" if debug else "off",
        "messages": messages,
        "message_count": len(messages),
        "visible_message_count": min(len(messages), RECENT_MESSAGE_LIMIT),
        "max_stored_messages": MAX_STORED_MESSAGES,
        "load_error": load_error,
        "estimated_tokens": estimated_tokens,
        "model_context_window": model_window,
        "model_context_window_setting": _model_context_window_setting_hint(model_label),
        "model_context_percent": (estimated_tokens / model_window) if model_window else None,
        "agent_policy": "agent_default",
        "agent_budget": policy.token_budget,
        "agent_trigger": int(policy.token_budget * policy.trigger_ratio),
        "agent_trigger_ratio": policy.trigger_ratio,
        "token_counter": "heuristic",
        "models": dict(_session_status_items(user_id, chat_id, debug)),
    }


def _status_session_table(snapshot: dict[str, object]) -> Table:
    table = Table(title="Session", box=box.SIMPLE_HEAVY, border_style=INFO_BORDER, width=_panel_width(92))
    table.add_column("Field", style="bright_cyan")
    table.add_column("Value", style="white")
    table.add_row("user", str(snapshot["user_id"]))
    table.add_row("chat", str(snapshot["chat_id"]))
    table.add_row("memory", str(snapshot["memory_id"]))
    table.add_row("debug", str(snapshot["debug"]))
    return table


def _status_context_table(snapshot: dict[str, object]) -> Table:
    estimated_tokens = int(snapshot["estimated_tokens"])
    model_window = snapshot["model_context_window"]
    load_error = snapshot["load_error"]
    table = Table(title="Context", box=box.SIMPLE_HEAVY, border_style=BRAND_BORDER, width=_panel_width(92))
    table.add_column("Metric", style="bright_yellow")
    table.add_column("Value", style="white")
    table.add_row("estimated stored", _format_tokens(estimated_tokens))
    if isinstance(model_window, int):
        table.add_row("model window", f"{_format_tokens(estimated_tokens)} / {_format_tokens(model_window)}")
        table.add_row("model usage", _format_percent(estimated_tokens / model_window))
    else:
        table.add_row("model window", f"window config missing ({snapshot['model_context_window_setting']})")
    table.add_row("stored messages", f"{snapshot['message_count']} / {snapshot['max_stored_messages']}")
    table.add_row("visible recent", f"{snapshot['visible_message_count']} / {RECENT_MESSAGE_LIMIT}")
    table.add_row("agent policy", str(snapshot["agent_policy"]))
    table.add_row(
        "agent prompt",
        f"{_format_tokens(estimated_tokens)} / {_format_tokens(int(snapshot['agent_budget']))}",
    )
    table.add_row(
        "agent trigger",
        f"{_format_tokens(int(snapshot['agent_trigger']))} ({_format_percent(float(snapshot['agent_trigger_ratio']))} of agent budget)",
    )
    table.add_row("token counter", str(snapshot["token_counter"]))
    if load_error:
        table.add_row("history load", f"[yellow]{load_error}[/]")
    return table


def _status_models_table(snapshot: dict[str, object]) -> Table:
    models = snapshot["models"]
    assert isinstance(models, dict)
    table = Table(title="Models & Retrieval", box=box.SIMPLE_HEAVY, border_style=INFO_BORDER, width=_panel_width(92))
    table.add_column("Component", style="bright_cyan")
    table.add_column("Value", style="white")
    table.add_row("query", str(models["queryModel"]))
    table.add_row("route", str(models["routeModel"]))
    table.add_row("embedding", str(models["embeddingModel"]))
    table.add_row("rewrite", str(models["rewriteModel"]))
    table.add_row("curation", str(models["curationModel"]))
    table.add_row("reranker", str(models["reranker"]))
    return table


def _load_messages_for_status(user_id: str, chat_id: str) -> tuple[list[ChatTurn], str | None]:
    try:
        return _load_chat_messages(user_id, chat_id), None
    except Exception as exc:
        return [], str(exc)


def _render_system_notice(title: str, message: str) -> None:
    console.print(
        Panel(
            message,
            title=f"[bold bright_yellow]{title}[/]",
            border_style=BRAND_BORDER,
            box=box.ROUNDED,
            width=_panel_width(84),
            expand=False,
        )
    )


def _render_user_message(content: str) -> None:
    console.print(
        Panel(
            Text(content, style="bright_green"),
            title="[bold bright_green]You[/]",
            title_align="left",
            border_style=USER_BORDER,
            box=box.ROUNDED,
            width=_panel_width(),
            expand=False,
        )
    )


def _render_response(response: AgentChatResponse, *, debug: bool) -> None:
    console.print(
        Panel(
            Markdown(response.answer or "(empty answer)"),
            title=f"[bold bright_cyan]{BRAND}[/]",
            title_align="left",
            border_style=ASSISTANT_BORDER,
            style="white",
            width=_panel_width(),
            expand=False,
            box=box.ROUNDED,
        )
    )
    if response.issues:
        _render_issues(response, debug=debug)
    if response.evidence:
        table = Table(title="Evidence", box=box.ROUNDED, border_style=INFO_BORDER, width=_panel_width())
        table.add_column("Tool", style="bright_cyan")
        table.add_column("Source", style="bright_green")
        table.add_column("Summary", style="white")
        for item in response.evidence:
            summary = item.summary.get("document_id") or item.summary.get("title") or json.dumps(item.summary, ensure_ascii=False)
            table.add_row(item.tool_name, item.source, str(summary))
        console.print(table)
    if debug:
        _render_debug(response)


def _render_issues(response: AgentChatResponse, *, debug: bool) -> None:
    table = Table(title="Issues", box=box.ROUNDED, border_style=BRAND_BORDER, width=_panel_width())
    table.add_column("Code", style="bright_yellow")
    table.add_column("Component", style="bright_cyan")
    table.add_column("Severity", style="white")
    table.add_column("Message", style="white")
    if debug:
        table.add_column("Detail", style="dim")
    for issue in response.issues:
        row = [issue.code, issue.component, issue.severity, issue.message]
        if debug:
            row.append(issue.detail or "")
        table.add_row(*row)
    console.print(table)


def _render_debug(response: AgentChatResponse) -> None:
    route = response.route
    route_table = Table(title="Route", box=box.ROUNDED, border_style=DEBUG_BORDER, width=_panel_width())
    route_table.add_column("Intent", style="bright_cyan")
    route_table.add_column("Target", style="bright_green")
    route_table.add_column("Confidence", style="bright_yellow")
    route_table.add_column("Fallback", style="white")
    route_table.add_row(route.normalized_intent, route.target_agent, f"{route.confidence:.2f}", route.fallback_cause)
    console.print(route_table)
    if response.tool_calls:
        tool_table = Table(title="Tool calls", box=box.ROUNDED, border_style=DEBUG_BORDER, width=_panel_width())
        tool_table.add_column("Step", style="dim")
        tool_table.add_column("Tool", style="bright_cyan")
        tool_table.add_column("Status", style="bright_green")
        tool_table.add_column("Duration", style="bright_yellow")
        tool_table.add_column("Summary")
        for tool in response.tool_calls:
            tool_table.add_row(
                "" if tool.step_index is None else str(tool.step_index),
                tool.tool_name,
                tool.status,
                f"{tool.duration_ms} ms",
                tool.observation_summary or "",
            )
        console.print(tool_table)
    if response.route_debug:
        console.print_json(json.dumps({"route_debug": response.route_debug}, ensure_ascii=False, default=str))
    if response.retrieval_debug:
        console.print_json(json.dumps({"retrieval_debug": response.retrieval_debug}, ensure_ascii=False, default=str))


def _render_help() -> None:
    table = Table(
        title="Slash commands",
        box=box.SIMPLE_HEAVY,
        border_style=INFO_BORDER,
        width=_panel_width(76) - 4,
    )
    table.add_column("Command", style="bright_cyan")
    table.add_column("Action", style="white")
    table.add_row("/help", "Show commands")
    table.add_row("/new [chat-id]", "Start a new conversation")
    table.add_row("/resume <chat-id>", "Resume a stored conversation")
    table.add_row("/sessions [limit]", "List recent conversations")
    table.add_row("/history [limit]", "Show current conversation memory")
    table.add_row("/debug on|off", "Toggle debug output")
    table.add_row("/status", "Show context, model, and session status")
    table.add_row("/context", "Alias for /status")
    table.add_row("/clear", "Clear terminal")
    table.add_row("/exit", "Leave chat")
    console.print(
        Panel(
            Group(table),
            title="[bold bright_blue]Help[/]",
            border_style=INFO_BORDER,
            box=box.ROUNDED,
            width=_panel_width(76),
            expand=False,
        )
    )


def _render_sessions(user_id: str, limit: int = 10) -> None:
    try:
        sessions = _list_chat_sessions(user_id, limit)
    except Exception as exc:
        _render_system_notice("Sessions", f"Unable to load sessions: {exc}")
        return
    if not sessions:
        _render_system_notice("Sessions", f"No stored conversations for user={user_id}.")
        return
    table = Table(title="Recent conversations", box=box.ROUNDED, border_style=INFO_BORDER, width=_panel_width())
    table.add_column("Chat", style="bright_cyan")
    table.add_column("Turns", justify="right", style="bright_yellow")
    table.add_column("Updated", style="white")
    table.add_column("Last user message", style="white")
    for session in sessions:
        table.add_row(
            session.chat_id,
            str(session.message_count),
            _format_timestamp(session.updated_at),
            _clip(session.last_user_message or session.last_assistant_message or "", 44),
        )
    console.print(table)


def _render_resume_notice(user_id: str, chat_id: str) -> None:
    try:
        messages = _load_chat_messages(user_id, chat_id)
    except Exception as exc:
        _render_system_notice("Resume", f"chat_id={chat_id}\nUnable to load stored messages: {exc}")
        return
    if messages:
        _render_system_notice("Resume", f"chat_id={chat_id}\nloaded_messages={len(messages)}")
    else:
        _render_system_notice("Resume", f"chat_id={chat_id}\nNo stored messages found; continuing with this chat id.")


def _render_chat_history(user_id: str, chat_id: str, limit: int = 12) -> None:
    try:
        messages = _load_chat_messages(user_id, chat_id)
    except Exception as exc:
        _render_system_notice("History", f"Unable to load history: {exc}")
        return
    if not messages:
        _render_system_notice("History", f"No stored messages for chat_id={chat_id}.")
        return
    table = Table(title=f"History: {chat_id}", box=box.ROUNDED, border_style=INFO_BORDER, width=_panel_width())
    table.add_column("Role", style="bright_cyan")
    table.add_column("Message", style="white")
    for message in messages[-limit:]:
        table.add_row(message.role, _clip(message.content, 72))
    console.print(table)


def _list_chat_sessions(user_id: str, limit: int) -> list[ChatSessionSummary]:
    return chat_store_factory().list_sessions(user_id, limit)


def _load_chat_messages(user_id: str, chat_id: str) -> list[ChatTurn]:
    return chat_store_factory().get_messages(_memory_id(user_id, chat_id))


def _memory_id(user_id: str, chat_id: str) -> str:
    return f"chat:{user_id}:{chat_id}"


def _parse_limit(value: str, *, default: int, minimum: int = 1, maximum: int = 50) -> int:
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _render_system_notice("Limit", f"Invalid limit '{value}', using {default}.")
        return default
    return max(minimum, min(parsed, maximum))


def _format_timestamp(value) -> str:
    if value is None:
        return "-"
    return value.strftime("%Y-%m-%d %H:%M:%S") if hasattr(value, "strftime") else str(value)


def _clip(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max(0, max_chars - 1)] + "…"


def _status_label(status: str) -> str:
    if status == "ok":
        return "[green]ok[/]"
    if status == "degraded":
        return "[yellow]degraded[/]"
    return "[red]failed[/]"


def _new_chat_id() -> str:
    return f"cli-{uuid.uuid4().hex[:12]}"


def _prompt_user_input(history: FileHistory, user_id: str, chat_id: str, debug: bool) -> str:
    app_ref: dict[str, Application[str]] = {}

    def accept(buffer) -> bool:
        app_ref["app"].exit(result=buffer.text)
        return True

    input_area = TextArea(
        multiline=False,
        accept_handler=accept,
        history=history,
        wrap_lines=False,
        prompt=HTML("<input.user>You</input.user> <input.symbol>›</input.symbol> "),
        style="class:input.text",
        height=1,
    )
    status_lines = _status_toolbar_lines(user_id, chat_id, debug)
    status = Window(
        FormattedTextControl("\n".join(status_lines)),
        height=max(1, len(status_lines)),
        dont_extend_height=True,
        style="class:input.status",
    )
    container = HSplit(
        [
            Frame(
                input_area,
                title=HTML("<input.title>Input</input.title>"),
                style="class:input.frame",
                width=_input_frame_width(),
            ),
            status,
        ],
        width=_input_frame_width(),
    )
    bindings = KeyBindings()

    @bindings.add("c-c")
    def _cancel(event) -> None:
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add("c-d")
    def _eof(event) -> None:
        event.app.exit(exception=EOFError())

    application: Application[str] = Application(
        layout=Layout(container, focused_element=input_area),
        key_bindings=bindings,
        style=_prompt_style(),
        full_screen=False,
        erase_when_done=True,
    )
    app_ref["app"] = application
    return application.run()


def _prompt_style() -> Style:
    return Style.from_dict(
        {
            "frame.border": "ansiyellow bold",
            "frame.label": "ansiyellow bold",
            "input.frame": "ansiyellow",
            "input.status": "ansiwhite",
            "input.text": "ansiwhite",
            "input.title": "ansiyellow bold",
            "input.user": "ansigreen bold",
            "input.symbol": "ansiwhite bold",
            "text-area.prompt": "ansigreen bold",
        }
    )


def _status_toolbar_lines(user_id: str, chat_id: str, debug: bool) -> list[str]:
    items = dict(_session_status_items(user_id, chat_id, debug))
    width = _input_frame_width()
    groups = [
        ("session", [("user", items["user"]), ("chat", items["chat"]), ("debug", items["debug"])]),
    ]
    lines: list[str] = []
    for group_name, group_items in groups:
        wrapped = _wrap_status_items(group_items, width=max(24, width - 10))
        for index, line in enumerate(wrapped):
            prefix = f"{group_name:<8}" if index == 0 else " " * 8
            lines.append(f"{prefix} {line}")
    return lines


def _wrap_status_items(items: list[tuple[str, str]], *, width: int) -> list[str]:
    segments = [f"{label}={value}" for label, value in items]
    lines: list[str] = []
    current = ""
    for segment in segments:
        candidate = segment if not current else f"{current}  {segment}"
        if len(candidate) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = segment
    if current:
        lines.append(current)
    return lines


def _session_status_items(user_id: str, chat_id: str, debug: bool) -> list[tuple[str, str]]:
    return [
        ("user", user_id),
        ("chat", chat_id),
        ("queryModel", _model_label()),
        ("routeModel", _enabled_provider_model(settings.routing_llm_enabled, settings.routing_provider, settings.routing_model_name)),
        ("embeddingModel", _enabled_value(settings.vector_enabled, settings.embedding_model_name)),
        ("rewriteModel", _enabled_value(settings.query_rewriter_enabled, settings.query_rewriter_model_name)),
        (
            "curationModel",
            _enabled_provider_model(
                settings.rag_curation_enabled,
                settings.rag_curation_provider,
                settings.rag_curation_model_name,
            ),
        ),
        ("reranker", "on" if settings.reranker_enabled else "off"),
        ("debug", "on" if debug else "off"),
    ]


def _enabled_provider_model(enabled: bool, provider: str, model_name: str) -> str:
    return _enabled_value(enabled, f"{provider}:{model_name}")


def _enabled_value(enabled: bool, value: str) -> str:
    return value if enabled else f"off:{value}"


def _model_context_window_tokens(model_label: str) -> int | None:
    windows = settings.model_context_windows or {}
    model_name = model_label.split(":", 1)[-1]
    value = windows.get(model_label) or windows.get(model_name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _model_context_window_setting_hint(model_label: str) -> str:
    model_name = model_label.split(":", 1)[-1]
    return f"MODEL_CONTEXT_WINDOWS[{model_label} or {model_name}]"


def _format_tokens(value: int) -> str:
    if value >= 1000:
        rounded = value / 1000
        return f"{rounded:.1f}k" if value % 1000 else f"{value // 1000}k"
    return str(value)


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _panel_width(preferred: int = 96) -> int:
    terminal_width = console.width or preferred
    if terminal_width <= 40:
        return max(24, terminal_width - 2)
    return min(preferred, terminal_width - 4)


def _input_frame_width() -> int:
    terminal_width = console.width or 88
    if terminal_width <= 40:
        return max(24, terminal_width - 1)
    return terminal_width - 2


def _model_label() -> str:
    return f"{settings.ai_provider}:{settings.ai_model_name}"


def _app_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0.1.0"


def _history_path() -> Path:
    path = Path.home() / ".finpilot" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/]")
    raise typer.Exit(1)


if __name__ == "__main__":
    main()
