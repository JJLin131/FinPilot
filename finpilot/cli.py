from __future__ import annotations

import json
import logging
import contextlib
import io
import uuid
import warnings
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
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
from finpilot.models import AgentChatResponse
from finpilot.mysql import connect_runtime_mysql
from finpilot.responses import prepare_chat_response

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
PACKAGE_NAME = "finpilot"
THINKING_TEXT = "[bold cyan]FinPilot is thinking[/] [dim]routing / retrieving / composing[/]"
ServiceFactory = Callable[[], "FinPilotService"]

console = Console()
app = typer.Typer(
    name="finpilot",
    help="FinPilot command line assistant.",
    no_args_is_help=True,
    rich_markup_mode=None,
)
knowledge_app = typer.Typer(help="Manage shared FinPilot knowledge resources.", rich_markup_mode=None)
app.add_typer(knowledge_app, name="knowledge")


def _default_service_factory() -> "FinPilotService":
    _configure_cli_runtime()
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from finpilot.agent.service import FinPilotService

        return FinPilotService()


service_factory: ServiceFactory = _default_service_factory


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
    debug: bool = typer.Option(False, "--debug", help="Show route, retrieval, and tool details."),
) -> None:
    resolved_user_id = user_id or settings.finpilot_default_user_id
    current_chat_id = chat_id or _new_chat_id()
    debug_enabled = debug
    history = FileHistory(str(_history_path()))

    _render_splash(resolved_user_id, current_chat_id, debug_enabled)
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
        )
        _render_response(response, debug=debug_enabled)


@app.command()
def doctor() -> None:
    table = Table(title="FinPilot doctor", box=box.ASCII)
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    _add_check(
        table,
        "DeepSeek API key",
        bool(settings.deepseek_api_key),
        "Configured" if settings.deepseek_api_key else "Set DEEPSEEK_API_KEY in .env.",
    )
    _add_mysql_check(table)
    _add_http_check(table, "Ollama", settings.ollama_base_url, ["/api/tags", "/"])
    _add_http_check(table, "Embedding service", settings.embedding_base_url, ["/api/tags", "/"])
    _add_http_check(table, "Chroma", settings.chroma_base_url, ["/api/v1/heartbeat", "/api/v2/heartbeat", "/"])
    _add_http_check(table, "Reranker", settings.reranker_base_url, ["/healthz", "/"])
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

    table = Table(title="Knowledge bootstrap", box=box.ASCII)
    table.add_column("Document")
    table.add_column("Domain")
    table.add_column("Status")
    table.add_column("Chunks", justify="right")
    for result in results:
        table.add_row(result.document_id, result.domain, result.status, str(result.chunk_count))
    console.print(table)


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
        chat_id = argument or _new_chat_id()
        _render_system_notice("New chat", f"chat_id={chat_id}")
    elif command == "/debug":
        if argument.lower() in {"on", "true", "1"}:
            debug = True
        elif argument.lower() in {"off", "false", "0"}:
            debug = False
        else:
            debug = not debug
        _render_system_notice("Debug", "on" if debug else "off")
    elif command == "/context":
        _render_context(user_id, chat_id, debug)
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
) -> AgentChatResponse:
    service: FinPilotService | None = None
    try:
        service = _create_service()
        if quiet:
            response = service.chat(user_id, chat_id, content)
        else:
            with console.status(status_message, spinner="dots", spinner_style="cyan"):
                response = service.chat(user_id, chat_id, content)
    except Exception as exc:
        _fail(f"Chat failed: {exc}")
    finally:
        if service is not None:
            service.shutdown()
    return prepare_chat_response(response, debug_enabled=debug)


def _create_service() -> FinPilotService:
    return service_factory()


def _render_splash(user_id: str, chat_id: str, debug: bool) -> None:
    art = Text(
        "\n".join(
            [
                " ______ _       ____  _ _       _",
                "|  ____(_)     |  _ \\(_) |     | |",
                "| |__   _ _ __ | |_) |_| | ___ | |_",
                "|  __| | | '_ \\|  ___/ | |/ _ \\| __|",
                "| |    | | | | | |   | | | (_) | |_",
                "|_|    |_|_| |_|_|   |_|_|\\___/ \\__|",
            ]
        ),
        style="bold cyan",
        justify="center",
    )
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", justify="right")
    grid.add_column()
    grid.add_row("author", f"[white]{AUTHOR}[/]")
    grid.add_row("version", f"[white]{_app_version()}[/]")
    for label, value in _session_status_items(user_id, chat_id, debug):
        grid.add_row(label, f"[white]{value}[/]")
    grid.add_row("knowledge", "[white]shared finance knowledge[/]")
    grid.add_row("memory", "[white]user scoped[/]")
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
            border_style="bright_yellow",
            box=box.ROUNDED,
            width=_panel_width(88),
            expand=False,
        )
    )


def _render_context(user_id: str, chat_id: str, debug: bool) -> None:
    table = Table(title="Runtime context", box=box.ROUNDED, border_style="bright_yellow", width=_panel_width(92))
    table.add_column("Setting", style="bright_yellow")
    table.add_column("Value", style="white")
    for label, value in _session_status_items(user_id, chat_id, debug):
        table.add_row(label, value)
    console.print(
        Panel(
            table,
            title="[bold bright_yellow]Session[/]",
            border_style="bright_yellow",
            box=box.ROUNDED,
            width=_panel_width(96),
            expand=False,
        )
    )


def _render_system_notice(title: str, message: str) -> None:
    console.print(
        Panel(
            message,
            title=f"[bold bright_yellow]{title}[/]",
            border_style="bright_yellow",
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
            border_style="bright_green",
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
            border_style="bright_cyan",
            style="white",
            width=_panel_width(),
            expand=False,
            box=box.ROUNDED,
        )
    )
    if response.issues:
        _render_issues(response, debug=debug)
    if response.evidence:
        table = Table(title="Evidence", box=box.ROUNDED, border_style="bright_blue", width=_panel_width())
        table.add_column("Tool", style="cyan")
        table.add_column("Source", style="green")
        table.add_column("Summary", style="white")
        for item in response.evidence:
            summary = item.summary.get("document_id") or item.summary.get("title") or json.dumps(item.summary, ensure_ascii=False)
            table.add_row(item.tool_name, item.source, str(summary))
        console.print(table)
    if debug:
        _render_debug(response)


def _render_issues(response: AgentChatResponse, *, debug: bool) -> None:
    table = Table(title="Issues", box=box.ROUNDED, border_style="yellow", width=_panel_width())
    table.add_column("Code", style="yellow")
    table.add_column("Component", style="cyan")
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
    route_table = Table(title="Route", box=box.ROUNDED, border_style="magenta", width=_panel_width())
    route_table.add_column("Intent", style="cyan")
    route_table.add_column("Target", style="green")
    route_table.add_column("Confidence", style="yellow")
    route_table.add_column("Fallback", style="white")
    route_table.add_row(route.normalized_intent, route.target_agent, f"{route.confidence:.2f}", route.fallback_cause)
    console.print(route_table)
    if response.tool_calls:
        tool_table = Table(title="Tool calls", box=box.ROUNDED, border_style="magenta", width=_panel_width())
        tool_table.add_column("Step", style="dim")
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Status", style="green")
        tool_table.add_column("Duration", style="yellow")
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
    table = Table(title="Slash commands", box=box.ROUNDED, border_style="bright_blue", width=_panel_width(76) - 4)
    table.add_column("Command", style="cyan")
    table.add_column("Action", style="white")
    table.add_row("/help", "Show commands")
    table.add_row("/new [chat-id]", "Start a new conversation")
    table.add_row("/debug on|off", "Toggle debug output")
    table.add_row("/context", "Show current chat settings")
    table.add_row("/clear", "Clear terminal")
    table.add_row("/exit", "Leave chat")
    console.print(
        Panel(
            Group(table),
            title="[bold bright_blue]Help[/]",
            border_style="bright_blue",
            box=box.ROUNDED,
            width=_panel_width(76),
            expand=False,
        )
    )


def _add_mysql_check(table: Table) -> None:
    try:
        with connect_runtime_mysql() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1")
                cursor.fetchone()
        _add_check(table, "MySQL", True, f"{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    except Exception as exc:
        _add_check(table, "MySQL", False, str(exc))


def _add_http_check(table: Table, name: str, base_url: str, paths: list[str]) -> None:
    ok, detail = _probe_http(base_url, paths)
    _add_check(table, name, ok, detail)


def _probe_http(base_url: str, paths: list[str]) -> tuple[bool, str]:
    root = base_url.rstrip("/")
    last_error = "not checked"
    with httpx.Client(timeout=2) as client:
        for path in paths:
            url = root + path
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return True, f"{url} -> HTTP {response.status_code}"
                last_error = f"{url} -> HTTP {response.status_code}"
            except Exception as exc:
                last_error = f"{url} -> {exc}"
    return False, last_error


def _add_check(table: Table, name: str, ok: bool, details: str) -> None:
    table.add_row(name, "[green]ok[/]" if ok else "[red]fail[/]", details)


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
            "input.status": "ansiyellow",
            "input.text": "ansiwhite",
            "input.title": "ansiyellow bold",
            "input.user": "ansigreen bold",
            "input.symbol": "ansiwhite bold",
            "text-area.prompt": "ansigreen bold",
        }
    )


def _status_toolbar_lines(user_id: str, chat_id: str, debug: bool) -> list[str]:
    return _wrap_status_items(_session_status_items(user_id, chat_id, debug), width=_input_frame_width())


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
