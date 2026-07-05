from __future__ import annotations

import json
import uuid
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import clear as clear_terminal
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from finpilot.config import settings
from finpilot.models import AgentChatResponse
from finpilot.mysql import connect_runtime_mysql

if TYPE_CHECKING:
    from finpilot.agent.service import FinPilotService

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change*",
    category=Warning,
)

BRAND = "FinPilot"
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
    from finpilot.agent.service import FinPilotService

    return FinPilotService()


service_factory: ServiceFactory = _default_service_factory


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
        status_message="FinPilot is thinking...",
        quiet=json_output,
    )
    if json_output:
        typer.echo(response.model_dump_json(exclude_none=True))
        return
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
    session = PromptSession(history=FileHistory(str(_history_path())))

    console.print(Panel.fit(f"{BRAND} ready\nuser={resolved_user_id} chat={current_chat_id}", title=BRAND, box=box.ASCII))
    _render_help()
    while True:
        try:
            raw = session.prompt("finpilot> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not raw:
            continue
        command_result = _handle_slash_command(raw, current_chat_id, debug_enabled)
        if command_result.exit_requested:
            break
        if command_result.handled:
            current_chat_id = command_result.chat_id
            debug_enabled = command_result.debug
            continue

        response = _run_chat(
            user_id=resolved_user_id,
            chat_id=current_chat_id,
            content=raw,
            debug=debug_enabled,
            status_message="FinPilot is working...",
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


def _handle_slash_command(raw: str, chat_id: str, debug: bool) -> SlashCommandResult:
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
        console.print(f"[green]New chat:[/] {chat_id}")
    elif command == "/debug":
        if argument.lower() in {"on", "true", "1"}:
            debug = True
        elif argument.lower() in {"off", "false", "0"}:
            debug = False
        else:
            debug = not debug
        console.print(f"[green]Debug:[/] {'on' if debug else 'off'}")
    elif command == "/context":
        console.print(f"[cyan]chat_id[/]: {chat_id}")
        console.print(f"[cyan]debug[/]: {'on' if debug else 'off'}")
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
            with console.status(status_message):
                response = service.chat(user_id, chat_id, content)
    except Exception as exc:
        _fail(f"Chat failed: {exc}")
    finally:
        if service is not None:
            service.shutdown()
    if not debug:
        response.route_debug = None
        response.retrieval_debug = None
        response.tool_calls = None
    return response


def _create_service() -> FinPilotService:
    return service_factory()


def _render_response(response: AgentChatResponse, *, debug: bool) -> None:
    console.print(Panel(Markdown(response.answer or "(empty answer)"), title=BRAND, expand=False, box=box.ASCII))
    if response.evidence:
        table = Table(title="Evidence", box=box.ASCII)
        table.add_column("Tool")
        table.add_column("Source")
        table.add_column("Summary")
        for item in response.evidence:
            summary = item.summary.get("document_id") or item.summary.get("title") or json.dumps(item.summary, ensure_ascii=False)
            table.add_row(item.tool_name, item.source, str(summary))
        console.print(table)
    if debug:
        _render_debug(response)


def _render_debug(response: AgentChatResponse) -> None:
    route = response.route
    route_table = Table(title="Route", box=box.ASCII)
    route_table.add_column("Intent")
    route_table.add_column("Target")
    route_table.add_column("Confidence")
    route_table.add_column("Fallback")
    route_table.add_row(route.normalized_intent, route.target_agent, f"{route.confidence:.2f}", route.fallback_cause)
    console.print(route_table)
    if response.tool_calls:
        tool_table = Table(title="Tool calls", box=box.ASCII)
        tool_table.add_column("Step")
        tool_table.add_column("Tool")
        tool_table.add_column("Status")
        tool_table.add_column("Duration")
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
    table = Table(title="Commands", box=box.ASCII)
    table.add_column("Command", style="cyan")
    table.add_column("Action")
    table.add_row("/help", "Show commands")
    table.add_row("/new [chat-id]", "Start a new conversation")
    table.add_row("/debug on|off", "Toggle debug output")
    table.add_row("/context", "Show current chat settings")
    table.add_row("/clear", "Clear terminal")
    table.add_row("/exit", "Leave chat")
    console.print(table)


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


def _history_path() -> Path:
    path = Path.home() / ".finpilot" / "history"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/]")
    raise typer.Exit(1)


if __name__ == "__main__":
    main()
