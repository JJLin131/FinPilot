from __future__ import annotations

from queue import Empty, Queue
import threading
import time
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.table import Table
from rich.text import Text

from finpilot.agent.runtime_events import AgentRuntimeEvent


_TERMINAL_NODE_STATUSES = {"SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED"}


class CliRuntimeProgress:
    """把后台运行事件串行化为一个可持久保留的 Rich Live 区域。"""

    def __init__(self, *, console: Console, refresh_seconds: float = 0.1) -> None:
        self.console = console
        self.refresh_seconds = refresh_seconds
        self.started_at = time.perf_counter()
        self.planner_started_at = self.started_at
        self.planner_elapsed: float | None = None
        self.current_stage = "planner"
        self.plan: dict[str, Any] = {}
        self.node_statuses: dict[str, str] = {}
        self.node_details: dict[str, dict[str, Any]] = {}
        self.node_started_at: dict[str, float] = {}
        self.node_elapsed: dict[str, float] = {}
        self._terminal_nodes: set[str] = set()
        self._events: Queue[tuple[float, AgentRuntimeEvent]] = Queue()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._live: Live | None = None
        self._live_started = False

    def __enter__(self) -> CliRuntimeProgress:
        self._live = Live(
            self.render(),
            console=self.console,
            refresh_per_second=max(1, int(1 / self.refresh_seconds)),
            transient=False,
            auto_refresh=False,
        )
        self._live.start(refresh=True)
        self._live_started = True
        self._thread = threading.Thread(target=self._refresh, name="finpilot-cli-progress", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self.process_pending()
        if self._live is not None and self._live_started:
            self._live.update(self.render(), refresh=True)
            self._live.stop()
            self._live_started = False
        return False

    def emit(self, event: AgentRuntimeEvent) -> None:
        # Agent 工作线程只负责入队，Rich Live 始终由刷新线程统一更新。
        self._events.put((time.perf_counter(), event))

    def pause(self) -> None:
        self._paused.set()
        if self._live is not None and self._live_started:
            # 审批期间临时清除 Live，最终退出时再用 transient=False 持久保留结果。
            self._live.transient = True
            self._live.stop()
            self._live.transient = False
            self._live_started = False

    def resume(self) -> None:
        if self._live is not None and not self._live_started:
            self._live.start(refresh=True)
            self._live_started = True
        self._paused.clear()
        self._update_live()

    def process_pending(self) -> int:
        processed = 0
        while True:
            try:
                occurred_at, event = self._events.get_nowait()
            except Empty:
                return processed
            self._apply_event(event, occurred_at)
            processed += 1

    def render(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self.plan:
            parts.append(self._plan_table())

        if self.planner_elapsed is not None:
            parts.append(
                Text.from_markup(f"[green]✓ Planner generated plan[/]  [dim]{_elapsed(self.planner_elapsed)}[/]")
            )

        if self.current_stage == "planner":
            parts.append(
                Text.from_markup(
                    f"[cyan]⠋ Planner is generating plan...[/]  [yellow]已用时 {_elapsed(self._stage_elapsed())}[/]"
                )
            )
        elif self.current_stage == "planner_failed":
            parts.append(Text.from_markup("[red]✗ Planner failed to generate plan[/]"))
        elif self.current_stage == "answer":
            parts.append(
                Text.from_markup(
                    f"[yellow]⠋ FinPilot is generating final answer...[/]  [yellow]已用时 {_elapsed(self._stage_elapsed())}[/]"
                )
            )
        else:
            for node_id, status in self.node_statuses.items():
                if status != "RUNNING":
                    continue
                detail = self.node_details.get(node_id, {})
                agent_name = str(detail.get("agent_name", node_id))
                elapsed = time.perf_counter() - self.node_started_at.get(node_id, time.perf_counter())
                parts.append(
                    Text.from_markup(f"[cyan]⠋ {agent_name} is executing...[/]  [yellow]已用时 {_elapsed(elapsed)}[/]")
                )

        return Group(*parts)

    def _refresh(self) -> None:
        while not self._stop.wait(self.refresh_seconds):
            self.process_pending()
            if not self._paused.is_set():
                self._update_live()

    def _update_live(self) -> None:
        if self._live is not None and self._live_started:
            self._live.update(self.render(), refresh=True)

    def _apply_event(self, event: AgentRuntimeEvent, occurred_at: float) -> None:
        now = occurred_at
        if event.kind == "PLANNER_STARTED":
            self.current_stage = "planner"
            self.planner_started_at = now
        elif event.kind == "PLAN_READY":
            self.plan = dict(event.plan)
            self.planner_elapsed = now - self.planner_started_at
            self.current_stage = "execute"
            for node in self.plan.get("nodes", []):
                node_id = str(node["node_id"])
                self.node_details.setdefault(node_id, dict(node))
                self.node_statuses.setdefault(node_id, "PENDING")
        elif event.kind == "PLAN_FAILED":
            self.current_stage = "planner_failed"
        elif event.kind == "NODE_STARTED" and event.node_id:
            if event.node_id in self._terminal_nodes:
                return
            self.current_stage = "execute"
            self.node_details.setdefault(
                event.node_id,
                {"node_id": event.node_id, "agent_name": event.agent_name, "task": event.task, "depends_on": []},
            )
            self.node_statuses[event.node_id] = "RUNNING"
            self.node_started_at.setdefault(event.node_id, now)
        elif event.kind == "NODE_FINISHED" and event.node_id:
            if event.node_id in self._terminal_nodes:
                return
            status = event.node_status or "FAILED"
            self.node_details.setdefault(
                event.node_id,
                {"node_id": event.node_id, "agent_name": event.agent_name, "task": event.task, "depends_on": []},
            )
            self.node_statuses[event.node_id] = status
            if status in _TERMINAL_NODE_STATUSES:
                self._terminal_nodes.add(event.node_id)
            if event.node_id in self.node_started_at:
                self.node_elapsed[event.node_id] = now - self.node_started_at[event.node_id]
        elif event.kind == "ANSWER_STARTED":
            self.current_stage = "answer"
            self._answer_started_at = now
        elif event.kind == "WORKFLOW_FINISHED":
            self.current_stage = "finished"

    def _plan_table(self) -> Table:
        table = Table(title="Execution Plan", border_style="bright_cyan", expand=False)
        table.add_column("Node", style="bright_cyan")
        table.add_column("Agent", style="white")
        table.add_column("Task", style="white")
        table.add_column("Depends On", style="bright_blue")
        table.add_column("Status")
        table.add_column("Duration", style="bright_yellow", justify="right")
        for node in self.plan.get("nodes", []):
            node_id = str(node["node_id"])
            status = self.node_statuses.get(node_id, "PENDING")
            table.add_row(
                node_id,
                str(node.get("agent_name", "")),
                str(node.get("task", "")),
                ", ".join(node.get("depends_on", [])) or "—",
                _styled_status(status),
                self._node_duration(node_id, status),
            )
        return table

    def _node_duration(self, node_id: str, status: str) -> str:
        if status == "RUNNING" and node_id in self.node_started_at:
            return _elapsed(time.perf_counter() - self.node_started_at[node_id])
        if node_id in self.node_elapsed:
            return _elapsed(self.node_elapsed[node_id])
        return "—"

    def _stage_elapsed(self) -> float:
        if self.current_stage == "answer":
            return time.perf_counter() - getattr(self, "_answer_started_at", time.perf_counter())
        return time.perf_counter() - self.planner_started_at


def _elapsed(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, remaining = divmod(total_seconds, 60)
    if minutes == 0:
        return f"{remaining}s"
    hours, minutes = divmod(minutes, 60)
    if hours == 0:
        return f"{minutes}m {remaining:02d}s"
    return f"{hours}h {minutes:02d}m {remaining:02d}s"


def _styled_status(status: str) -> str:
    styles = {
        "PENDING": "dim",
        "RUNNING": "yellow",
        "SUCCEEDED": "green",
        "FAILED": "red",
        "BLOCKED": "red",
        "SKIPPED": "magenta",
    }
    return f"[{styles.get(status, 'white')}]{status}[/]"
