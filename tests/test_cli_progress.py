from __future__ import annotations

import io
import threading

import pytest
from rich.console import Console

from finpilot.agent.runtime_events import AgentRuntimeEvent
from finpilot import cli_progress
from finpilot.cli_progress import CliRuntimeProgress


def _plan_event() -> AgentRuntimeEvent:
    return AgentRuntimeEvent(
        request_id="request-1",
        kind="PLAN_READY",
        plan={
            "status": "READY",
            "reason": "ready",
            "nodes": [
                {
                    "node_id": "account",
                    "agent_name": "TreasuryDataAgent",
                    "task": "查询账户余额",
                    "depends_on": [],
                },
                {
                    "node_id": "knowledge",
                    "agent_name": "QueryAgent",
                    "task": "查询财务规则",
                    "depends_on": [],
                },
            ],
        },
    )


def test_progress_renders_plan_and_all_parallel_running_agents():
    console = Console(record=True, width=120)
    progress = CliRuntimeProgress(console=console)
    progress.emit(_plan_event())

    threads = [
        threading.Thread(
            target=progress.emit,
            args=(
                AgentRuntimeEvent(
                    request_id="request-1",
                    kind="NODE_STARTED",
                    node_id=node_id,
                    agent_name=agent_name,
                    task=task,
                ),
            ),
        )
        for node_id, agent_name, task in [
            ("account", "TreasuryDataAgent", "查询账户余额"),
            ("knowledge", "QueryAgent", "查询财务规则"),
        ]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    progress.process_pending()
    console.print(progress.render())
    text = console.export_text()

    assert "Execution Plan" in text
    assert "Duration" in text
    assert "TreasuryDataAgent is executing" in text
    assert "QueryAgent is executing" in text
    assert progress.node_statuses == {"account": "RUNNING", "knowledge": "RUNNING"}


def test_progress_keeps_first_terminal_node_event_and_starts_final_answer():
    progress = CliRuntimeProgress(console=Console(width=100))
    progress.emit(_plan_event())
    progress.emit(
        AgentRuntimeEvent(
            request_id="request-1",
            kind="NODE_FINISHED",
            node_id="account",
            agent_name="TreasuryDataAgent",
            task="查询账户余额",
            node_status="SUCCEEDED",
        )
    )
    progress.emit(
        AgentRuntimeEvent(
            request_id="request-1",
            kind="NODE_FINISHED",
            node_id="account",
            agent_name="TreasuryDataAgent",
            task="查询账户余额",
            node_status="FAILED",
            failure_reason="late duplicate",
        )
    )
    progress.emit(AgentRuntimeEvent(request_id="request-1", kind="ANSWER_STARTED"))

    progress.process_pending()

    assert progress.node_statuses["account"] == "SUCCEEDED"
    assert progress.current_stage == "answer"


def test_progress_marks_planner_failure_without_creating_agent_rows():
    progress = CliRuntimeProgress(console=Console(width=100))
    progress.emit(
        AgentRuntimeEvent(
            request_id="request-1",
            kind="PLAN_FAILED",
            failure_reason="invalid plan",
        )
    )

    progress.process_pending()

    assert progress.current_stage == "planner_failed"
    assert progress.node_statuses == {}


def test_progress_uses_event_enqueue_time_for_elapsed(monkeypatch):
    clock = {"now": 10.0}
    monkeypatch.setattr(cli_progress.time, "perf_counter", lambda: clock["now"])
    progress = CliRuntimeProgress(console=Console(width=100))

    progress.emit(AgentRuntimeEvent(request_id="request-1", kind="PLANNER_STARTED"))
    clock["now"] = 12.4
    progress.emit(_plan_event())
    clock["now"] = 100.0
    progress.process_pending()

    assert progress.planner_elapsed == pytest.approx(2.4)


def test_progress_live_can_pause_resume_and_persist_plan():
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=120)

    with CliRuntimeProgress(console=console) as progress:
        progress.emit(_plan_event())
        progress.process_pending()
        progress.pause()
        progress.resume()
        progress.emit(AgentRuntimeEvent(request_id="request-1", kind="WORKFLOW_FINISHED"))

    assert "Execution Plan" in output.getvalue()
