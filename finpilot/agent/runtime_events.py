from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field


RuntimeEventKind = Literal[
    "PLANNER_STARTED",
    "PLAN_READY",
    "PLAN_FAILED",
    "NODE_STARTED",
    "NODE_FINISHED",
    "ANSWER_STARTED",
    "WORKFLOW_FINISHED",
]
RuntimeNodeStatus = Literal["SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED"]


class AgentRuntimeEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    kind: RuntimeEventKind
    plan: dict[str, object] = Field(default_factory=dict)
    node_id: str | None = None
    agent_name: str | None = None
    task: str | None = None
    node_status: RuntimeNodeStatus | None = None
    failure_reason: str | None = None


AgentRuntimeEventCallback = Callable[[AgentRuntimeEvent], None]


def emit_runtime_event(
    callback: AgentRuntimeEventCallback | None,
    event: AgentRuntimeEvent,
) -> None:
    """运行过程展示属于旁路能力，回调异常不得影响主业务。"""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        return
