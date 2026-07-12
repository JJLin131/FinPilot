from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field

from finpilot.models import SessionContext, SubAgentResult


class ExecutionPlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=64)
    agent_name: str = Field(min_length=1, max_length=64)
    task: str = Field(min_length=1, max_length=4_000)
    depends_on: list[str] = Field(default_factory=list)


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["READY", "UNSUPPORTED"] = "READY"
    reason: str = ""
    attempts: int = Field(default=1, exclude=True)
    nodes: list[ExecutionPlanNode] = Field(default_factory=list)

    def validate_for_registry(self, registered_agents: set[str], *, max_nodes: int = 3) -> None:
        if self.status == "UNSUPPORTED":
            if self.nodes:
                raise ValueError("unsupported execution plan must not contain nodes")
            if not self.reason.strip():
                raise ValueError("unsupported execution plan must include a reason")
            return
        if not self.nodes:
            raise ValueError("execution plan must contain at least one node")
        if len(self.nodes) > max_nodes:
            raise ValueError(f"execution plan exceeds maximum node count: {max_nodes}")

        node_ids = [node.node_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("execution plan node ids must be unique")

        known_nodes = set(node_ids)
        for node in self.nodes:
            if node.agent_name not in registered_agents:
                raise ValueError(f"execution plan uses unregistered agent: {node.agent_name}")
            unknown_dependencies = set(node.depends_on) - known_nodes
            if unknown_dependencies:
                names = ", ".join(sorted(unknown_dependencies))
                raise ValueError(f"execution plan contains unknown dependency: {names}")
            if node.node_id in node.depends_on:
                raise ValueError(f"execution plan contains cycle at node: {node.node_id}")

        # 使用拓扑遍历保证计划可执行，并在循环依赖时给出明确错误。
        remaining = {node.node_id: set(node.depends_on) for node in self.nodes}
        ready = deque(node_id for node_id, dependencies in remaining.items() if not dependencies)
        visited = 0
        while ready:
            completed = ready.popleft()
            visited += 1
            for node_id, dependencies in remaining.items():
                if completed in dependencies:
                    dependencies.remove(completed)
                    if not dependencies:
                        ready.append(node_id)
        if visited != len(self.nodes):
            raise ValueError("execution plan contains dependency cycle")


AgentExecutor = Callable[[ExecutionPlanNode, SessionContext], SubAgentResult]


@dataclass(frozen=True)
class AgentRegistration:
    name: str
    execution_mode: Literal["read_only", "operation"]
    execute: AgentExecutor


class ExecutionScheduler:
    def __init__(self, registry: dict[str, AgentRegistration], *, max_parallelism: int = 3):
        self.registry = dict(registry)
        self.max_parallelism = max_parallelism

    def execute(self, plan: ExecutionPlan, session: SessionContext) -> list[SubAgentResult]:
        plan.validate_for_registry(set(self.registry), max_nodes=self.max_parallelism)
        pending = list(plan.nodes)
        results: list[SubAgentResult] = []
        results_by_node: dict[str, SubAgentResult] = {}

        while pending:
            skipped = self._skip_failed_dependencies(pending, results_by_node)
            if skipped:
                for result in skipped:
                    results.append(result)
                    results_by_node[result.node_id] = result
                    session.subagent_results.append(result)
                pending = [node for node in pending if node.node_id not in results_by_node]
                continue

            ready = [
                node
                for node in pending
                if all(
                    dependency in results_by_node and results_by_node[dependency].status == "SUCCEEDED"
                    for dependency in node.depends_on
                )
            ]
            if not ready:
                raise RuntimeError("execution plan has no runnable nodes")

            # 同一依赖层共享只读快照，避免并发任务直接修改 session。
            layer_session = session.model_copy(deep=True)
            layer_results = self._run_layer(ready, layer_session)
            for result in layer_results:
                results.append(result)
                results_by_node[result.node_id] = result
                session.subagent_results.append(result)
            pending = [node for node in pending if node.node_id not in results_by_node]

        return results

    def _run_layer(self, nodes: list[ExecutionPlanNode], session: SessionContext) -> list[SubAgentResult]:
        read_only = [node for node in nodes if self.registry[node.agent_name].execution_mode == "read_only"]
        operations = [node for node in nodes if self.registry[node.agent_name].execution_mode == "operation"]
        results: dict[str, SubAgentResult] = {}

        if read_only:
            worker_count = min(len(read_only), self.max_parallelism)
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="finpilot-agent") as executor:
                futures = {
                    node.node_id: executor.submit(self._run_node, node, session.model_copy(deep=True))
                    for node in read_only
                }
                for node in read_only:
                    results[node.node_id] = futures[node.node_id].result()

        # 操作节点串行执行，避免同层副作用出现不可预测的执行顺序。
        for node in operations:
            results[node.node_id] = self._run_node(node, session.model_copy(deep=True))

        return [results[node.node_id] for node in nodes]

    def _run_node(self, node: ExecutionPlanNode, session: SessionContext) -> SubAgentResult:
        try:
            return self.registry[node.agent_name].execute(node, session)
        except Exception as exc:
            return SubAgentResult(
                node_id=node.node_id,
                agent_name=node.agent_name,
                task=node.task,
                status="FAILED",
                failure_reason=str(exc),
            )

    @staticmethod
    def _skip_failed_dependencies(
        pending: list[ExecutionPlanNode],
        results_by_node: dict[str, SubAgentResult],
    ) -> list[SubAgentResult]:
        skipped: list[SubAgentResult] = []
        for node in pending:
            failed_dependencies = [
                dependency
                for dependency in node.depends_on
                if dependency in results_by_node and results_by_node[dependency].status != "SUCCEEDED"
            ]
            if failed_dependencies:
                skipped.append(
                    SubAgentResult(
                        node_id=node.node_id,
                        agent_name=node.agent_name,
                        task=node.task,
                        status="SKIPPED",
                        failure_reason=f"dependency unavailable: {', '.join(failed_dependencies)}",
                    )
                )
        return skipped


class ExecutionPlanningService:
    def __init__(self, client=None, model_name: str | None = None):
        self.client = client or self._build_client()
        self.model_name = model_name or self._model_name()

    def plan(self, prompt: str, registered_agents: set[str]) -> ExecutionPlan:
        current_prompt = prompt
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = self.client.generate(current_prompt, model_name=self.model_name)
                plan = ExecutionPlan.model_validate(self._extract_json(raw))
                plan.validate_for_registry(registered_agents)
                return plan.model_copy(update={"attempts": attempt + 1})
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    current_prompt = self._repair_prompt(prompt, exc, registered_agents)
        raise ValueError(f"execution planning failed after 2 attempts: {last_error}") from last_error

    @staticmethod
    def _repair_prompt(original_prompt: str, error: Exception, registered_agents: set[str]) -> str:
        agents = ", ".join(sorted(registered_agents))
        return (
            f"{original_prompt}\n\n"
            "The previous execution plan was invalid. Return a corrected JSON plan only.\n"
            f"Validation error: {error}\n"
            f"Registered agents: {agents}"
        )

    @staticmethod
    def _extract_json(raw: str) -> dict:
        text = raw.strip().replace("```json", "```")
        if "```" in text:
            for part in (item.strip() for item in text.split("```") if item.strip()):
                if part.startswith("{") and part.endswith("}"):
                    return json.loads(part)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)

    @staticmethod
    def _model_name() -> str:
        from finpilot.config import settings

        return settings.ai_model_name

    @staticmethod
    def _build_client():
        from finpilot.config import settings
        from finpilot.llm import DeepSeekChatClient, OllamaClient

        if settings.ai_provider.lower() == "deepseek":
            return DeepSeekChatClient(
                model_name=settings.ai_model_name,
                timeout_seconds=settings.query_rewriter_timeout_seconds,
            )
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.ai_model_name,
            timeout_seconds=settings.query_rewriter_timeout_seconds,
        )
