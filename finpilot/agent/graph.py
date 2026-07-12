from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from finpilot.agent.agents import FinanceQaSubAgent, TreasuryDataAgent, TreasuryOperationAgent
from finpilot.agent.orchestration import AgentRegistration, ExecutionPlan, ExecutionPlanNode, ExecutionPlanningService, ExecutionScheduler
from finpilot.agent.runtime_events import AgentRuntimeEvent, AgentRuntimeEventCallback, emit_runtime_event
from finpilot.agent.prompts import build_execution_plan_prompt
from finpilot.agent.runtime import CONTEXT_BUDGET_EXCEEDED_ANSWER
from finpilot.agent.tools import ToolRegistry
from finpilot.context.builders import build_answer_prompt_bundle, build_global_prompt_bundle, build_plan_prompt_bundle, build_session_context
from finpilot.context.compression import ContextEventCallback
from finpilot.issues import dependency_degraded_issue, issue_from_safety_finding, issue_from_tool_failure
from finpilot.llm import FinanceAnsweringService
from finpilot.memory.service import MemoryManager
from finpilot.models import (
    AgentChatResponse,
    AgentEvidence,
    AgentIssue,
    GraphState,
    SafetyFinding,
    SubAgentResult,
    ToolInvocation,
)
from finpilot.observability.audit import AuditStore
from finpilot.observability.tracing import current_trace_id, span
from finpilot.safety.service import SafetyReviewService


SAFETY_BLOCKED_ANSWER = "请求被安全策略阻断，无法继续执行。"


class FinPilotGraph:
    def __init__(
        self,
        router: Any | None = None,
        *,
        tools: ToolRegistry,
        audit_store: AuditStore,
        memory_manager: MemoryManager,
        safety: SafetyReviewService | None = None,
        planner: ExecutionPlanningService | None = None,
        answering_service: FinanceAnsweringService | None = None,
        context_event_callback: ContextEventCallback | None = None,
        runtime_event_callback: AgentRuntimeEventCallback | None = None,
    ):
        del router
        self.tools = tools
        self.audit_store = audit_store
        self.memory_manager = memory_manager
        self.safety = safety or SafetyReviewService()
        self.planner = planner or ExecutionPlanningService()
        self.answering_service = answering_service or FinanceAnsweringService()
        self.context_event_callback = context_event_callback
        self.runtime_event_callback = runtime_event_callback
        self.query_agent = FinanceQaSubAgent()
        self.treasury_data_agent = TreasuryDataAgent()
        self.treasury_operation_agent = TreasuryOperationAgent()
        self.graph = self._build_graph()

    def run(self, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        request_id = str(uuid.uuid4())
        with span(
            "agent.chat",
            {
                "user_id": user_id,
                "chat_id": chat_id,
                "intent": "PENDING",
                "model": "deepseek-v4-pro",
            },
        ):
            trace_id = current_trace_id()
            state = GraphState(
                request_id=request_id,
                trace_id=trace_id,
                user_id=user_id,
                chat_id=chat_id,
                memory_id=f"chat:{user_id}:{chat_id}",
                user_message=content,
            )
            try:
                final_state = self.graph.invoke(state.model_dump())
            finally:
                self._emit_runtime_event(AgentRuntimeEvent(request_id=request_id, kind="WORKFLOW_FINISHED"))
            graph_state = GraphState.model_validate(final_state)
            response = AgentChatResponse(
                request_id=request_id,
                trace_id=trace_id,
                domain="FINANCE",
                status=self._response_status(graph_state),
                answer=graph_state.final_answer,
                evidence=graph_state.evidence,
                plan=graph_state.execution_plan,
                issues=graph_state.issues,
                safety_findings=graph_state.safety_findings,
                plan_debug={
                    "planning_status": graph_state.planning_status,
                    "planning_reason": graph_state.planning_reason,
                    "planning_attempts": graph_state.planning_attempts,
                    "execution_plan": graph_state.execution_plan,
                    "issues": [issue.model_dump(mode="json") for issue in graph_state.issues],
                    "latency_breakdown": graph_state.latency_breakdown,
                    "loop_count": graph_state.loop_count,
                    "stop_reason": graph_state.stop_reason,
                    "evidence_sufficient": graph_state.evidence_sufficient,
                    "context_usage": graph_state.context_usage,
                    "context_compactions": graph_state.context_compactions,
                    "safety_findings": [finding.model_dump(mode="json") for finding in graph_state.safety_findings],
                },
                retrieval_debug={
                    "retrieved_docs": [item.model_dump() for item in graph_state.retrieved_docs],
                    "reranked_docs": [item.model_dump() for item in graph_state.reranked_docs],
                    "step_history": [item.model_dump(mode="json") for item in graph_state.step_history],
                },
                tool_calls=graph_state.tool_invocations,
            )
            self.memory_manager.remember_interaction(state.memory_id, user_id, chat_id, content, response)
            return response

    def _build_graph(self):
        builder = StateGraph(dict)
        builder.add_node("input_safety_review", self._input_safety_review)
        builder.add_node("context_load", self._context_load)
        builder.add_node("plan_build", self._plan_build)
        builder.add_node("dag_execute", self._dag_execute)
        builder.add_node("answer_compose", self._answer_compose)
        builder.add_node("response_safety_review", self._response_safety_review)
        builder.add_node("audit_persist", self._audit_persist)
        builder.add_edge(START, "input_safety_review")
        builder.add_conditional_edges(
            "input_safety_review",
            self._after_input_safety_review,
            {"continue": "context_load", "blocked": "audit_persist"},
        )
        builder.add_conditional_edges(
            "context_load",
            self._after_context_load,
            {"continue": "plan_build", "blocked": "response_safety_review"},
        )
        builder.add_edge("plan_build", "dag_execute")
        builder.add_edge("dag_execute", "answer_compose")
        builder.add_edge("answer_compose", "response_safety_review")
        builder.add_edge("response_safety_review", "audit_persist")
        builder.add_edge("audit_persist", END)
        return builder.compile()

    def _input_safety_review(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "safety.input"):
            result = self.safety.review_input(graph_state)
            self._apply_safety_review(graph_state, result)
            if result.action == "BLOCK":
                graph_state.final_answer = SAFETY_BLOCKED_ANSWER
            return graph_state.model_dump()

    def _after_input_safety_review(self, state: dict[str, Any]) -> str:
        graph_state = GraphState.model_validate(state)
        return "blocked" if graph_state.final_answer == SAFETY_BLOCKED_ANSWER else "continue"

    def _context_load(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "context.load"):
            memory_context = self.memory_manager.load(
                graph_state.user_id,
                graph_state.chat_id,
                graph_state.user_message,
            )
            graph_state.recent_messages = [item.model_dump(mode="json") for item in memory_context.recent_messages]
            graph_state.structured_memory = dict(memory_context.structured_memory)
            graph_state.semantic_memory = [item.model_dump(mode="json") for item in memory_context.semantic_memory]
            bundle = build_global_prompt_bundle(
                graph_state,
                summary_cache={},
                event_callback=self.context_event_callback,
            )
            graph_state.global_context = bundle.payload
            graph_state.context_usage["global"] = bundle.usage.model_dump(mode="json")
            for event in bundle.compression_events:
                enriched = dict(event)
                enriched.setdefault("stage", "global")
                enriched.setdefault("policy", bundle.usage.effective_policy)
                graph_state.context_compactions.append(enriched)
            if bundle.status == "over_budget":
                graph_state.final_answer = CONTEXT_BUDGET_EXCEEDED_ANSWER
                graph_state.issues.append(
                    AgentIssue(
                        code="CONTEXT_BUDGET_EXCEEDED",
                        component="context:global",
                        message=CONTEXT_BUDGET_EXCEEDED_ANSWER,
                        severity="error",
                        retryable=True,
                        detail=json.dumps(
                            {
                                "budget": bundle.usage.effective_input_budget,
                                "estimated": bundle.usage.estimated_prompt_tokens,
                                "paths": bundle.over_budget_paths,
                            },
                            ensure_ascii=False,
                        ),
                    )
                )
            return graph_state.model_dump()

    def _after_context_load(self, state: dict[str, Any]) -> str:
        graph_state = GraphState.model_validate(state)
        return "blocked" if graph_state.final_answer == CONTEXT_BUDGET_EXCEEDED_ANSWER else "continue"

    def _plan_build(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "agent.plan"):
            self._emit_runtime_event(
                AgentRuntimeEvent(request_id=graph_state.request_id, kind="PLANNER_STARTED")
            )
            agents = self._agent_descriptors()
            session = build_session_context(graph_state)
            bundle = build_plan_prompt_bundle(session, agents, summary_cache={})
            self._record_context_bundle(graph_state, "planner", bundle)
            if bundle.status == "over_budget":
                self._record_context_budget_issue(graph_state, bundle)
                self._emit_runtime_event(
                    AgentRuntimeEvent(
                        request_id=graph_state.request_id,
                        kind="PLAN_FAILED",
                        failure_reason="Planner context budget exceeded.",
                    )
                )
                return graph_state.model_dump()
            try:
                plan = self.planner.plan(
                    build_execution_plan_prompt(bundle.payload),
                    set(self._agent_instances()),
                )
            except Exception as exc:
                self._emit_runtime_event(
                    AgentRuntimeEvent(
                        request_id=graph_state.request_id,
                        kind="PLAN_FAILED",
                        failure_reason=str(exc),
                    )
                )
                graph_state.issues.append(
                    dependency_degraded_issue(
                        code="EXECUTION_PLAN_FAILED",
                        component="execution_planner",
                        message="Execution planner failed after the repair attempt.",
                        exc=exc,
                    ).model_copy(update={"severity": "error", "retryable": True})
                )
                graph_state.planning_status = "FAILED"
                graph_state.planning_reason = str(exc)
                graph_state.planning_attempts = 2
                return graph_state.model_dump()
            graph_state.planning_status = plan.status
            graph_state.planning_reason = plan.reason
            graph_state.planning_attempts = plan.attempts
            graph_state.execution_plan = plan.model_dump(mode="json")
            self._emit_runtime_event(
                AgentRuntimeEvent(
                    request_id=graph_state.request_id,
                    kind="PLAN_READY",
                    plan=graph_state.execution_plan,
                )
            )
            if plan.status == "UNSUPPORTED":
                graph_state.final_answer = plan.reason
            return graph_state.model_dump()

    def _dag_execute(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "agent.execute"):
            if graph_state.final_answer or graph_state.planning_status in {"FAILED", "UNSUPPORTED"}:
                return graph_state.model_dump()
            try:
                plan = ExecutionPlan.model_validate(graph_state.execution_plan)
                scheduler = ExecutionScheduler(
                    self._agent_registrations(graph_state),
                    request_id=graph_state.request_id,
                    event_callback=self.runtime_event_callback,
                )
                session = build_session_context(graph_state)
                results = scheduler.execute(plan, session)
            except Exception as exc:
                graph_state.issues.append(
                    dependency_degraded_issue(
                        code="EXECUTION_DAG_FAILED",
                        component="execution_scheduler",
                        message="Execution plan could not be completed.",
                        exc=exc,
                    )
                )
                return graph_state.model_dump()

            graph_state.subagent_results = list(session.subagent_results)
            for result in results:
                self._merge_execution_result(graph_state, result)

            global_bundle = build_global_prompt_bundle(
                graph_state,
                summary_cache={},
                event_callback=self.context_event_callback,
            )
            self._record_context_bundle(graph_state, "global", global_bundle)
            if global_bundle.status == "ready":
                graph_state.global_context = global_bundle.payload
            else:
                self._record_context_budget_issue(graph_state, global_bundle)
            graph_state.issues.extend(
                issue_from_tool_failure(invocation)
                for invocation in graph_state.tool_invocations
                if invocation.status == "FAILED"
            )
            return graph_state.model_dump()

    def _answer_compose(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "answer.compose"):
            if graph_state.final_answer:
                return graph_state.model_dump()
            if graph_state.issues and not graph_state.evidence and not graph_state.subagent_results:
                system_answer = self._issue_answer(graph_state)
                if system_answer:
                    graph_state.final_answer = system_answer
                    return graph_state.model_dump()
            self._emit_runtime_event(
                AgentRuntimeEvent(request_id=graph_state.request_id, kind="ANSWER_STARTED")
            )
            bundle = build_answer_prompt_bundle(
                build_session_context(graph_state),
                graph_state.answer_evidence,
                summary_cache={},
            )
            self._record_context_bundle(graph_state, "answer", bundle)
            if bundle.status == "over_budget":
                self._record_context_budget_issue(graph_state, bundle)
                return graph_state.model_dump()
            graph_state.final_answer = self.answering_service.answer_with_context(bundle.payload)
            consume = getattr(self.answering_service, "consume_issues", None)
            if callable(consume):
                graph_state.issues.extend(consume())
            graph_state.scores["faithfulness"] = 1.0 if graph_state.evidence else 0.4
            return graph_state.model_dump()

    def _emit_runtime_event(self, event: AgentRuntimeEvent) -> None:
        emit_runtime_event(self.runtime_event_callback, event)

    def _agent_instances(self) -> dict[str, Any]:
        return {
            self.query_agent.name: self.query_agent,
            self.treasury_data_agent.name: self.treasury_data_agent,
            self.treasury_operation_agent.name: self.treasury_operation_agent,
        }

    def _agent_descriptors(self) -> list[dict[str, Any]]:
        descriptors: list[dict[str, Any]] = []
        for agent in self._agent_instances().values():
            context = agent.build_context(self.tools)
            tool_names = [tool.name for tool in context.allowed_tools]
            risk_level = self.tools.max_risk_level(tool_names)
            descriptors.append(
                {
                    "name": agent.name,
                    "role": context.role,
                    "goal": context.goal,
                    "allowed_tools": tool_names,
                    "execution_mode": "read_only" if risk_level == "low" else "operation",
                }
            )
        return descriptors

    def _agent_registrations(self, parent_state: GraphState) -> dict[str, AgentRegistration]:
        registrations: dict[str, AgentRegistration] = {}
        for agent in self._agent_instances().values():
            context = agent.build_context(self.tools)
            tool_names = [tool.name for tool in context.allowed_tools]
            risk_level = self.tools.max_risk_level(tool_names)

            def execute(node: ExecutionPlanNode, session, *, current_agent=agent) -> SubAgentResult:
                # 子节点在独立状态副本中执行，调度器在每层结束后统一写回主状态。
                local_state = parent_state.model_copy(deep=True)
                local_state.global_context = {"session": session.model_dump(mode="json")}
                local_state.subagent_results = [item.model_copy(deep=True) for item in session.subagent_results]
                return current_agent.execute(
                    local_state,
                    self.tools,
                    node_id=node.node_id,
                    task=node.task,
                )

            registrations[agent.name] = AgentRegistration(
                name=agent.name,
                execution_mode="read_only" if risk_level == "low" else "operation",
                execute=execute,
            )
        return registrations

    def _merge_execution_result(self, state: GraphState, result: SubAgentResult) -> None:
        for evidence in result.evidence_summary:
            try:
                state.evidence.append(AgentEvidence.model_validate(evidence))
            except Exception:
                state.issues.append(
                    AgentIssue(
                        code="SUBAGENT_EVIDENCE_INVALID",
                        component=f"subagent:{result.agent_name}",
                        message="Sub-agent returned invalid evidence metadata.",
                        severity="warning",
                    )
                )
        if result.status == "SUCCEEDED":
            state.answer_evidence.extend(result.raw_evidence)

        runtime_data = result.runtime_data
        for raw_issue in runtime_data.get("issues", []):
            try:
                state.issues.append(AgentIssue.model_validate(raw_issue))
            except Exception:
                continue
        for raw_invocation in runtime_data.get("tool_invocations", []):
            try:
                state.tool_invocations.append(ToolInvocation.model_validate(raw_invocation))
            except Exception:
                continue
        for raw_finding in runtime_data.get("safety_findings", []):
            try:
                state.safety_findings.append(SafetyFinding.model_validate(raw_finding))
            except Exception:
                continue

        if runtime_data.get("context_usage"):
            execution_usage = state.context_usage.setdefault("execution", {})
            execution_usage[result.node_id] = runtime_data["context_usage"]
        state.context_compactions.extend(runtime_data.get("context_compactions", []))
        state.loop_count += int(runtime_data.get("loop_count", 0))

    @staticmethod
    def _record_context_bundle(graph_state: GraphState, key: str, bundle, *, append: bool = False) -> None:
        usage = bundle.usage.model_dump(mode="json")
        if append:
            graph_state.context_usage.setdefault(key, []).append(usage)
        else:
            graph_state.context_usage[key] = usage
        for event in bundle.compression_events:
            enriched = dict(event)
            enriched.setdefault("stage", bundle.usage.stage)
            enriched.setdefault("policy", bundle.usage.effective_policy)
            graph_state.context_compactions.append(enriched)

    @staticmethod
    def _record_context_budget_issue(graph_state: GraphState, bundle) -> None:
        graph_state.final_answer = CONTEXT_BUDGET_EXCEEDED_ANSWER
        graph_state.issues.append(
            AgentIssue(
                code="CONTEXT_BUDGET_EXCEEDED",
                component=f"context:{bundle.usage.stage}",
                message=CONTEXT_BUDGET_EXCEEDED_ANSWER,
                severity="error",
                retryable=True,
                detail=json.dumps(
                    {
                        "budget": bundle.usage.effective_input_budget,
                        "estimated": bundle.usage.estimated_prompt_tokens,
                        "paths": bundle.over_budget_paths,
                    },
                    ensure_ascii=False,
                ),
            )
        )

    def _response_safety_review(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "safety.response"):
            result = self.safety.review_response(graph_state)
            self._apply_safety_review(graph_state, result)
            if result.action == "BLOCK":
                graph_state.final_answer = SAFETY_BLOCKED_ANSWER
            elif result.action == "REDACT" and isinstance(result.sanitized_payload, str):
                graph_state.final_answer = result.sanitized_payload
            return graph_state.model_dump()

    def _audit_persist(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "audit.persist"):
            try:
                for issue in graph_state.issues:
                    self.audit_store.record_issue(graph_state, issue)
                for finding in graph_state.safety_findings:
                    self.audit_store.record_safety_finding(graph_state, finding)
                for decision_event in graph_state.safety_approval_decisions:
                    self.audit_store.record_approval_decision(graph_state, decision_event)
                for invocation in graph_state.tool_invocations:
                    self.audit_store.record_tool(graph_state, invocation)
            except Exception as exc:
                graph_state.issues.append(
                    dependency_degraded_issue(
                        code="AUDIT_PERSIST_DEGRADED",
                        component="audit",
                        message="Audit persistence failed; response remains available with degraded diagnostics.",
                        exc=exc,
                    )
                )
            return graph_state.model_dump()

    def _response_status(self, state: GraphState) -> str:
        # 状态在图执行末端统一收口，避免中间节点把系统故障误标成 unsupported。
        if any(issue.component == "safety" and issue.severity == "error" for issue in state.issues):
            return "FAILED"
        if state.planning_status == "UNSUPPORTED":
            return "UNSUPPORTED"
        if state.planning_status == "FAILED":
            return "FAILED"
        if state.issues:
            if not state.evidence and any(issue.severity == "error" for issue in state.issues):
                return "FAILED"
            return "DEGRADED"
        return "SUCCEEDED"

    def _issue_answer(self, state: GraphState) -> str:
        for issue in state.issues:
            if issue.severity == "error":
                return issue.message
        return state.issues[0].message if state.issues else ""

    def _apply_safety_review(self, state: GraphState, result) -> None:
        if not result.findings:
            return
        state.safety_findings.extend(result.findings)
        state.issues.extend(issue_from_safety_finding(finding) for finding in result.findings)

    def _timed_span(self, state: GraphState, span_name: str):
        class TimedContext:
            def __init__(self):
                self.started = 0.0
                self._span = None

            def __enter__(self):
                self.started = time.perf_counter()
                self._span = span(
                    span_name,
                    {
                        "user_id": state.user_id,
                        "chat_id": state.chat_id,
                        "planning_status": state.planning_status,
                    },
                )
                self._span.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                duration_ms = round((time.perf_counter() - self.started) * 1000, 2)
                state.latency_breakdown[span_name] = duration_ms
                self._span.__exit__(exc_type, exc, tb)
                return False

        return TimedContext()

