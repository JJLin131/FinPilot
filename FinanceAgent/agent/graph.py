from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langgraph.graph import END, START, StateGraph

from FinanceAgent.agent.router import IntentRouter
from FinanceAgent.agent.subagents import QuerySubAgent, TransferSubAgent
from FinanceAgent.agent.tools import ToolRegistry
from FinanceAgent.intents import UNKNOWN_INTENT_ANSWER
from FinanceAgent.memory.service import MemoryManager
from FinanceAgent.models import AgentChatResponse, GraphState, RouteDecision
from FinanceAgent.observability.audit import AuditStore
from FinanceAgent.observability.tracing import current_trace_id, span


class FinanceAgentGraph:
    def __init__(self, router: IntentRouter, tools: ToolRegistry, audit_store: AuditStore, memory_manager: MemoryManager):
        self.router = router
        self.tools = tools
        self.audit_store = audit_store
        self.memory_manager = memory_manager
        self.query_agent = QuerySubAgent()
        self.transfer_agent = TransferSubAgent()
        self.graph = self._build_graph()

    def run(self, tenant_id: str, user_id: str, chat_id: str, content: str) -> AgentChatResponse:
        request_id = str(uuid.uuid4())
        with span(
            "agent.chat",
            {
                "tenant_id": tenant_id,
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
                tenant_id=tenant_id,
                user_id=user_id,
                chat_id=chat_id,
                memory_id=f"chat:{user_id}:{chat_id}",
                user_message=content,
            )
            final_state = self.graph.invoke(state.model_dump())
            graph_state = GraphState.model_validate(final_state)
            route = self._to_route(graph_state)
            response = AgentChatResponse(
                request_id=request_id,
                trace_id=trace_id,
                domain="FINANCE",
                status="SUCCEEDED",
                answer=graph_state.final_answer,
                evidence=graph_state.evidence,
                route=route,
                route_debug={
                    "classifier_intent": graph_state.classifier_intent,
                    "embedding_top1": graph_state.embedding_top1,
                    "embedding_top2": graph_state.embedding_top2,
                    "fallback_cause": graph_state.fallback_cause,
                    "latency_breakdown": graph_state.latency_breakdown,
                    "loop_count": graph_state.loop_count,
                    "stop_reason": graph_state.stop_reason,
                    "evidence_sufficient": graph_state.evidence_sufficient,
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
        builder.add_node("context_load", self._context_load)
        builder.add_node("intent_classify", self._intent_classify)
        builder.add_node("embedding_score", self._embedding_score)
        builder.add_node("route_decide", self._route_decide)
        builder.add_node("subagent", self._subagent)
        builder.add_node("answer_compose", self._answer_compose)
        builder.add_node("audit_persist", self._audit_persist)
        builder.add_edge(START, "context_load")
        builder.add_edge("context_load", "intent_classify")
        builder.add_edge("intent_classify", "embedding_score")
        builder.add_edge("embedding_score", "route_decide")
        builder.add_edge("route_decide", "subagent")
        builder.add_edge("subagent", "answer_compose")
        builder.add_edge("answer_compose", "audit_persist")
        builder.add_edge("audit_persist", END)
        return builder.compile()

    def _context_load(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "context.load"):
            memory_context = self.memory_manager.load(
                graph_state.memory_id,
                graph_state.user_id,
                graph_state.user_message,
            )
            graph_state.recent_messages = [item.model_dump(mode="json") for item in memory_context.recent_messages]
            graph_state.structured_memory = dict(memory_context.structured_memory)
            graph_state.semantic_memory = [item.model_dump(mode="json") for item in memory_context.semantic_memory]
            graph_state.long_term_memory = list(memory_context.long_term_memory)
            return graph_state.model_dump()

    def _intent_classify(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "intent.classify"):
            classifier_intent, reason = self.router.classify(graph_state.user_message)
            graph_state.classifier_intent = classifier_intent
            graph_state.route_reason = reason
            graph_state.raw_intent_json = json.dumps(
                {"intent": classifier_intent, "reason": reason},
                ensure_ascii=False,
            )
            return graph_state.model_dump()

    def _embedding_score(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "intent.embed_score"):
            top1, top2, semantic_score, margin_score = self.router.embedding_score(graph_state.user_message)
            graph_state.embedding_top1 = top1
            graph_state.embedding_top2 = top2
            graph_state.semantic_score = semantic_score
            graph_state.margin_score = margin_score
            return graph_state.model_dump()

    def _route_decide(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "route.decide"):
            decision = self.router.route_from_scores(
                graph_state.classifier_intent,
                graph_state.route_reason,
                graph_state.embedding_top1,
                graph_state.embedding_top2,
                graph_state.semantic_score,
                graph_state.margin_score,
            )
            graph_state.raw_intent_json = decision.raw_intent_json
            graph_state.classifier_intent = decision.classifier_intent
            graph_state.embedding_top1 = decision.embedding_top1_intent
            graph_state.embedding_top2 = decision.embedding_top2_intent
            graph_state.route_reason = decision.reason
            graph_state.normalized_intent = decision.normalized_intent
            graph_state.target_agent = decision.target_agent
            graph_state.route_confidence = decision.confidence
            graph_state.fallback_cause = decision.fallback_cause
            graph_state.agreement_score = decision.agreement_score
            graph_state.semantic_score = decision.semantic_score
            graph_state.margin_score = decision.margin_score
            return graph_state.model_dump()

    def _subagent(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "tool.invoke"):
            if graph_state.normalized_intent == "UNKNOWN":
                graph_state.final_answer = UNKNOWN_INTENT_ANSWER
                return graph_state.model_dump()
            if graph_state.target_agent == self.transfer_agent.name:
                graph_state = self.transfer_agent.handle(graph_state, self.tools)
            else:
                graph_state = self.query_agent.handle(graph_state, self.tools)
            return graph_state.model_dump()

    def _answer_compose(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "answer.compose"):
            if not graph_state.final_answer:
                graph_state.final_answer = UNKNOWN_INTENT_ANSWER
            graph_state.scores["faithfulness"] = 1.0 if graph_state.evidence else 0.4
            return graph_state.model_dump()

    def _audit_persist(self, state: dict[str, Any]) -> dict[str, Any]:
        graph_state = GraphState.model_validate(state)
        with self._timed_span(graph_state, "audit.persist"):
            decision = self._to_route(graph_state)
            if decision.normalized_intent == "UNKNOWN":
                self.audit_store.record_unknown_intent(graph_state, decision)
            for invocation in graph_state.tool_invocations:
                self.audit_store.record_tool(graph_state, invocation)
            return graph_state.model_dump()

    def _to_route(self, state: GraphState) -> RouteDecision:
        return RouteDecision(
            raw_intent_json=state.raw_intent_json,
            normalized_intent=state.normalized_intent,
            reason=state.route_reason,
            confidence=state.route_confidence,
            valid=state.normalized_intent != "UNKNOWN",
            target_agent=state.target_agent,
            classifier_intent=state.classifier_intent,
            embedding_top1_intent=state.embedding_top1,
            embedding_top2_intent=state.embedding_top2,
            fallback_cause=state.fallback_cause,
            semantic_score=state.semantic_score,
            margin_score=state.margin_score,
            agreement_score=state.agreement_score,
        )

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
                        "tenant_id": state.tenant_id,
                        "user_id": state.user_id,
                        "chat_id": state.chat_id,
                        "intent": state.normalized_intent,
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
