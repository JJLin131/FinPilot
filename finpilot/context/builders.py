from __future__ import annotations

from typing import Any

from finpilot.agent.tools import ToolRegistry
from finpilot.agent.prompts import build_agent_decision_prompt
from finpilot.context.compression import ContextBuilder, ContextSegment, PromptContextBundle
from finpilot.llm import render_finance_answer_prompt
from finpilot.models import GraphState, LoopContext, SessionContext, SubAgentContext


_context_builder = ContextBuilder()


def build_session_context(state: GraphState) -> SessionContext:
    global_session = state.global_context.get("session") if isinstance(state.global_context, dict) else None
    if isinstance(global_session, dict):
        current_session = dict(global_session)
        current_session.update(
            {
                "user_id": state.user_id,
                "chat_id": state.chat_id,
                "memory_id": state.memory_id,
                "user_message": state.user_message,
                "normalized_intent": state.normalized_intent,
                "target_agent": state.target_agent,
            }
        )
        return SessionContext.model_validate(current_session)
    return SessionContext(
        user_id=state.user_id,
        chat_id=state.chat_id,
        memory_id=state.memory_id,
        user_message=state.user_message,
        normalized_intent=state.normalized_intent,
        target_agent=state.target_agent,
        recent_messages=state.recent_messages[-6:],
        structured_memory=dict(state.structured_memory),
        semantic_memory=state.semantic_memory[-8:],
    )


def build_subagent_context(state: GraphState, agent_name: str, tools: ToolRegistry) -> SubAgentContext:
    return SubAgentContext(
        agent_name=agent_name,
        role=f"You are the {agent_name} sub-agent.",
        goal="Handle only the task assigned by the outer graph.",
        constraints=["Do not bypass routing, auditing, or graph control."],
        success_criteria=["Return a final answer or an explicit fallback."],
        allowed_tools=[],
        max_steps=1,
    )


def init_loop_context(state: GraphState, subagent_context: SubAgentContext) -> LoopContext:
    return LoopContext(
        step_index=0,
        max_steps=max(subagent_context.max_steps, 1),
        step_history=list(state.step_history),
        working_notes=[],
        evidence_sufficient=state.evidence_sufficient,
        stop_reason=state.stop_reason,
    )


def build_loop_prompt_context(
    session_context: SessionContext,
    subagent_context: SubAgentContext,
    loop_context: LoopContext,
) -> dict[str, Any]:
    return build_loop_prompt_bundle(session_context, subagent_context, loop_context).payload


def build_loop_prompt_bundle(
    session_context: SessionContext,
    subagent_context: SubAgentContext,
    loop_context: LoopContext,
    *,
    summary_cache: dict[str, Any] | None = None,
) -> PromptContextBundle:
    return _context_builder.build(
        subagent_context.context_policy,
        [
            ContextSegment(name="session", value=session_context, priority=10),
            ContextSegment(name="sub_agent", value=subagent_context, priority=20),
            ContextSegment(name="loop", value=loop_context, priority=30),
        ],
        stage="decision",
        query=session_context.user_message,
        prompt_renderer=build_agent_decision_prompt,
        summary_cache=summary_cache,
    )


def build_answer_prompt_bundle(
    session_context: SessionContext,
    subagent_context: SubAgentContext,
    loop_context: LoopContext,
    evidence: list[dict[str, Any]],
    *,
    summary_cache: dict[str, Any] | None = None,
) -> PromptContextBundle:
    answer_loop = {
        "working_notes": loop_context.working_notes[-3:],
        "step_history": [step.model_dump(mode="json") for step in loop_context.step_history[-2:]],
        "evidence_sufficient": loop_context.evidence_sufficient,
        "draft_answer": loop_context.draft_answer,
    }
    return _context_builder.build(
        subagent_context.context_policy,
        [
            ContextSegment(name="session", value=session_context, priority=10),
            ContextSegment(name="loop", value=answer_loop, priority=20),
            ContextSegment(name="evidence", value=evidence, priority=30),
        ],
        stage="answer",
        query=session_context.user_message,
        prompt_renderer=render_finance_answer_prompt,
        summary_cache=summary_cache,
    )


def build_global_prompt_bundle(
    state: GraphState,
    *,
    subagent_results: list[dict[str, Any]] | None = None,
    summary_cache: dict[str, Any] | None = None,
) -> PromptContextBundle:
    session_context = build_session_context(state)
    results = subagent_results
    if results is None and isinstance(state.global_context, dict):
        existing = state.global_context.get("subagent_results")
        results = existing if isinstance(existing, list) else None
    segments = [ContextSegment(name="session", value=session_context, priority=10)]
    if results:
        segments.append(ContextSegment(name="subagent_results", value=results, priority=20))
    return _context_builder.build(
        "global_default",
        segments,
        stage="global",
        query=state.user_message,
        summary_cache=summary_cache,
    )

