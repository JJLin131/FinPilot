from __future__ import annotations

from typing import Any

from FinanceAgent.agent.tools import ToolRegistry
from FinanceAgent.models import GraphState, LoopContext, SessionContext, SubAgentContext


def build_session_context(state: GraphState) -> SessionContext:
    return SessionContext(
        tenant_id=state.tenant_id,
        user_id=state.user_id,
        chat_id=state.chat_id,
        memory_id=state.memory_id,
        user_message=state.user_message,
        normalized_intent=state.normalized_intent,
        target_agent=state.target_agent,
        recent_messages=state.recent_messages[-6:],
        structured_memory=dict(state.structured_memory),
        semantic_memory=state.semantic_memory[-8:],
        long_term_memory=state.long_term_memory[-8:],
    )


def build_subagent_context(state: GraphState, agent_name: str, tools: ToolRegistry) -> SubAgentContext:
    if agent_name == "QueryAgent":
        return SubAgentContext(
            agent_name=agent_name,
            role="You are a finance knowledge QA sub-agent.",
            goal="Answer the user's current finance question using retrieved knowledge evidence.",
            constraints=[
                "Do not invent finance rules.",
                "Use retrieved evidence when it is available.",
                "If evidence is insufficient, say that the knowledge base does not contain enough information.",
                "Do not perform route decisions, audit persistence, or system governance work.",
            ],
            success_criteria=[
                "A supported answer can be produced from evidence.",
                "Or the available evidence is insufficient and the agent stops clearly.",
            ],
            allowed_tools=tools.list_allowed(["search_finance_knowledge"]),
            max_steps=3,
        )
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
    return {
        "session": session_context.model_dump(mode="json"),
        "sub_agent": subagent_context.model_dump(mode="json"),
        "loop": loop_context.model_dump(mode="json"),
    }


def build_answer_prompt_context(
    session_context: SessionContext,
    subagent_context: SubAgentContext,
    loop_context: LoopContext,
) -> dict[str, Any]:
    return {
        "user_message": session_context.user_message,
        "agent_goal": subagent_context.goal,
        "evidence_sufficient": loop_context.evidence_sufficient,
        "recent_messages": session_context.recent_messages[-4:],
        "structured_memory": session_context.structured_memory,
        "semantic_memory": session_context.semantic_memory[-5:],
        "long_term_memory": session_context.long_term_memory[-5:],
        "working_notes": loop_context.working_notes[-3:],
        "step_history": [step.model_dump(mode="json") for step in loop_context.step_history[-2:]],
    }
