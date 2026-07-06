from __future__ import annotations

import json

from finpilot.agent.runtime import AgentRuntime
from finpilot.context.compression import ContextBuilder, ContextPolicy, ContextSegment
from finpilot.models import (
    AgentDecision,
    GraphState,
    LoopContext,
    LoopStepRecord,
    SessionContext,
    SubAgentContext,
    ToolCard,
    ToolObservation,
)


def test_context_builder_does_not_compress_under_trigger():
    builder = ContextBuilder(
        policies={
            "test_policy": ContextPolicy(
                token_budget=10_000,
                trigger_ratio=0.85,
                compression_order=["session.recent_messages"],
            )
        }
    )

    bundle = builder.build(
        "test_policy",
        [
            ContextSegment(name="session", value={"user_message": "hello"}),
            ContextSegment(name="sub_agent", value={"agent_name": "QueryAgent"}),
        ],
    )

    assert bundle.payload["session"]["user_message"] == "hello"
    assert bundle.usage["compressed"] is False
    assert bundle.compression_events == []


def test_finance_policy_compresses_rag_documents_and_preserves_current_message():
    long_text = "finance rule " * 2000
    loop = {
        "step_history": [
            {
                "step_index": 1,
                "decision": {"decision": "act", "tool_name": "search_finance_knowledge"},
                "observation": {
                    "tool_name": "search_finance_knowledge",
                    "status": "SUCCEEDED",
                    "summary": "retrieved 2 documents",
                    "output": {
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "title": "Payroll rules",
                                "source": "manual",
                                "score": 0.91,
                                "text": long_text,
                            }
                        ]
                    },
                },
            },
            {
                "step_index": 2,
                "decision": {"decision": "answer"},
                "observation": None,
            },
        ]
    }
    builder = ContextBuilder(
        policies={
            "finance_qa_agent": ContextPolicy(
                token_budget=900,
                trigger_ratio=0.85,
                compression_order=["loop.tool_outputs", "loop.old_step_history"],
                protected_paths=["session.user_message", "loop.step_history[-1]"],
            )
        }
    )

    bundle = builder.build(
        "finance_qa_agent",
        [
            ContextSegment(name="session", value={"user_message": "payroll approval?", "recent_messages": []}),
            ContextSegment(name="sub_agent", value={"agent_name": "QueryAgent", "goal": "answer finance question"}),
            ContextSegment(name="loop", value=loop),
        ],
    )

    assert bundle.payload["session"]["user_message"] == "payroll approval?"
    assert bundle.payload["loop"]["step_history"][-1]["step_index"] == 2
    serialized = json.dumps(bundle.payload, ensure_ascii=False)
    assert long_text not in serialized
    assert "documents_summary" in serialized
    assert bundle.usage["compressed"] is True
    assert any(event["action"] == "summarized_tool_output" for event in bundle.compression_events)
    assert bundle.usage["estimated_tokens_after"] <= bundle.usage["token_budget"]


def test_context_builder_preserves_negative_index_protected_paths():
    latest_note = "must keep latest step details " * 100
    builder = ContextBuilder(
        policies={
            "protect_latest": ContextPolicy(
                token_budget=100,
                trigger_ratio=0.1,
                protected_paths=["loop.step_history[-1]"],
                compression_order=[],
            )
        }
    )

    bundle = builder.build(
        "protect_latest",
        [
            ContextSegment(
                name="loop",
                value={
                    "step_history": [
                        {"step_index": 1, "note": "old details " * 100},
                        {"step_index": 2, "note": latest_note},
                    ]
                },
            )
        ],
    )

    assert bundle.payload["loop"]["step_history"][-1]["note"] == latest_note


class RecordingDecisionService:
    def __init__(self) -> None:
        self.prompt = ""

    def decide(self, prompt: str) -> AgentDecision:
        self.prompt = prompt
        return AgentDecision(decision="answer", enough_information=True, reason="ready")


def test_runtime_uses_prompt_bundle_and_records_context_debug():
    decision_service = RecordingDecisionService()
    runtime = AgentRuntime(decision_service=decision_service)
    huge_doc = "RAG body " * 80_000
    state = GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="payroll rule?",
        normalized_intent="FINANCE_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )
    session_context = SessionContext(
        user_id=state.user_id,
        chat_id=state.chat_id,
        memory_id=state.memory_id,
        user_message=state.user_message,
        normalized_intent=state.normalized_intent,
        target_agent=state.target_agent,
    )
    subagent_context = SubAgentContext(
        agent_name="QueryAgent",
        role="finance qa",
        goal="answer finance question",
        allowed_tools=[
            ToolCard(
                name="search_finance_knowledge",
                description="search",
                when_to_use="finance qa",
                arguments={"query": "string"},
            )
        ],
        context_policy="finance_qa_agent",
    )
    loop_context = LoopContext(
        step_index=1,
        max_steps=3,
        step_history=[
            LoopStepRecord(
                step_index=1,
                decision=AgentDecision(
                    decision="act",
                    tool_name="search_finance_knowledge",
                    tool_args={"query": "payroll"},
                ),
                observation=ToolObservation(
                    tool_name="search_finance_knowledge",
                    status="SUCCEEDED",
                    summary="retrieved 1 documents",
                    output={
                        "documents": [
                            {
                                "document_id": "doc-1",
                                "title": "Payroll",
                                "source": "manual",
                                "score": 0.9,
                                "text": huge_doc,
                            }
                        ]
                    },
                ),
            )
        ],
    )

    decision = runtime._decide(session_context, subagent_context, loop_context, state)

    assert decision.decision == "answer"
    assert "decision" in state.context_usage
    assert state.context_compactions
    assert huge_doc not in decision_service.prompt
    assert "documents_summary" in decision_service.prompt
