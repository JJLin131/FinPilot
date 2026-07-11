from __future__ import annotations

import json

import pytest

from finpilot.agent.runtime import AgentRuntime
from finpilot.context.builders import build_answer_prompt_bundle
from finpilot.context.compression import (
    CompressionRule,
    ContextBuilder,
    ContextLifecycleEvent,
    ContextPolicy,
    ContextSegment,
)
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


class FakeSummarizer:
    def summarize(self, **kwargs):
        raise AssertionError(f"unexpected summarization call: {kwargs}")


def test_context_builder_does_not_compress_under_trigger():
    builder = ContextBuilder(
        policies={
            "test_policy": ContextPolicy(
                token_budget=10_000,
                trigger_ratio=0.85,
                compression_rules=[
                    CompressionRule(
                        name="recent_messages",
                        path="session.recent_messages",
                        method="llm",
                        target_tokens=500,
                    )
                ],
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


def test_global_context_builder_emits_build_and_compression_events():
    events: list[ContextLifecycleEvent] = []
    builder = ContextBuilder(
        policies={
            "global_default": ContextPolicy(
                token_budget=160,
                trigger_ratio=0.25,
                reserved_output_tokens=10,
                compression_rules=[
                    CompressionRule(
                        name="recent_messages",
                        path="session.recent_messages",
                        method="deterministic",
                        target_tokens=40,
                        priority=10,
                        preserve_last=1,
                    )
                ],
            )
        },
        summarizer=FakeSummarizer(),
    )

    bundle = builder.build(
        "global_default",
        [ContextSegment(name="session", value={"recent_messages": [{"content": "x" * 2000}]})],
        stage="global",
        event_callback=events.append,
    )

    assert [event.kind for event in events] == [
        "build_started",
        "compression_started",
        "compression_finished",
        "build_finished",
    ]
    assert all(event.stage == "global" for event in events)
    assert all(event.policy == "global_default" for event in events)
    assert events[-1].estimated_tokens == bundle.usage.estimated_tokens_after
    assert events[-1].effective_input_budget == bundle.usage.effective_input_budget
    assert events[-1].within_budget == bundle.usage.within_budget


def test_context_builder_does_not_emit_events_without_callback():
    builder = ContextBuilder(policies={"plain": ContextPolicy(token_budget=1000)}, summarizer=FakeSummarizer())
    bundle = builder.build(
        "plain",
        [ContextSegment(name="session", value={"user_message": "hello"})],
        stage="global",
    )

    assert bundle.status == "ready"


def test_context_callback_failure_does_not_break_context_build():
    def broken_callback(event: ContextLifecycleEvent) -> None:
        raise RuntimeError("ui closed")

    builder = ContextBuilder(policies={"plain": ContextPolicy(token_budget=1000)}, summarizer=FakeSummarizer())
    bundle = builder.build(
        "plain",
        [ContextSegment(name="session", value={"user_message": "hello"})],
        stage="global",
        event_callback=broken_callback,
    )

    assert bundle.status == "ready"


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
                compression_rules=[
                    CompressionRule(
                        name="tool_outputs",
                        path="loop.tool_outputs",
                        method="llm",
                        target_tokens=300,
                        priority=10,
                    ),
                    CompressionRule(
                        name="old_step_history",
                        path="loop.old_step_history",
                        method="deterministic",
                        target_tokens=200,
                        priority=20,
                    ),
                ],
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
    assert any(
        event["action"] in {"llm_summarized", "summarization_failed"}
        and event["path"].endswith("observation.output.documents")
        for event in bundle.compression_events
    )
    assert all(event["stage"] == "decision" for event in bundle.compression_events)
    assert all(event["policy"] == "finance_qa_agent" for event in bundle.compression_events)
    assert all(event["method"] in {"deterministic", "llm"} for event in bundle.compression_events)
    assert bundle.usage["estimated_tokens_after"] <= bundle.usage["token_budget"]


def test_context_builder_preserves_negative_index_protected_paths():
    latest_note = "must keep latest step details " * 100
    builder = ContextBuilder(
        policies={
            "protect_latest": ContextPolicy(
                token_budget=100,
                trigger_ratio=0.1,
                protected_paths=["loop.step_history[-1]"],
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


def test_negative_index_protection_applies_to_latest_step_child_fields():
    protected_reason = "protected reason " * 100
    builder = ContextBuilder(
        policies={
            "protect_latest_child": ContextPolicy(
                token_budget=1_000,
                trigger_ratio=0.1,
                protected_paths=["loop.step_history[-1].decision.reason"],
                segment_limits={"loop": 300},
            )
        }
    )

    bundle = builder.build(
        "protect_latest_child",
        [
            ContextSegment(
                name="loop",
                value={
                    "step_history": [
                        {
                            "step_index": 1,
                            "decision": {"decision": "act", "reason": protected_reason},
                            "observation": {"summary": "x" * 5_000},
                        }
                    ]
                },
            )
        ],
        stage="decision",
        query="current",
    )

    assert bundle.payload["loop"]["step_history"][-1]["decision"]["reason"] == protected_reason


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
    assert isinstance(state.context_usage["decision"], list)
    assert state.context_usage["decision"][0]["stage"] == "decision"
    assert state.context_compactions
    assert huge_doc not in decision_service.prompt
    assert "documents_summary" in decision_service.prompt


def test_policy_override_merges_with_default_rules_and_protection():
    builder = ContextBuilder(policies={"finance_qa_agent": {"token_budget": 16_000}})

    policy = builder.policies["finance_qa_agent"]

    assert policy.token_budget == 16_000
    assert "session.user_message" in policy.protected_paths
    assert policy.compression_rules


def test_policy_override_merges_individual_compression_rule_fields():
    builder = ContextBuilder(
        policies={
            "finance_qa_agent": {
                "compression_rules": [
                    {"name": "evidence", "target_tokens": 12_000},
                ]
            }
        }
    )

    rule = next(item for item in builder.policies["finance_qa_agent"].compression_rules if item.name == "evidence")

    assert rule.target_tokens == 12_000
    assert rule.path == "evidence"
    assert rule.method == "llm"
    assert rule.priority == 50


def test_unknown_policy_is_rejected_instead_of_silently_falling_back():
    builder = ContextBuilder()

    with pytest.raises(ValueError, match="unknown context policy"):
        builder.build(
            "missing_policy",
            [ContextSegment(name="session", value={"user_message": "hello"})],
            stage="decision",
            query="hello",
        )


def test_uncompressible_payload_returns_over_budget_status():
    builder = ContextBuilder(
        policies={
            "tiny": ContextPolicy(
                token_budget=100,
                trigger_ratio=0.5,
                protected_paths=["session.user_message"],
            )
        }
    )

    bundle = builder.build(
        "tiny",
        [
            ContextSegment(name="session", value={"user_message": "hello"}),
            ContextSegment(name="memory", value={str(index): "x" * 20 for index in range(500)}),
        ],
        stage="decision",
        query="hello",
    )

    assert bundle.status == "over_budget"
    assert bundle.usage.within_budget is False
    assert bundle.usage.estimated_tokens_after > bundle.usage.effective_input_budget


def test_current_user_message_over_budget_is_never_truncated():
    user_message = "重" * 500
    builder = ContextBuilder(
        policies={
            "tiny": ContextPolicy(
                token_budget=100,
                trigger_ratio=0.5,
                protected_paths=["session.user_message"],
            )
        }
    )

    bundle = builder.build(
        "tiny",
        [ContextSegment(name="session", value={"user_message": user_message})],
        stage="answer",
        query=user_message,
    )

    assert bundle.status == "over_budget"
    assert bundle.payload["session"]["user_message"] == user_message
    assert "session.user_message" in bundle.over_budget_paths


def test_wildcard_protected_path_is_reported_when_it_exceeds_budget():
    builder = ContextBuilder(
        policies={
            "tool_policy": ContextPolicy(
                token_budget=100,
                trigger_ratio=0.5,
                protected_paths=["sub_agent.allowed_tools[*].name"],
            )
        }
    )

    bundle = builder.build(
        "tool_policy",
        [
            ContextSegment(
                name="sub_agent",
                value={"allowed_tools": [{"name": "工" * 200, "arguments": {}}]},
            )
        ],
        stage="decision",
        query="current",
    )

    assert bundle.status == "over_budget"
    assert "sub_agent.allowed_tools[*].name" in bundle.over_budget_paths


def test_global_context_uses_global_policy_and_keeps_original_state_unchanged():
    from finpilot.context.builders import build_global_prompt_bundle, build_session_context

    state = GraphState(
        request_id="req-global",
        trace_id="trace-global",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="current question",
        recent_messages=[{"role": "user", "content": "old" * 80_000}],
        semantic_memory=[{"memory_key": "preference", "memory_value": "value" * 20_000}],
    )
    original_recent_messages = list(state.recent_messages)

    bundle = build_global_prompt_bundle(state, summary_cache={})

    assert bundle.usage.effective_policy == "global_default"
    assert bundle.usage.stage == "global"
    assert bundle.payload["session"]["user_message"] == "current question"
    assert state.recent_messages == original_recent_messages
    assert "global_context" in GraphState.model_fields

    state.global_context = bundle.payload
    state.normalized_intent = "FINANCE_KNOWLEDGE_QA"
    state.target_agent = "QueryAgent"
    routed_session = build_session_context(state)
    assert routed_session.normalized_intent == "FINANCE_KNOWLEDGE_QA"
    assert routed_session.target_agent == "QueryAgent"


def test_model_context_window_reserves_output_tokens_from_input_budget():
    builder = ContextBuilder(
        policies={
            "windowed": ContextPolicy(
                token_budget=10_000,
                reserved_output_tokens=1_000,
            )
        },
        model_context_windows={"provider:model": 5_000},
        model_label="provider:model",
    )

    bundle = builder.build(
        "windowed",
        [ContextSegment(name="session", value={"user_message": "hello"})],
        stage="decision",
        query="hello",
    )

    assert bundle.usage.effective_input_budget == 4_000
    assert bundle.usage.model_window_source == "model_context_windows"


def test_effective_model_window_drives_final_deterministic_compaction():
    builder = ContextBuilder(
        policies={
            "windowed": ContextPolicy(
                token_budget=10_000,
                reserved_output_tokens=1_000,
            )
        },
        model_context_windows={"provider:model": 1_900},
        model_label="provider:model",
    )

    bundle = builder.build(
        "windowed",
        [
            ContextSegment(name="session", value={"user_message": "hello"}),
            ContextSegment(name="memory", value={"content": "x" * 12_000}),
        ],
        stage="decision",
        query="hello",
    )

    assert bundle.usage.effective_input_budget == 900
    assert bundle.status == "ready"
    assert bundle.usage.compressed is True
    assert any(event["action"] == "truncated_strings" for event in bundle.compression_events)


def test_compression_rule_rejects_unsupported_path():
    with pytest.raises(ValueError, match="unsupported compression path"):
        CompressionRule(
            name="invalid",
            path="session.unknown_field",
            method="llm",
            target_tokens=100,
        )


def test_answer_bundle_uses_global_answer_policy_without_loop_context():
    session = SessionContext(
        user_id="user-1",
        chat_id="chat-1",
        memory_id="memory-1",
        user_message="根据证据回答",
        normalized_intent="FINANCE_KNOWLEDGE_QA",
        target_agent="QueryAgent",
        subagent_results=[
            {
                "node_id": "knowledge",
                "agent_name": "QueryAgent",
                "task": "检索规则",
                "status": "SUCCEEDED",
                "summary": "命中规则",
            }
        ],
    )

    bundle = build_answer_prompt_bundle(session, [{"source": "manual", "text": "证据正文"}])

    assert bundle.usage.effective_policy == "answer_default"
    assert bundle.usage.stage == "answer"
    assert bundle.payload["session"]["subagent_results"][0]["summary"] == "命中规则"
    assert bundle.payload["evidence"][0]["text"] == "证据正文"
    assert "loop" not in bundle.payload
