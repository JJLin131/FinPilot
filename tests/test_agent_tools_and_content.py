from __future__ import annotations

from finpilot.agent.agents.finance_qa_subagent import FinanceQaSubAgent
from finpilot.agent.runtime import AgentRuntime
from finpilot.config import settings
from finpilot.context.builders import build_session_context
from finpilot.context.compression import ContextBuilder
from finpilot.context.summarization import SemanticSummary
from finpilot.memory.models import MemoryContext
from finpilot.models import (
    AgentDecision,
    GraphState,
    LoopContext,
    LoopStepRecord,
    RagMatch,
    SessionContext,
    SubAgentContext,
    ToolCard,
    ToolObservation,
)


class FakeTools:
    def list_allowed(self, tool_names: list[str]) -> list[ToolCard]:
        return [
            ToolCard(
                name=name,
                description=f"{name} description",
                when_to_use=f"use {name}",
                arguments={"path": "string"},
            )
            for name in tool_names
        ]


class RecordingAnsweringService:
    def __init__(self):
        self.prompt_context = None

    def answer_with_context(self, prompt_context: dict) -> str:
        self.prompt_context = prompt_context
        return "answer from content"


class DraftAnswerDecisionService:
    def decide(self, prompt: str) -> AgentDecision:
        del prompt
        return AgentDecision(
            decision="answer",
            enough_information=True,
            draft_answer="candidate draft",
        )


class DeterministicSummarizer:
    def summarize(self, **kwargs) -> SemanticSummary:
        del kwargs
        return SemanticSummary(
            summary="压缩后的 RAG 证据",
            key_facts=["保留文档事实"],
        )


def _state() -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="summarize local note",
        normalized_intent="GENERAL_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


def test_query_agent_uses_configured_tool_allowlist(monkeypatch):
    monkeypatch.setattr(
        settings,
        "agent_tool_allowlists",
        {"QueryAgent": ["search_finance_knowledge", "read_file", "web_search"]},
        raising=False,
    )

    context = FinanceQaSubAgent().build_context(FakeTools())

    assert [tool.name for tool in context.allowed_tools] == [
        "search_finance_knowledge",
        "read_file",
        "web_search",
    ]


def test_runtime_merges_generic_content_into_evidence_and_answer_context():
    answering = RecordingAnsweringService()
    runtime = AgentRuntime(answering_service=answering)
    state = _state()
    state.recent_messages = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    state.structured_memory = {"city": "南京"}
    state.semantic_memory = [{"memory_key": "userBank", "memory_value": "用户常用招商银行。"}]
    output = {
        "content": [
            {
                "source": "D:/workspace/note.txt",
                "title": "note.txt",
                "text": "local file evidence",
                "metadata": {"kind": "file"},
            }
        ]
    }
    loop_context = LoopContext(
        step_history=[
            LoopStepRecord(
                step_index=1,
                decision=AgentDecision(decision="act", tool_name="read_file", tool_args={"path": "note.txt"}),
                observation=ToolObservation(
                    tool_name="read_file",
                    status="SUCCEEDED",
                    summary="read 19 chars",
                    output=output,
                ),
            )
        ]
    )
    subagent = SubAgentContext(agent_name="QueryAgent", role="role", goal="goal")

    runtime._merge_tool_output(state, output)
    runtime._compose_final_answer(state, subagent, loop_context)

    assert state.evidence[0].tool_name == "read_file"
    assert state.evidence[0].source == "D:/workspace/note.txt"
    assert state.evidence[0].summary["title"] == "note.txt"
    assert answering.prompt_context["evidence"][0]["text"] == "local file evidence"
    assert answering.prompt_context["session"]["user_message"] == state.user_message
    assert answering.prompt_context["session"]["recent_messages"] == state.recent_messages
    assert answering.prompt_context["session"]["structured_memory"] == {"city": "南京"}
    assert answering.prompt_context["session"]["semantic_memory"] == state.semantic_memory
    assert "loop" not in answering.prompt_context


def test_context_models_do_not_duplicate_long_term_memory():
    assert "long_term_memory" not in MemoryContext.model_fields
    assert "long_term_memory" not in GraphState.model_fields
    assert "long_term_memory" not in SessionContext.model_fields


def test_compatibility_answer_helper_uses_unified_answer_policy():
    answering = RecordingAnsweringService()
    runtime = AgentRuntime(answering_service=answering)
    state = _state()
    state.recent_messages = [{"role": "user", "content": "x" * 200_000}]
    loop_context = LoopContext(working_notes=["summary"])
    subagent = SubAgentContext(agent_name="QueryAgent", role="role", goal="goal")

    runtime._compose_final_answer(state, subagent, loop_context)

    assert answering.prompt_context["session"]["recent_messages"][0]["content"] == "x" * 200_000
    assert state.context_usage["answer"]["effective_policy"] == "answer_default"


def test_final_answer_includes_evidence_in_budgeted_bundle_without_raw_bypass(monkeypatch):
    builder = ContextBuilder(
        policies={
            "answer_default": {
                "token_budget": 4_000,
                "reserved_output_tokens": 512,
                "segment_limits": {"session": 1_000, "evidence": 2_000},
                "compression_rules": [
                    {
                        "name": "evidence",
                        "path": "evidence",
                        "method": "llm",
                        "target_tokens": 1_000,
                        "priority": 40,
                    }
                ],
            }
        },
        summarizer=DeterministicSummarizer(),
        model_context_windows={},
    )
    monkeypatch.setattr("finpilot.context.builders._context_builder", builder)
    answering = RecordingAnsweringService()
    runtime = AgentRuntime(answering_service=answering)
    state = _state()
    huge_text = "RAG evidence " * 2_000
    state.reranked_docs = [
        RagMatch(
            document_id=f"doc-{index}",
            title=f"Document {index}",
            source="manual",
            text=huge_text,
            score=0.9,
        )
        for index in range(3)
    ]

    runtime._compose_final_answer(
        state,
        SubAgentContext(agent_name="QueryAgent", role="role", goal="goal", context_policy="finance_qa_agent"),
        LoopContext(working_notes=["summary"]),
    )

    serialized = str(answering.prompt_context)
    assert huge_text not in serialized
    assert answering.prompt_context["evidence"]
    assert state.context_usage["answer"]["within_budget"] is True


def test_subagent_result_is_budgeted_before_writing_global_context():
    runtime = AgentRuntime(answering_service=RecordingAnsweringService())
    state = _state()
    state.global_context = {"session": build_session_context(state).model_dump(mode="json")}
    raw_rag = "raw rag body " * 20_000
    loop_context = LoopContext(
        step_history=[
            LoopStepRecord(
                step_index=1,
                decision=AgentDecision(decision="act", tool_name="search_finance_knowledge"),
                observation=ToolObservation(
                    tool_name="search_finance_knowledge",
                    status="SUCCEEDED",
                    summary="retrieved",
                    output={"documents": [{"document_id": "doc-1", "text": raw_rag}]},
                ),
            )
        ]
    )

    runtime._sync_global_context(
        state,
        SubAgentContext(agent_name="QueryAgent", role="role", goal="goal"),
        loop_context,
    )

    serialized = str(state.global_context)
    assert raw_rag not in serialized
    assert state.subagent_results[0].agent_name == "QueryAgent"
    assert state.context_usage["global"][-1]["within_budget"] is True


def test_runtime_run_does_not_generate_final_answer_inside_subagent():
    answering = RecordingAnsweringService()
    runtime = AgentRuntime(
        decision_service=DraftAnswerDecisionService(),
        answering_service=answering,
    )
    state = _state()
    subagent = SubAgentContext(agent_name="QueryAgent", role="role", goal="goal", max_steps=1)

    runtime.run(state, subagent, tools=object())

    assert state.final_answer == ""
    assert answering.prompt_context is None
    assert state.context_usage["decision"][0]["stage"] == "decision"


def test_runtime_execute_returns_structured_result_without_generating_final_answer():
    runtime = AgentRuntime(decision_service=DraftAnswerDecisionService())
    state = _state()
    subagent = SubAgentContext(agent_name="QueryAgent", role="role", goal="goal", max_steps=1)

    result = runtime.execute(state, subagent, tools=object(), node_id="knowledge", task="检索适用规则")

    assert result.node_id == "knowledge"
    assert result.agent_name == "QueryAgent"
    assert result.status == "SUCCEEDED"
    assert result.summary == "candidate draft"
    assert state.final_answer == ""
    assert "documents" not in result.output
