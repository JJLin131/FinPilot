from __future__ import annotations

from finpilot.agent.agents.finance_qa_subagent import FinanceQaSubAgent
from finpilot.agent.runtime import AgentRuntime
from finpilot.config import settings
from finpilot.models import AgentDecision, GraphState, LoopContext, LoopStepRecord, SubAgentContext, ToolCard, ToolObservation


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
        self.snippets = []

    def answer_with_context(self, question: str, snippets: list[str]) -> str:
        self.snippets = snippets
        return "answer from content"


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
    assert answering.snippets == ["local file evidence"]

