from __future__ import annotations

from finpilot.agent.agents.finance_qa_subagent import FinanceQaSubAgent
from finpilot.agent.subagents import QuerySubAgent
from finpilot.models import GraphState, ToolCard


class RecordingRuntime:
    def __init__(self) -> None:
        self.contexts = []

    def run(self, state: GraphState, subagent_context, tools):
        self.contexts.append(subagent_context)
        state.final_answer = "handled"
        return state


class FakeTools:
    def list_allowed(self, tool_names: list[str]) -> list[ToolCard]:
        return [
            ToolCard(
                name=name,
                description=f"{name} description",
                when_to_use=f"use {name}",
                arguments={"query": "string"},
            )
            for name in tool_names
        ]


def _state(intent: str = "FINANCE_KNOWLEDGE_QA") -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="what finance rule applies?",
        normalized_intent=intent,
        target_agent="QueryAgent",
    )


def test_finance_qa_subagent_builds_finance_policy_context():
    runtime = RecordingRuntime()
    agent = FinanceQaSubAgent(runtime=runtime)

    result = agent.handle(_state(), FakeTools())

    assert result.final_answer == "handled"
    assert len(runtime.contexts) == 1
    context = runtime.contexts[0]
    assert context.agent_name == "QueryAgent"
    assert context.context_policy == "finance_qa_agent"
    assert [tool.name for tool in context.allowed_tools] == ["search_finance_knowledge"]


def test_query_subagent_remains_a_compatibility_alias():
    assert QuerySubAgent is FinanceQaSubAgent

