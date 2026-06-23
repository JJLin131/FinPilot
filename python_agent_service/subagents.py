from __future__ import annotations

from python_agent_service.intents import UNKNOWN_INTENT_ANSWER
from python_agent_service.llm import FinanceAnsweringService
from python_agent_service.models import AgentEvidence, GraphState, RagMatch
from python_agent_service.tools import ToolRegistry


class QuerySubAgent:
    name = "QueryAgent"

    def __init__(self):
        self.answering = FinanceAnsweringService()

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        if state.normalized_intent == "ACCOUNT_BALANCE":
            invocation = tools.invoke(state, "query_account_balance")
            state.tool_invocations.append(invocation)
            state.evidence.append(
                AgentEvidence(
                    tool_name=invocation.tool_name,
                    source="finance-ledger",
                    summary=invocation.output,
                )
            )
            state.final_answer = f"当前可用余额为 {invocation.output['available_balance']} {invocation.output['currency']}。"
            return state

        if state.normalized_intent in {"KNOWLEDGE_QA", "GENERAL_FINANCE"}:
            invocation = tools.invoke(state, "search_finance_knowledge")
            state.tool_invocations.append(invocation)
            documents = invocation.output.get("documents", [])
            state.retrieved_docs = []
            state.reranked_docs = []
            snippets = []
            for item in documents:
                match = RagMatch(**item)
                state.retrieved_docs.append(match)
                state.reranked_docs.append(match)
                state.evidence.append(
                    AgentEvidence(
                        tool_name=invocation.tool_name,
                        source=item["source"],
                        summary={"document_id": item["document_id"], "score": item["score"]},
                    )
                )
                snippets.append(item["text"])
            state.final_answer = self.answering.answer_with_context(state.user_message, snippets)
            return state

        state.final_answer = UNKNOWN_INTENT_ANSWER
        return state


class TransferSubAgent:
    name = "TransferAgent"

    def handle(self, state: GraphState, tools: ToolRegistry) -> GraphState:
        intent_to_tool = {
            "CANCEL_TRANSFER": "cancel_transfer",
            "TRANSFER_STATUS": "check_transfer_status",
            "TRANSFER_RECEIPT": "get_transfer_receipt",
        }
        tool_name = intent_to_tool.get(state.normalized_intent)
        if not tool_name:
            state.final_answer = UNKNOWN_INTENT_ANSWER
            return state

        invocation = tools.invoke(state, tool_name)
        state.tool_invocations.append(invocation)
        state.evidence.append(
            AgentEvidence(
                tool_name=invocation.tool_name,
                source="transfer-system",
                summary=invocation.output,
            )
        )

        if state.normalized_intent == "CANCEL_TRANSFER":
            state.final_answer = f"转账撤销请求已提交，参考号 {invocation.output['reference']}。"
        elif state.normalized_intent == "TRANSFER_STATUS":
            state.final_answer = f"当前转账状态为 {invocation.output['transfer_status']}，预计 {invocation.output['eta']} 内更新。"
        else:
            state.final_answer = f"回单已生成，下载地址 {invocation.output['download_url']}。"
        return state
