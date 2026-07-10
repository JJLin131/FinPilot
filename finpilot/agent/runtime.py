from __future__ import annotations

import json
import logging
import re

from finpilot.context.builders import (
    build_answer_prompt_bundle,
    build_loop_prompt_bundle,
    build_session_context,
    init_loop_context,
)
from finpilot.agent.decision import AgentDecisionService
from finpilot.agent.prompts import build_agent_decision_prompt
from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.llm import FinanceAnsweringService
from finpilot.models import (
    AgentDecision,
    AgentEvidence,
    AgentIssue,
    GraphState,
    LoopContext,
    LoopStepRecord,
    RagMatch,
    SessionContext,
    SubAgentContext,
    ToolObservation,
)
from finpilot.safety.models import SafetyFinding

logger = logging.getLogger(__name__)


class AgentRuntime:
    def __init__(
        self,
        decision_service: AgentDecisionService | None = None,
        answering_service: FinanceAnsweringService | None = None,
    ):
        self.decision_service = decision_service or AgentDecisionService()
        self.answering_service = answering_service or FinanceAnsweringService()

    def run(self, state: GraphState, subagent_context: SubAgentContext, tools: ToolRegistry) -> GraphState:
        session_context = build_session_context(state)
        loop_context = init_loop_context(state, subagent_context)
        state.retrieved_docs = []
        state.reranked_docs = []

        while loop_context.step_index < loop_context.max_steps:
            self._debug_contexts(session_context, subagent_context, loop_context)
            decision = self._decide(session_context, subagent_context, loop_context, state)
            self._debug(
                "agent loop decision",
                {
                    "agent": subagent_context.agent_name,
                    "step": loop_context.step_index + 1,
                    "decision": decision.decision,
                    "tool_name": decision.tool_name,
                    "reason": decision.reason,
                },
            )
            if decision.decision == "act":
                self._act(state, subagent_context, loop_context, tools, decision)
                if loop_context.evidence_sufficient:
                    loop_context.stop_reason = "evidence_sufficient"
                    break
                continue

            loop_context.stop_reason = decision.decision
            if decision.draft_answer and not state.final_answer:
                state.final_answer = decision.draft_answer
            break

        if loop_context.stop_reason is None:
            loop_context.stop_reason = "max_steps"

        self._compose_final_answer(state, subagent_context, loop_context)
        self._sync_loop_state(state, loop_context)
        return state

    def _decide(
        self,
        session_context,
        subagent_context: SubAgentContext,
        loop_context: LoopContext,
        state: GraphState,
    ) -> AgentDecision:
        if state.evidence:
            return AgentDecision(decision="answer", reason="Evidence is available.", enough_information=True)

        prompt_bundle = build_loop_prompt_bundle(session_context, subagent_context, loop_context)
        state.context_usage["decision"] = prompt_bundle.usage
        state.context_compactions.extend(prompt_bundle.compression_events)
        try:
            decision = self.decision_service.decide(build_agent_decision_prompt(prompt_bundle.payload))
        except Exception as exc:
            logger.warning("Agent decision LLM failed; using deterministic fallback: %s", exc)
            return self._fallback_decision(loop_context, state, subagent_context)

        allowed = {tool.name for tool in subagent_context.allowed_tools}
        if decision.decision == "act" and decision.tool_name not in allowed:
            return self._fallback_decision(loop_context, state, subagent_context)
        if decision.decision == "answer" and not decision.enough_information and not state.evidence:
            return self._fallback_decision(loop_context, state, subagent_context)
        return decision

    def _fallback_decision(
        self,
        loop_context: LoopContext,
        state: GraphState,
        subagent_context: SubAgentContext | None = None,
    ) -> AgentDecision:
        if loop_context.step_index == 0:
            allowed = {tool.name for tool in subagent_context.allowed_tools} if subagent_context else set()
            if "query_account_balance" in allowed and state.normalized_intent == "TREASURY_DATA_QUERY":
                return AgentDecision(
                    decision="act",
                    reason="Query treasury account balance with deterministic fallback.",
                    tool_name="query_account_balance",
                    tool_args={"accountId": self._extract_identifier(state.user_message, "ACC-001")},
                    enough_information=False,
                )
            if "create_transfer_order" in allowed and state.normalized_intent == "TREASURY_OPERATION":
                return AgentDecision(
                    decision="act",
                    reason="Create mock transfer order with deterministic fallback.",
                    tool_name="create_transfer_order",
                    tool_args={
                        "fromAccountId": self._extract_identifier(state.user_message, "ACC-001"),
                        "toAccountId": self._extract_identifier(state.user_message, "ACC-002", skip_first=True),
                        "amount": self._extract_amount(state.user_message),
                        "currency": "CNY",
                        "purpose": "mock treasury operation",
                    },
                    enough_information=False,
                )
            return AgentDecision(
                decision="act",
                reason="Search finance knowledge before answering.",
                tool_name="search_finance_knowledge",
                tool_args={"query": state.user_message, "limit": 3},
                enough_information=False,
            )
        return AgentDecision(
            decision="answer",
            reason="No additional useful tool decision is available.",
            enough_information=bool(state.evidence),
        )

    def _extract_identifier(self, text: str, default: str, *, skip_first: bool = False) -> str:
        matches = re.findall(r"\b[A-Z]{2,8}-[A-Z0-9-]+\b", text.upper())
        if skip_first and len(matches) > 1:
            return matches[1]
        return matches[0] if matches else default

    def _extract_amount(self, text: str) -> float:
        match = re.search(r"\b\d+(?:\.\d+)?\b", text.replace(",", ""))
        return float(match.group(0)) if match else 1.0

    def _act(
        self,
        state: GraphState,
        subagent_context: SubAgentContext,
        loop_context: LoopContext,
        tools: ToolRegistry,
        decision: AgentDecision,
    ) -> None:
        loop_context.step_index += 1
        tool_name = decision.tool_name or ""
        allowed = {tool.name for tool in subagent_context.allowed_tools}
        if tool_name not in allowed:
            observation = ToolObservation(
                tool_name=tool_name,
                status="FAILED",
                summary=f"Tool is not allowed for {subagent_context.agent_name}: {tool_name}",
            )
            loop_context.step_history.append(
                LoopStepRecord(step_index=loop_context.step_index, decision=decision, observation=observation)
            )
            return

        invocation = tools.invoke(
            state,
            tool_name,
            **decision.tool_args,
            step_index=loop_context.step_index,
            reason=decision.reason,
        )
        state.tool_invocations.append(invocation)
        observation = ToolObservation(
            tool_name=invocation.tool_name,
            status=invocation.status,
            summary=invocation.observation_summary or "",
            output=invocation.output,
        )
        loop_context.step_history.append(
            LoopStepRecord(step_index=loop_context.step_index, decision=decision, observation=observation)
        )
        loop_context.working_notes.append(observation.summary)
        self._debug(
            "agent loop observation",
            {
                "agent": subagent_context.agent_name,
                "step": loop_context.step_index,
                "tool_name": invocation.tool_name,
                "status": invocation.status,
                "summary": observation.summary,
            },
        )
        self._merge_tool_output(state, invocation.output)
        loop_context.evidence_sufficient = bool(state.evidence)

    def _merge_tool_output(self, state: GraphState, output: dict) -> None:
        for item in output.get("issues", []):
            state.issues.append(AgentIssue.model_validate(item))
        safety_payload = output.get("safety", {})
        for item in safety_payload.get("findings", []) if isinstance(safety_payload, dict) else []:
            state.safety_findings.append(SafetyFinding.model_validate(item))
        documents = output.get("documents", [])
        for item in documents:
            match = RagMatch(**item)
            state.retrieved_docs.append(match)
            state.reranked_docs.append(match)
            state.evidence.append(
                AgentEvidence(
                    tool_name="search_finance_knowledge",
                    source=item["source"],
                    summary={"document_id": item["document_id"], "score": item["score"]},
                )
            )
        for item in output.get("content", []):
            if not isinstance(item, dict):
                continue
            state.evidence.append(
                AgentEvidence(
                    tool_name=self._content_tool_name(item, output),
                    source=str(item.get("source") or ""),
                    summary={
                        "title": item.get("title"),
                        "metadata": item.get("metadata") or {},
                    },
                )
            )

    def _compose_final_answer(self, state: GraphState, subagent_context: SubAgentContext, loop_context: LoopContext) -> None:
        if state.final_answer:
            return
        snippets = [match.text for match in state.reranked_docs[:3]]
        snippets.extend(self._content_snippets(loop_context))
        answer_bundle = build_answer_prompt_bundle(build_session_context(state), subagent_context, loop_context)
        state.context_usage["answer"] = answer_bundle.usage
        state.context_compactions.extend(answer_bundle.compression_events)
        session_context = SessionContext.model_validate(answer_bundle.payload["session"])
        answer_loop = answer_bundle.payload["loop"]
        state.final_answer = self.answering_service.answer_with_context(
            session_context,
            snippets,
            list(answer_loop.get("working_notes") or []),
        )
        consume = getattr(self.answering_service, "consume_issues", None)
        if callable(consume):
            state.issues.extend(consume())

    def _content_snippets(self, loop_context: LoopContext) -> list[str]:
        snippets: list[str] = []
        for step in loop_context.step_history:
            output = step.observation.output if step.observation is not None else {}
            content = output.get("content", []) if isinstance(output, dict) else []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text") or "").strip()
                if text:
                    snippets.append(text[:6000])
        return snippets[:5]

    def _content_tool_name(self, item: dict, output: dict) -> str:
        if item.get("tool_name"):
            return str(item["tool_name"])
        if output.get("tool_name"):
            return str(output["tool_name"])
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        kind = metadata.get("kind")
        if kind == "file":
            return "read_file"
        if kind == "web_search":
            return "web_search"
        if kind == "web_page":
            return "fetch_url"
        return "content"

    def _sync_loop_state(self, state: GraphState, loop_context: LoopContext) -> None:
        state.loop_count = loop_context.step_index
        state.stop_reason = loop_context.stop_reason
        state.evidence_sufficient = loop_context.evidence_sufficient
        state.step_history = list(loop_context.step_history)

    def _debug(self, message: str, payload: dict) -> None:
        if settings.ai_debug_trace:
            logger.warning("%s: %s", message, payload)

    def _debug_contexts(self, session_context, subagent_context: SubAgentContext, loop_context: LoopContext) -> None:
        if not settings.ai_debug_trace:
            return
        payload = {
            "SessionContext": session_context.model_dump(mode="json"),
            "SubAgentContext": subagent_context.model_dump(mode="json"),
            "LoopContext": self._compact_loop_context(loop_context),
        }
        logger.warning("agent loop contexts:\n%s", json.dumps(payload, ensure_ascii=False, indent=2))

    def _compact_loop_context(self, loop_context: LoopContext) -> dict:
        payload = loop_context.model_dump(mode="json")
        compact_history = []
        for step in loop_context.step_history:
            observation = None
            if step.observation is not None:
                output = step.observation.output
                documents = output.get("documents", []) if isinstance(output, dict) else []
                observation = {
                    "tool_name": step.observation.tool_name,
                    "status": step.observation.status,
                    "summary": step.observation.summary,
                    "document_count": len(documents) if isinstance(documents, list) else 0,
                    "document_ids": [
                        item.get("document_id")
                        for item in documents[:3]
                        if isinstance(item, dict) and item.get("document_id")
                    ],
                }
            compact_history.append(
                {
                    "step_index": step.step_index,
                    "decision": step.decision.model_dump(mode="json"),
                    "observation": observation,
                }
            )
        payload["step_history"] = compact_history[-3:]
        return payload

