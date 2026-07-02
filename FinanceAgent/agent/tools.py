from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from FinanceAgent.models import AgentEvidence, GraphState, ToolCard, ToolInvocation
from FinanceAgent.rag.service import RagKnowledgeService


@dataclass
class ToolResult:
    answer_fragment: str
    evidence: AgentEvidence | None = None
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    when_to_use: str
    arguments: dict[str, str]
    executor: Callable[[GraphState, dict[str, Any]], dict[str, Any]]

    def card(self) -> ToolCard:
        return ToolCard(
            name=self.name,
            description=self.description,
            when_to_use=self.when_to_use,
            arguments=self.arguments,
        )


class ToolRegistry:
    def __init__(self, rag_service: RagKnowledgeService):
        self.rag_service = rag_service
        self._tools: dict[str, ToolSpec] = {}
        self.register(
            ToolSpec(
                name="search_finance_knowledge",
                description="Search the finance knowledge base and return relevant rule snippets.",
                when_to_use="Use when the answer needs finance rules, policies, timing, approval, or compliance evidence.",
                arguments={"query": "string", "limit": "integer, optional"},
                executor=self._search_finance_knowledge,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_allowed(self, tool_names: list[str]) -> list[ToolCard]:
        return [self._tools[name].card() for name in tool_names if name in self._tools]

    def invoke(self, state: GraphState, tool_name: str, **parameters: Any) -> ToolInvocation:
        started = time.perf_counter()
        spec = self._tools.get(tool_name)
        reason = parameters.pop("reason", None)
        step_index = parameters.pop("step_index", None)
        if spec is None:
            status = "FAILED"
            output = {"error": f"Unsupported tool: {tool_name}"}
        else:
            try:
                output = spec.executor(state, parameters)
                status = "SUCCEEDED"
            except Exception as exc:
                status = "FAILED"
                output = {"error": str(exc)}
        duration_ms = int((time.perf_counter() - started) * 1000)
        return ToolInvocation(
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            duration_ms=duration_ms,
            output=output,
            step_index=step_index,
            reason=reason,
            observation_summary=self._summarize(tool_name, status, output),
        )

    def _search_finance_knowledge(self, state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
        tenant_id = state.tenant_id if state.tenant_id else "__GLOBAL__"
        query = str(parameters.get("query") or state.user_message)
        limit = int(parameters.get("limit") or 3)
        docs = [match.model_dump() for match in self.rag_service.search(query, tenant_id=tenant_id, limit=limit)]
        return {"documents": docs}

    def _summarize(self, tool_name: str, status: str, output: dict[str, Any]) -> str:
        if status == "FAILED":
            return str(output.get("error") or f"{tool_name} failed")
        documents = output.get("documents")
        if isinstance(documents, list):
            if not documents:
                return "retrieved 0 documents"
            top = documents[0]
            return f"retrieved {len(documents)} documents; top_document={top.get('document_id')}"
        return f"{tool_name} succeeded"
