from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from finpilot.issues import issue_from_safety_finding
from finpilot.models import AgentEvidence, GraphState, ToolCard, ToolInvocation
from finpilot.rag.service import RagKnowledgeService
from finpilot.safety.service import SafetyReviewService


class SearchFinanceKnowledgeArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: StrictStr = Field(min_length=1, max_length=1000)
    limit: StrictInt = Field(default=3, ge=1, le=5)


class TransferMockFundsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_no: StrictStr = Field(min_length=4, max_length=64)
    amount: float = Field(gt=0, le=1_000_000)
    currency: StrictStr = Field(default="CNY", min_length=3, max_length=3)


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
    args_model: type[BaseModel] | None
    executor: Callable[[GraphState, dict[str, Any]], dict[str, Any]]
    risk_level: Literal["low", "medium", "high"] = "low"

    def card(self) -> ToolCard:
        return ToolCard(
            name=self.name,
            description=self.description,
            when_to_use=self.when_to_use,
            arguments=self.arguments,
        )


class ToolRegistry:
    def __init__(self, rag_service: RagKnowledgeService, safety: SafetyReviewService | None = None):
        self.rag_service = rag_service
        self.safety = safety or SafetyReviewService()
        self._tools: dict[str, ToolSpec] = {}
        self.register(
            ToolSpec(
                name="search_finance_knowledge",
                description="Search the finance knowledge base and return relevant rule snippets.",
                when_to_use="Use when the answer needs finance rules, policies, timing, approval, or compliance evidence.",
                arguments={"query": "string", "limit": "integer, optional"},
                args_model=SearchFinanceKnowledgeArgs,
                executor=self._search_finance_knowledge,
            )
        )
        self.register(
            ToolSpec(
                name="transfer_mock_funds",
                description="Mock a funds transfer request for manual safety approval testing; it never moves money.",
                when_to_use=(
                    "Use only when the user explicitly wants to test high-risk operation approval or asks for a mock transfer."
                ),
                arguments={"account_no": "string", "amount": "number", "currency": "string, optional"},
                args_model=TransferMockFundsArgs,
                executor=self._transfer_mock_funds,
                risk_level="high",
            )
        )
        from finpilot.agent.tooling.files import register_file_tools
        from finpilot.agent.tooling.web import register_web_tools

        register_file_tools(self)
        register_web_tools(self)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def list_allowed(self, tool_names: list[str]) -> list[ToolCard]:
        return [self._tools[name].card() for name in tool_names if name in self._tools]

    def list_tools(self) -> list[ToolCard]:
        return [spec.card() for spec in self._tools.values()]

    def invoke(self, state: GraphState, tool_name: str, **parameters: Any) -> ToolInvocation:
        started = time.perf_counter()
        spec = self._tools.get(tool_name)
        reason = parameters.pop("reason", None)
        step_index = parameters.pop("step_index", None)
        if spec is None:
            status = "FAILED"
            output = {"error": f"Unsupported tool: {tool_name}"}
        else:
            review = self.safety.review_tool_call(state, spec, parameters, reason)
            parameters = review.sanitized_payload or parameters
            if review.action != "ALLOW":
                status = "BLOCKED"
                output = self._safety_output("Tool call blocked by safety policy.", review)
            else:
                try:
                    output = spec.executor(state, parameters)
                    status = "SUCCEEDED"
                except Exception as exc:
                    status = "FAILED"
                    output = {"error": str(exc)}
        duration_ms = int((time.perf_counter() - started) * 1000)
        invocation = ToolInvocation(
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            duration_ms=duration_ms,
            output=output,
            step_index=step_index,
            reason=reason,
            observation_summary=self._summarize(tool_name, status, output),
        )
        if status == "SUCCEEDED":
            result_review = self.safety.review_tool_result(state, invocation)
            if result_review.action == "REDACT":
                invocation.output = result_review.sanitized_payload or {}
                invocation.observation_summary = self._summarize(tool_name, status, invocation.output)
                invocation.output.setdefault("issues", []).extend(
                    issue_from_safety_finding(finding).model_dump(mode="json")
                    for finding in result_review.findings
                )
        return invocation

    def _search_finance_knowledge(self, state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
        query = str(parameters.get("query") or state.user_message)
        limit = int(parameters.get("limit") or 3)
        docs = [match.model_dump() for match in self.rag_service.search(query, limit=limit)]
        issues = self._consume_rag_issues()
        output: dict[str, Any] = {"documents": docs}
        if issues:
            output["issues"] = [issue.model_dump(mode="json") for issue in issues]
        return output

    def _transfer_mock_funds(self, state: GraphState, parameters: dict[str, Any]) -> dict[str, Any]:
        return {
            "mock": True,
            "approved_operation": "transfer_mock_funds",
            "user_id": state.user_id,
            "account_no": parameters["account_no"],
            "amount": parameters["amount"],
            "currency": parameters.get("currency", "CNY"),
            "message": "Mock transfer approved for safety approval testing; no money was moved.",
        }

    def _consume_rag_issues(self):
        consume = getattr(self.rag_service, "consume_issues", None)
        if callable(consume):
            return consume()
        return []

    def _summarize(self, tool_name: str, status: str, output: dict[str, Any]) -> str:
        if status == "BLOCKED":
            return str(output.get("error") or f"{tool_name} blocked")
        if status == "FAILED":
            return str(output.get("error") or f"{tool_name} failed")
        documents = output.get("documents")
        if isinstance(documents, list):
            if not documents:
                return "retrieved 0 documents"
            top = documents[0]
            return f"retrieved {len(documents)} documents; top_document={top.get('document_id')}"
        content = output.get("content")
        if isinstance(content, list):
            return f"{tool_name} returned {len(content)} content item(s)"
        return f"{tool_name} succeeded"

    def _safety_output(self, message: str, review) -> dict[str, Any]:
        return {
            "error": message,
            "safety": review.model_dump(mode="json"),
            "issues": [
                issue_from_safety_finding(finding).model_dump(mode="json")
                for finding in review.findings
            ],
        }

