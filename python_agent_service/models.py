from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FinanceChatRequest(BaseModel):
    tenant_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


class AgentEvidence(BaseModel):
    tool_name: str
    period: str | None = None
    source: str
    summary: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(BaseModel):
    raw_intent_json: str
    normalized_intent: str
    reason: str
    confidence: float
    valid: bool
    target_agent: str
    classifier_intent: str
    embedding_top1_intent: str | None = None
    embedding_top2_intent: str | None = None
    fallback_cause: str = "NONE"
    semantic_score: float = 0.0
    margin_score: float = 0.0
    agreement_score: float = 0.0


class ToolInvocation(BaseModel):
    tool_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    status: Literal["SUCCEEDED", "FAILED"] = "SUCCEEDED"
    duration_ms: int = 0
    output: dict[str, Any] = Field(default_factory=dict)


class AgentChatResponse(BaseModel):
    request_id: str
    trace_id: str
    domain: str
    status: str
    answer: str
    evidence: list[AgentEvidence] = Field(default_factory=list)
    route: RouteDecision
    route_debug: dict[str, Any] | None = None
    retrieval_debug: dict[str, Any] | None = None
    tool_calls: list[ToolInvocation] | None = None


class RagMatch(BaseModel):
    document_id: str
    title: str
    source: str
    text: str
    score: float


class EvalCase(BaseModel):
    suite: str
    name: str
    tenant_id: str = "tenant-a"
    user_id: str = "eval-user"
    chat_id: str
    content: str
    expected_intent: str | None = None
    expected_tool: str | None = None
    expected_answer_contains: str | None = None
    relevant_document_ids: list[str] = Field(default_factory=list)
    threat: str | None = None


class EvalSuiteResult(BaseModel):
    suite: str
    total_cases: int
    passed_cases: int
    score: float
    details: list[dict[str, Any]] = Field(default_factory=list)


class GraphState(BaseModel):
    request_id: str
    trace_id: str
    tenant_id: str
    user_id: str
    chat_id: str
    memory_id: str
    user_message: str
    normalized_intent: str = "UNKNOWN"
    classifier_intent: str = "UNKNOWN"
    embedding_top1: str | None = None
    embedding_top2: str | None = None
    raw_intent_json: str = ""
    fallback_cause: str = "NONE"
    target_agent: str = "UNSUPPORTED"
    route_reason: str = ""
    route_confidence: float = 0.0
    semantic_score: float = 0.0
    margin_score: float = 0.0
    agreement_score: float = 0.0
    retrieved_docs: list[RagMatch] = Field(default_factory=list)
    reranked_docs: list[RagMatch] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    workflow_memory: dict[str, Any] | None = None
    system_operation_hints: list[str] = Field(default_factory=list)
    final_answer: str = ""
    latency_breakdown: dict[str, float] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
