from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from finpilot.safety.models import SafetyFinding


class FinPilotChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.:@-]+$")
    chat_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")
    content: str = Field(min_length=1, max_length=8000)


class AgentEvidence(BaseModel):
    tool_name: str
    period: str | None = None
    source: str
    summary: dict[str, Any] = Field(default_factory=dict)


class SubAgentResult(BaseModel):
    node_id: str
    agent_name: str
    task: str
    status: Literal["SUCCEEDED", "FAILED", "BLOCKED", "SKIPPED"]
    summary: str = ""
    output: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: list[dict[str, Any]] = Field(default_factory=list)
    failure_reason: str | None = None
    raw_evidence: list[dict[str, Any]] = Field(default_factory=list, exclude=True, repr=False)
    runtime_data: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class AgentIssue(BaseModel):
    code: str
    component: str
    message: str
    severity: Literal["warning", "error"] = "error"
    retryable: bool = False
    detail: str | None = None


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
    status: Literal["SUCCEEDED", "FAILED", "BLOCKED"] = "SUCCEEDED"
    duration_ms: int = 0
    output: dict[str, Any] = Field(default_factory=dict)
    step_index: int | None = None
    reason: str | None = None
    observation_summary: str | None = None


class ToolCard(BaseModel):
    name: str
    description: str
    when_to_use: str
    arguments: dict[str, str] = Field(default_factory=dict)


class SessionContext(BaseModel):
    user_id: str
    chat_id: str
    memory_id: str
    user_message: str
    normalized_intent: str
    target_agent: str
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    structured_memory: dict[str, Any] = Field(default_factory=dict)
    semantic_memory: list[dict[str, Any]] = Field(default_factory=list)
    subagent_results: list[SubAgentResult] = Field(default_factory=list)


class SubAgentContext(BaseModel):
    agent_name: str
    role: str
    goal: str
    constraints: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    allowed_tools: list[ToolCard] = Field(default_factory=list)
    max_steps: int = 3
    context_policy: str = "agent_default"
    assigned_task: str = ""


class AgentDecision(BaseModel):
    decision: Literal["act", "answer", "fallback", "stop"]
    reason: str = ""
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)
    enough_information: bool = False
    draft_answer: str | None = None

    @field_validator("tool_args", mode="before")
    @classmethod
    def default_null_tool_args(cls, value):
        return {} if value is None else value


class ToolObservation(BaseModel):
    tool_name: str
    status: Literal["SUCCEEDED", "FAILED", "BLOCKED"]
    summary: str
    output: dict[str, Any] = Field(default_factory=dict)


class LoopStepRecord(BaseModel):
    step_index: int
    decision: AgentDecision
    observation: ToolObservation | None = None


class LoopContext(BaseModel):
    step_index: int = 0
    max_steps: int = 3
    step_history: list[LoopStepRecord] = Field(default_factory=list)
    working_notes: list[str] = Field(default_factory=list)
    evidence_sufficient: bool = False
    stop_reason: str | None = None
    draft_answer: str | None = None


class AgentChatResponse(BaseModel):
    request_id: str
    trace_id: str
    domain: str
    status: str
    answer: str
    evidence: list[AgentEvidence] = Field(default_factory=list)
    route: RouteDecision
    issues: list[AgentIssue] = Field(default_factory=list)
    safety_findings: list[SafetyFinding] = Field(default_factory=list)
    route_debug: dict[str, Any] | None = None
    retrieval_debug: dict[str, Any] | None = None
    tool_calls: list[ToolInvocation] | None = None


class RagMatch(BaseModel):
    document_id: str
    title: str
    source: str
    text: str
    score: float
    distance: float | None = None


class EvalCase(BaseModel):
    suite: str
    name: str
    user_id: str = "eval-user"
    chat_id: str
    content: str
    expected_intent: str | None = None
    expected_agent: str | None = None
    expected_status: str | None = None
    expected_tool: str | None = None
    expected_tool_status: str | None = None
    expected_tool_args: dict[str, Any] = Field(default_factory=dict)
    expected_answer_contains: str | None = None
    relevant_document_ids: list[str] = Field(default_factory=list)
    expected_evidence_tool: str | None = None
    expected_reranked_document_ids: list[str] = Field(default_factory=list)
    threat: str | None = None
    expected_safety_action: str | None = None
    expected_safety_code: str | None = None
    requires_approval: bool = False
    privacy_forbidden_fields: list[str] = Field(default_factory=list)
    metric_tags: list[str] = Field(default_factory=list)


class EvalSuiteResult(BaseModel):
    suite: str
    total_cases: int
    passed_cases: int
    score: float
    details: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class GraphState(BaseModel):
    request_id: str
    trace_id: str
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
    issues: list[AgentIssue] = Field(default_factory=list)
    safety_findings: list[SafetyFinding] = Field(default_factory=list)
    retrieved_docs: list[RagMatch] = Field(default_factory=list)
    reranked_docs: list[RagMatch] = Field(default_factory=list)
    tool_invocations: list[ToolInvocation] = Field(default_factory=list)
    evidence: list[AgentEvidence] = Field(default_factory=list)
    recent_messages: list[dict[str, Any]] = Field(default_factory=list)
    structured_memory: dict[str, Any] = Field(default_factory=dict)
    semantic_memory: list[dict[str, Any]] = Field(default_factory=list)
    subagent_results: list[SubAgentResult] = Field(default_factory=list)
    answer_evidence: list[dict[str, Any]] = Field(default_factory=list)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    global_context: dict[str, Any] = Field(default_factory=dict)
    loop_count: int = 0
    stop_reason: str | None = None
    evidence_sufficient: bool = False
    step_history: list[LoopStepRecord] = Field(default_factory=list)
    final_answer: str = ""
    latency_breakdown: dict[str, float] = Field(default_factory=dict)
    scores: dict[str, float] = Field(default_factory=dict)
    context_usage: dict[str, Any] = Field(default_factory=dict)
    context_compactions: list[dict[str, Any]] = Field(default_factory=list)
    safety_approval_decisions: list[dict[str, Any]] = Field(default_factory=list)

