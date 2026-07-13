from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictEvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EvalStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    ENV_UNAVAILABLE = "ENV_UNAVAILABLE"
    FIXTURE_UNAVAILABLE = "FIXTURE_UNAVAILABLE"
    EVALUATOR_ERROR = "EVALUATOR_ERROR"


class BaseEvalCase(StrictEvalModel):
    suite: str
    case_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    execution_mode: Literal["live", "controlled"] = "controlled"
    severity: Literal["critical", "high", "medium", "low"] = "medium"
    timeout_seconds: int = Field(default=30, gt=0, le=600)
    repetitions: int = Field(default=1, gt=0, le=100)
    fixtures: dict[str, Any] = Field(default_factory=dict)


class RagRetrievalCase(BaseEvalCase):
    suite: Literal["rag_retrieval"]
    query: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(default_factory=list)
    forbidden_document_ids: list[str] = Field(default_factory=list)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5])
    expected_no_answer: bool = False

    @field_validator("k_values")
    @classmethod
    def validate_k_values(cls, values: list[int]) -> list[int]:
        if not values or any(value <= 0 for value in values):
            raise ValueError("k_values must contain positive integers")
        if values != sorted(set(values)):
            raise ValueError("k_values must be unique and sorted")
        return values


class ReferenceContext(StrictEvalModel):
    document_id: str
    text: str


class RagGenerationCase(BaseEvalCase):
    suite: Literal["rag_generation"]
    question: str = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    contexts: list[ReferenceContext] = Field(min_length=1)
    required_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    expected_citations: list[str] = Field(default_factory=list)


class QueryRewriteMmrRerankerCase(BaseEvalCase):
    suite: Literal["query_rewrite_mmr_reranker"]
    query: str = Field(min_length=1)
    reference_rewrite: str = Field(min_length=1)
    relevant_document_ids: list[str] = Field(default_factory=list)
    forbidden_document_ids: list[str] = Field(default_factory=list)
    subtopics: list[str] = Field(default_factory=list)
    candidate_documents: list[ReferenceContext] = Field(default_factory=list)


class PlanningOrchestrationCase(BaseEvalCase):
    suite: Literal["planning_orchestration"]
    user_message: str = Field(min_length=1)
    expected_agents: list[str] = Field(default_factory=list)
    required_nodes: list[str] = Field(default_factory=list)
    required_dependencies: list[list[str]] = Field(default_factory=list)
    forbidden_nodes: list[str] = Field(default_factory=list)
    max_plan_nodes: int = Field(default=3, gt=0)
    expected_parallel_groups: list[list[str]] = Field(default_factory=list)

    @field_validator("required_dependencies")
    @classmethod
    def validate_dependency_edges(cls, values: list[list[str]]) -> list[list[str]]:
        if any(len(edge) != 2 for edge in values):
            raise ValueError("each dependency must be [node, dependency]")
        return values


class ExpectedToolCall(StrictEvalModel):
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallingCase(BaseEvalCase):
    suite: Literal["tool_calling"]
    user_message: str = Field(min_length=1)
    expected_calls: list[ExpectedToolCall] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class ConversationTurn(StrictEvalModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class EndToEndTaskCase(BaseEvalCase):
    suite: Literal["end_to_end_task"]
    turns: list[ConversationTurn] = Field(min_length=1)
    expected_final_state: dict[str, Any] = Field(default_factory=dict)
    required_constraints: list[str] = Field(default_factory=list)
    forbidden_states: list[dict[str, Any]] = Field(default_factory=list)


class MultiTurnMemoryCase(BaseEvalCase):
    suite: Literal["multi_turn_memory"]
    turns: list[ConversationTurn] = Field(min_length=2)
    expected_memories: dict[str, Any] = Field(default_factory=dict)
    forbidden_memories: list[str] = Field(default_factory=list)
    expected_forget_keys: list[str] = Field(default_factory=list)
    isolation_user_id: str | None = None


class SafetyRedteamCase(BaseEvalCase):
    suite: Literal["safety_redteam"]
    prompt: str = Field(min_length=1)
    attack_type: str = Field(min_length=1)
    expected_action: Literal["ALLOW", "BLOCK", "REDACT", "REQUIRE_APPROVAL"]
    expected_code: str | None = None
    expected_side_effect_count: int = Field(default=0, ge=0)
    legitimate: bool = False


class FaultSpec(StrictEvalModel):
    component: str = Field(min_length=1)
    behavior: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ResilienceDegradationCase(BaseEvalCase):
    suite: Literal["resilience_degradation"]
    prompt: str = Field(min_length=1)
    fault: FaultSpec
    expected_status: str = Field(min_length=1)
    max_attempts: int = Field(default=1, gt=0)
    expected_side_effect_count: int = Field(default=0, ge=0)


class PerformanceCostCase(BaseEvalCase):
    suite: Literal["performance_cost"]
    prompt: str = Field(min_length=1)
    warmups: int = Field(default=0, ge=0, le=20)
    max_p95_ms: float = Field(gt=0)
    max_total_tokens: int = Field(gt=0)
    max_cost: float | None = Field(default=None, ge=0)


class ObservabilityAuditCase(BaseEvalCase):
    suite: Literal["observability_audit"]
    prompt: str = Field(min_length=1)
    required_spans: list[str] = Field(default_factory=list)
    required_scores: list[str] = Field(default_factory=list)
    required_audit_events: list[str] = Field(default_factory=list)
    forbidden_fields: list[str] = Field(default_factory=list)


CASE_MODELS: dict[str, type[BaseEvalCase]] = {
    "rag_retrieval": RagRetrievalCase,
    "rag_generation": RagGenerationCase,
    "query_rewrite_mmr_reranker": QueryRewriteMmrRerankerCase,
    "planning_orchestration": PlanningOrchestrationCase,
    "tool_calling": ToolCallingCase,
    "end_to_end_task": EndToEndTaskCase,
    "multi_turn_memory": MultiTurnMemoryCase,
    "safety_redteam": SafetyRedteamCase,
    "resilience_degradation": ResilienceDegradationCase,
    "performance_cost": PerformanceCostCase,
    "observability_audit": ObservabilityAuditCase,
}


def parse_eval_case(payload: dict[str, Any]) -> BaseEvalCase:
    suite = str(payload.get("suite") or "")
    model = CASE_MODELS.get(suite)
    if model is None:
        raise ValueError(f"Unknown evaluation suite: {suite}")
    return model.model_validate(payload)


class EvalFailure(StrictEvalModel):
    code: str
    message: str
    detail: dict[str, Any] = Field(default_factory=dict)


class EvalCaseResult(StrictEvalModel):
    suite: str
    case_id: str
    status: EvalStatus
    passed: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    failure: EvalFailure | None = None
    raw_output: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    token_usage: dict[str, int] = Field(default_factory=dict)


class EvalSuiteResult(StrictEvalModel):
    suite: str
    mode: Literal["smoke", "regression", "release"]
    status: EvalStatus
    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    results: list[EvalCaseResult] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    slices: dict[str, Any] = Field(default_factory=dict)
    environment: dict[str, Any] = Field(default_factory=dict)


class EvalRunResult(StrictEvalModel):
    mode: Literal["smoke", "regression", "release"]
    status: EvalStatus
    suites: list[EvalSuiteResult] = Field(default_factory=list)
    release_gate: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
