from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from finpilot.evals.models import BaseEvalCase, CASE_MODELS


EVALUATION_SUITES = tuple(CASE_MODELS)


@dataclass(frozen=True)
class SuiteDefinition:
    name: str
    case_model: type[BaseEvalCase]
    required_environment: tuple[str, ...] = ()


class SuiteRegistry:
    def __init__(self, definitions: list[SuiteDefinition]):
        self._definitions = {definition.name: definition for definition in definitions}
        if len(self._definitions) != len(definitions):
            raise ValueError("evaluation suite names must be unique")

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definition(self, suite: str) -> SuiteDefinition:
        try:
            return self._definitions[suite]
        except KeyError as exc:
            raise ValueError(f"Unknown evaluation suite: {suite}") from exc

    def parse(self, payload: dict[str, Any]) -> BaseEvalCase:
        suite = str(payload.get("suite") or "")
        return self.definition(suite).case_model.model_validate(payload)


_REQUIREMENTS = {
    "rag_retrieval": ("runtime_service", "bm25", "chroma", "embedding", "reranker"),
    "rag_generation": ("runtime_service", "llm", "embedding", "ragas"),
    "query_rewrite_reranker": ("runtime_service", "bm25", "chroma", "embedding", "query_rewriter", "reranker"),
    "planning_orchestration": ("runtime_service", "planner"),
    "tool_calling": ("runtime_service", "agent"),
    "end_to_end_task": ("runtime_service", "agent", "mysql"),
    "multi_turn_memory": ("runtime_service", "mysql", "chroma"),
    "safety_redteam": ("runtime_service", "agent"),
    "resilience_degradation": ("controlled_backend",),
    "performance_cost": ("runtime_service", "agent", "cost_pricing"),
    "observability_audit": ("runtime_service", "langfuse", "otel", "audit"),
}


DEFAULT_SUITE_REGISTRY = SuiteRegistry(
    [
        SuiteDefinition(name=name, case_model=model, required_environment=_REQUIREMENTS[name])
        for name, model in CASE_MODELS.items()
    ]
)
