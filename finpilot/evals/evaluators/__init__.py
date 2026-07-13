from __future__ import annotations

from finpilot.evals.evaluators.end_to_end_task import EndToEndTaskEvaluator
from finpilot.evals.evaluators.multi_turn_memory import MultiTurnMemoryEvaluator
from finpilot.evals.evaluators.observability_audit import ObservabilityAuditEvaluator
from finpilot.evals.evaluators.performance_cost import PerformanceCostEvaluator
from finpilot.evals.evaluators.planning_orchestration import PlanningOrchestrationEvaluator
from finpilot.evals.evaluators.query_rewrite_reranker import QueryRewriteRerankerEvaluator
from finpilot.evals.evaluators.rag_generation import RagGenerationEvaluator
from finpilot.evals.evaluators.rag_retrieval import RagRetrievalEvaluator
from finpilot.evals.evaluators.resilience_degradation import ResilienceDegradationEvaluator
from finpilot.evals.evaluators.safety_redteam import SafetyRedteamEvaluator
from finpilot.evals.evaluators.tool_calling import ToolCallingEvaluator


DEFAULT_EVALUATORS = {
    "rag_retrieval": RagRetrievalEvaluator,
    "rag_generation": RagGenerationEvaluator,
    "query_rewrite_reranker": QueryRewriteRerankerEvaluator,
    "planning_orchestration": PlanningOrchestrationEvaluator,
    "tool_calling": ToolCallingEvaluator,
    "end_to_end_task": EndToEndTaskEvaluator,
    "multi_turn_memory": MultiTurnMemoryEvaluator,
    "safety_redteam": SafetyRedteamEvaluator,
    "resilience_degradation": ResilienceDegradationEvaluator,
    "performance_cost": PerformanceCostEvaluator,
    "observability_audit": ObservabilityAuditEvaluator,
}


def build_default_evaluators() -> dict[str, object]:
    return {name: evaluator() for name, evaluator in DEFAULT_EVALUATORS.items()}
