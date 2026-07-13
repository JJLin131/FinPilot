from __future__ import annotations

from finpilot.agent.orchestration import ExecutionPlan
from finpilot.evals.evaluators.base import f1, result
from finpilot.evals.models import EvalObservation, PlanningOrchestrationCase


class PlanningOrchestrationEvaluator:
    def evaluate(self, case: PlanningOrchestrationCase, observation: EvalObservation):
        nodes = observation.plan.get("nodes") or []
        agents = {str(node.get("agent_name")) for node in nodes if node.get("agent_name")}
        node_ids = {str(node.get("node_id")) for node in nodes if node.get("node_id")}
        dependencies = {
            (str(node.get("node_id")), str(dependency))
            for node in nodes
            for dependency in (node.get("depends_on") or [])
        }
        expected_dependencies = {tuple(edge) for edge in case.required_dependencies}
        executable = 1.0
        try:
            ExecutionPlan.model_validate(observation.plan).validate_for_registry(agents, max_nodes=case.max_plan_nodes)
        except Exception:
            executable = 0.0
        metrics = {
            "agent_set_accuracy": float(agents == set(case.expected_agents)),
            "node_f1": f1(set(case.required_nodes), node_ids) if case.required_nodes else 1.0,
            "dependency_f1": f1(expected_dependencies, dependencies),
            "forbidden_node_count": float(len(node_ids & set(case.forbidden_nodes))),
            "plan_executable": executable,
            "step_efficiency": round(len(case.required_nodes) / len(nodes), 4) if nodes else 0.0,
        }
        passed = all(metrics[key] == 1.0 for key in ("agent_set_accuracy", "node_f1", "dependency_f1", "plan_executable")) and metrics["forbidden_node_count"] == 0
        return result(case, observation, metrics, passed)
