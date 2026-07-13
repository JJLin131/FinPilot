from __future__ import annotations

from pydantic import BaseModel, Field

from finpilot.evals.models import (
    BaseEvalCase,
    PlanningOrchestrationCase,
    RagRetrievalCase,
    ResilienceDegradationCase,
    ToolCallingCase,
)


class CoverageInventory(BaseModel):
    agents: set[str] = Field(default_factory=set)
    tools: set[str] = Field(default_factory=set)
    document_ids: set[str] = Field(default_factory=set)
    fault_components: set[str] = Field(default_factory=set)


class CoverageReport(BaseModel):
    complete: bool
    missing: dict[str, list[str]] = Field(default_factory=dict)


def validate_coverage(cases: list[BaseEvalCase], inventory: CoverageInventory) -> CoverageReport:
    covered_agents: set[str] = set()
    covered_tools: set[str] = set()
    covered_documents: set[str] = set()
    covered_faults: set[str] = set()
    for case in cases:
        if isinstance(case, PlanningOrchestrationCase):
            covered_agents.update(case.expected_agents)
        elif isinstance(case, ToolCallingCase):
            covered_tools.update(call.tool_name for call in case.expected_calls)
        elif isinstance(case, RagRetrievalCase):
            covered_documents.update(case.relevant_document_ids)
        elif isinstance(case, ResilienceDegradationCase):
            covered_faults.add(case.fault.component)
    missing = {
        "agents": sorted(inventory.agents - covered_agents),
        "tools": sorted(inventory.tools - covered_tools),
        "document_ids": sorted(inventory.document_ids - covered_documents),
        "fault_components": sorted(inventory.fault_components - covered_faults),
    }
    missing = {key: values for key, values in missing.items() if values}
    return CoverageReport(complete=not missing, missing=missing)
