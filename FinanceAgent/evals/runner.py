from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from FinanceAgent.evals.ragas import evaluate_ragas_case
from FinanceAgent.models import AgentChatResponse, EvalCase, EvalSuiteResult
from FinanceAgent.observability.audit import AuditStore
from FinanceAgent.observability.langfuse_support import (
    build_run_name,
    ensure_dataset,
    run_experiment,
    score_trace,
    sync_local_datasets,
    upsert_dataset_item,
)


class EvalRunner:
    def __init__(self, agent_service, audit_store: AuditStore, root: Path | None = None):
        self.agent_service = agent_service
        self.audit_store = audit_store
        self.root = root or Path("./evals/datasets")

    def run_suite(self, suite: str) -> EvalSuiteResult:
        sync_local_datasets(self.root)
        cases = list(self._load_suite(suite))
        details = []
        passed = 0
        ensure_dataset(suite, description=f"{suite} evaluation dataset", metadata={"suite": suite})
        experiment_cases: list[dict] = []
        for case in cases:
            response = self.agent_service.chat(case.tenant_id, case.user_id, case.chat_id, case.content)
            upsert_dataset_item(
                suite,
                case_name=case.name,
                payload_input={
                    "tenant_id": case.tenant_id,
                    "user_id": case.user_id,
                    "chat_id": case.chat_id,
                    "content": case.content,
                },
                expected_output={
                    "expected_intent": case.expected_intent,
                    "expected_tool": case.expected_tool,
                    "expected_answer_contains": case.expected_answer_contains,
                    "relevant_document_ids": case.relevant_document_ids,
                    "threat": case.threat,
                },
                metadata={"suite": suite},
            )
            ok, detail = self._evaluate_case(case, response)
            passed += 1 if ok else 0
            ragas_scores = evaluate_ragas_case(case, response)
            detail["ragas"] = ragas_scores
            details.append(detail)
            experiment_cases.append(
                {
                    "input": {
                        "tenant_id": case.tenant_id,
                        "user_id": case.user_id,
                        "chat_id": case.chat_id,
                        "content": case.content,
                    },
                    "expected_output": {
                        "expected_intent": case.expected_intent,
                        "expected_tool": case.expected_tool,
                        "expected_answer_contains": case.expected_answer_contains,
                        "relevant_document_ids": case.relevant_document_ids,
                        "threat": case.threat,
                    },
                    "metadata": {"suite": suite, "case": case.name},
                }
            )
            score_trace(
                response.trace_id,
                name=f"eval.{suite}.passed",
                value=1.0 if ok else 0.0,
                comment=case.name,
                metadata={"suite": suite, "case": case.name},
            )
            for metric_name, metric_value in ragas_scores.items():
                score_trace(
                    response.trace_id,
                    name=f"ragas.{metric_name}",
                    value=metric_value,
                    comment=case.name,
                    metadata={"suite": suite, "case": case.name},
                )
        self._run_langfuse_experiment(suite, experiment_cases)
        result = EvalSuiteResult(
            suite=suite,
            total_cases=len(cases),
            passed_cases=passed,
            score=round(passed / len(cases), 4) if cases else 0.0,
            details=details,
        )
        self.audit_store.record_eval_run(result)
        return result

    def _load_suite(self, suite: str) -> Iterable[EvalCase]:
        path = self.root / f"{suite}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Unknown eval suite: {suite}")
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield EvalCase.model_validate(json.loads(line))

    def _evaluate_case(self, case: EvalCase, response: AgentChatResponse) -> tuple[bool, dict]:
        checks = []
        if case.expected_intent:
            intent_ok = response.route.normalized_intent == case.expected_intent
            checks.append(intent_ok)
            score_trace(
                response.trace_id,
                name="eval.intent_match",
                value=1.0 if intent_ok else 0.0,
                metadata={"expected_intent": case.expected_intent, "actual_intent": response.route.normalized_intent},
            )
        if case.expected_tool:
            tool_names = [tool.tool_name for tool in response.tool_calls or []]
            tool_ok = case.expected_tool in tool_names
            checks.append(tool_ok)
            score_trace(
                response.trace_id,
                name="eval.tool_match",
                value=1.0 if tool_ok else 0.0,
                metadata={"expected_tool": case.expected_tool, "actual_tools": tool_names},
            )
        if case.expected_answer_contains:
            answer_ok = case.expected_answer_contains in response.answer
            checks.append(answer_ok)
            score_trace(
                response.trace_id,
                name="eval.answer_contains",
                value=1.0 if answer_ok else 0.0,
                metadata={"expected_text": case.expected_answer_contains},
            )
        if case.relevant_document_ids:
            actual = {entry.summary.get("document_id") for entry in response.evidence if "document_id" in entry.summary}
            retrieval_ok = any(document_id in actual for document_id in case.relevant_document_ids)
            checks.append(retrieval_ok)
            score_trace(
                response.trace_id,
                name="eval.retrieval_hit",
                value=1.0 if retrieval_ok else 0.0,
                metadata={"expected_docs": case.relevant_document_ids, "actual_docs": list(actual)},
            )
        if case.threat:
            safety_ok = response.route.normalized_intent == "UNKNOWN"
            checks.append(safety_ok)
            score_trace(
                response.trace_id,
                name="eval.safety_blocked",
                value=1.0 if safety_ok else 0.0,
                metadata={"threat": case.threat},
            )
        passed = all(checks) if checks else True
        return passed, {
            "case": case.name,
            "passed": passed,
            "route": response.route.normalized_intent,
            "answer": response.answer,
        }

    def _run_langfuse_experiment(self, suite: str, cases: list[dict]) -> None:
        if not cases:
            return

        def task(item):
            payload = item["input"]
            response = self.agent_service.chat(
                payload["tenant_id"],
                payload["user_id"],
                payload["chat_id"],
                payload["content"],
            )
            return {
                "actual_intent": response.route.normalized_intent,
                "answer": response.answer,
                "tool_calls": [tool.tool_name for tool in response.tool_calls or []],
                "trace_id": response.trace_id,
            }

        def evaluator(item, output):
            expected = item["expected_output"]
            actual_intent = output.get("actual_intent")
            expected_intent = expected.get("expected_intent")
            score = 1.0 if not expected_intent or actual_intent == expected_intent else 0.0
            return {"name": "intent_match", "value": score, "comment": f"expected={expected_intent}, actual={actual_intent}"}

        def tool_evaluator(item, output):
            expected_tool = item["expected_output"].get("expected_tool")
            tools = output.get("tool_calls", [])
            score = 1.0 if not expected_tool or expected_tool in tools else 0.0
            return {"name": "tool_match", "value": score, "comment": f"expected={expected_tool}, actual={tools}"}

        run_experiment(
            suite,
            run_name=build_run_name(suite),
            description=f"Automated evaluation run for suite {suite}",
            cases=cases,
            task=task,
            evaluators=[evaluator, tool_evaluator],
            metadata={"suite": suite, "kind": "regression"},
        )
