from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from FinanceAgent.config import settings
from FinanceAgent.models import EvalSuiteResult, GraphState, RouteDecision, ToolInvocation


class AuditStore(ABC):
    @abstractmethod
    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_eval_run(self, result: EvalSuiteResult) -> None:
        raise NotImplementedError


class FileAuditStore(AuditStore):
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        self._append(
            "agent_tool_audit.jsonl",
            {
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "tenant_id": state.tenant_id,
                "user_id": state.user_id,
                "domain": "FINANCE",
                "tool_name": invocation.tool_name,
                "parameter_summary": invocation.parameters,
                "status": invocation.status,
                "duration_ms": invocation.duration_ms,
            },
        )

    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        self._append(
            "unknown_intent_audit.jsonl",
            {
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "tenant_id": state.tenant_id,
                "user_id": state.user_id,
                "domain": "FINANCE",
                "user_message": state.user_message,
                "raw_intent_json": decision.raw_intent_json,
                "classifier_intent": decision.classifier_intent,
                "embedding_top1_intent": decision.embedding_top1_intent,
                "embedding_top2_intent": decision.embedding_top2_intent,
                "reason": decision.reason,
                "semantic_score": decision.semantic_score,
                "margin_score": decision.margin_score,
                "agreement_score": decision.agreement_score,
                "final_confidence": decision.confidence,
                "fallback_cause": decision.fallback_cause,
            },
        )

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        self._append("agent_eval_run.jsonl", result.model_dump())

    def _append(self, file_name: str, payload: dict[str, Any]) -> None:
        line = {
            **payload,
            "created_at": datetime.now(UTC).isoformat(),
        }
        with (self.root / file_name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


class MySqlAuditStore(AuditStore):
    def __init__(self) -> None:
        try:
            import pymysql  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime guard
            raise RuntimeError("Install pymysql in the project virtual environment to use MySQL audits.") from exc
        self.pymysql = pymysql

    def _connect(self):
        return self.pymysql.connect(
            host=settings.mysql_host,
            port=settings.mysql_port,
            user=settings.mysql_user,
            password=settings.mysql_password,
            database=settings.mysql_database,
            charset="utf8mb4",
            autocommit=True,
        )

    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_tool_audit
                    (request_id, tenant_id, user_id, domain, tool_name, parameter_summary, status, duration_ms, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.tenant_id,
                        state.user_id,
                        "FINANCE",
                        invocation.tool_name,
                        json.dumps(invocation.parameters, ensure_ascii=False),
                        invocation.status,
                        invocation.duration_ms,
                    ),
                )

    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into unknown_intent_audit
                    (request_id, tenant_id, user_id, domain, user_message, raw_intent_json, classifier_intent,
                     embedding_top1_intent, embedding_top2_intent, reason, semantic_score, margin_score,
                     agreement_score, final_confidence, fallback_cause, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.tenant_id,
                        state.user_id,
                        "FINANCE",
                        state.user_message,
                        decision.raw_intent_json,
                        decision.classifier_intent,
                        decision.embedding_top1_intent,
                        decision.embedding_top2_intent,
                        decision.reason,
                        decision.semantic_score,
                        decision.margin_score,
                        decision.agreement_score,
                        decision.confidence,
                        decision.fallback_cause,
                    ),
                )

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                try:
                    cursor.execute(
                        """
                        insert into agent_eval_run
                        (suite_name, total_cases, passed_cases, score, details_json, created_at)
                        values (%s, %s, %s, %s, %s, current_timestamp(6))
                        """,
                        (
                            result.suite,
                            result.total_cases,
                            result.passed_cases,
                            result.score,
                            json.dumps(result.details, ensure_ascii=False),
                        ),
                    )
                except Exception:
                    return


def build_audit_store() -> AuditStore:
    if settings.audit_backend.lower() == "mysql":
        try:
            return MySqlAuditStore()
        except Exception:
            return FileAuditStore(settings.audit_dir)
    return FileAuditStore(settings.audit_dir)
