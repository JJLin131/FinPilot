from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from finpilot.config import settings
from finpilot.models import AgentIssue, EvalSuiteResult, GraphState, RouteDecision, ToolInvocation
from finpilot.mysql import connect_runtime_mysql
from finpilot.safety.models import SafetyFinding
from finpilot.safety.redaction import redact_value


class AuditStore(ABC):
    @abstractmethod
    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_unknown_intent(self, state: GraphState, decision: RouteDecision) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        raise NotImplementedError

    @abstractmethod
    def record_eval_run(self, result: EvalSuiteResult) -> None:
        raise NotImplementedError

    def record_safety_finding(self, state: GraphState, finding: SafetyFinding) -> None:
        pass

    def record_approval_decision(self, state: GraphState, decision: dict[str, Any]) -> None:
        pass


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
                "user_id": state.user_id,
                "domain": "FINANCE",
                "tool_name": invocation.tool_name,
                "parameter_summary": redact_value(invocation.parameters),
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

    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        self._append(
            "agent_issue_audit.jsonl",
            {
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "user_id": state.user_id,
                "domain": "FINANCE",
                "code": issue.code,
                "component": issue.component,
                "message": issue.message,
                "severity": issue.severity,
                "retryable": issue.retryable,
                "detail": issue.detail,
                "route_intent": state.normalized_intent,
                "fallback_cause": state.fallback_cause,
            },
        )

    def record_eval_run(self, result: EvalSuiteResult) -> None:
        self._append("agent_eval_run.jsonl", result.model_dump())

    def record_safety_finding(self, state: GraphState, finding: SafetyFinding) -> None:
        self._append(
            "agent_safety_finding_audit.jsonl",
            {
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "user_id": state.user_id,
                "domain": "FINANCE",
                "code": finding.code,
                "reviewer": finding.reviewer,
                "action": finding.action,
                "message": finding.message,
                "severity": finding.severity,
                "detail": redact_value(finding.detail),
                "route_intent": state.normalized_intent,
            },
        )

    def record_approval_decision(self, state: GraphState, decision: dict[str, Any]) -> None:
        self._append(
            "agent_safety_approval_audit.jsonl",
            {
                "request_id": state.request_id,
                "trace_id": state.trace_id,
                "user_id": state.user_id,
                "domain": "FINANCE",
                **redact_value(decision),
            },
        )

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
        self._ensure_schema()

    def _connect(self):
        return connect_runtime_mysql(pymysql_module=self.pymysql)

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists agent_tool_audit (
                        id bigint not null auto_increment primary key,
                        request_id varchar(64) not null,
                        user_id varchar(64) not null,
                        domain varchar(32) not null,
                        tool_name varchar(128) not null,
                        parameter_summary text null,
                        status varchar(32) not null,
                        duration_ms bigint not null,
                        created_at datetime(6) not null,
                        key idx_agent_tool_audit_user_created (user_id, created_at),
                        key idx_agent_tool_audit_request (request_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists unknown_intent_audit (
                        id bigint not null auto_increment primary key,
                        request_id varchar(64) not null,
                        user_id varchar(64) not null,
                        domain varchar(32) not null,
                        user_message text not null,
                        raw_intent_json text null,
                        classifier_intent varchar(64) not null,
                        embedding_top1_intent varchar(64) null,
                        embedding_top2_intent varchar(64) null,
                        reason text not null,
                        semantic_score double not null,
                        margin_score double not null,
                        agreement_score double not null,
                        final_confidence double not null,
                        fallback_cause varchar(64) not null,
                        created_at datetime(6) not null,
                        key idx_unknown_intent_audit_created (created_at),
                        key idx_unknown_intent_audit_request (request_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists agent_eval_run (
                        id bigint not null auto_increment primary key,
                        suite_name varchar(128) not null,
                        total_cases int not null,
                        passed_cases int not null,
                        score double not null,
                        details_json longtext null,
                        created_at datetime(6) not null,
                        key idx_agent_eval_run_suite_created (suite_name, created_at)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists agent_issue_audit (
                        id bigint not null auto_increment primary key,
                        request_id varchar(64) not null,
                        trace_id varchar(64) null,
                        user_id varchar(64) not null,
                        domain varchar(32) not null,
                        code varchar(96) not null,
                        component varchar(128) not null,
                        message text not null,
                        severity varchar(32) not null,
                        retryable tinyint(1) not null,
                        detail text null,
                        route_intent varchar(64) null,
                        fallback_cause varchar(96) null,
                        created_at datetime(6) not null,
                        key idx_agent_issue_audit_created (created_at),
                        key idx_agent_issue_audit_request (request_id),
                        key idx_agent_issue_audit_code (code)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists agent_trace_feedback (
                        id bigint not null auto_increment primary key,
                        trace_id varchar(64) not null,
                        request_id varchar(64) not null,
                        rating int not null,
                        comment text null,
                        created_at datetime(6) not null,
                        key idx_agent_trace_feedback_trace (trace_id),
                        key idx_agent_trace_feedback_request (request_id)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists agent_safety_finding_audit (
                        id bigint not null auto_increment primary key,
                        request_id varchar(64) not null,
                        trace_id varchar(64) null,
                        user_id varchar(64) not null,
                        domain varchar(32) not null,
                        code varchar(96) not null,
                        reviewer varchar(96) not null,
                        action varchar(32) not null,
                        message text not null,
                        severity varchar(32) not null,
                        detail_json longtext null,
                        route_intent varchar(64) null,
                        created_at datetime(6) not null,
                        key idx_agent_safety_finding_created (created_at),
                        key idx_agent_safety_finding_request (request_id),
                        key idx_agent_safety_finding_code (code)
                    )
                    """
                )
                cursor.execute(
                    """
                    create table if not exists agent_safety_approval_audit (
                        id bigint not null auto_increment primary key,
                        request_id varchar(64) not null,
                        trace_id varchar(64) null,
                        user_id varchar(64) not null,
                        domain varchar(32) not null,
                        tool_name varchar(128) not null,
                        finding_code varchar(96) not null,
                        approved tinyint(1) not null,
                        scope varchar(32) not null,
                        reused tinyint(1) not null,
                        approval_key varchar(64) null,
                        expires_at varchar(64) null,
                        created_at datetime(6) not null,
                        key idx_agent_safety_approval_created (created_at),
                        key idx_agent_safety_approval_request (request_id),
                        key idx_agent_safety_approval_tool (tool_name)
                    )
                    """
                )

    def record_tool(self, state: GraphState, invocation: ToolInvocation) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_tool_audit
                    (request_id, user_id, domain, tool_name, parameter_summary, status, duration_ms, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.user_id,
                        "FINANCE",
                        invocation.tool_name,
                        json.dumps(redact_value(invocation.parameters), ensure_ascii=False),
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
                    (request_id, user_id, domain, user_message, raw_intent_json, classifier_intent,
                     embedding_top1_intent, embedding_top2_intent, reason, semantic_score, margin_score,
                     agreement_score, final_confidence, fallback_cause, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
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

    def record_issue(self, state: GraphState, issue: AgentIssue) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_issue_audit
                    (request_id, trace_id, user_id, domain, code, component, message, severity, retryable, detail,
                     route_intent, fallback_cause, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.trace_id,
                        state.user_id,
                        "FINANCE",
                        issue.code,
                        issue.component,
                        issue.message,
                        issue.severity,
                        1 if issue.retryable else 0,
                        issue.detail,
                        state.normalized_intent,
                        state.fallback_cause,
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

    def record_safety_finding(self, state: GraphState, finding: SafetyFinding) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_safety_finding_audit
                    (request_id, trace_id, user_id, domain, code, reviewer, action, message, severity, detail_json,
                     route_intent, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.trace_id,
                        state.user_id,
                        "FINANCE",
                        finding.code,
                        finding.reviewer,
                        finding.action,
                        finding.message,
                        finding.severity,
                        json.dumps(redact_value(finding.detail), ensure_ascii=False),
                        state.normalized_intent,
                    ),
                )

    def record_approval_decision(self, state: GraphState, decision: dict[str, Any]) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    insert into agent_safety_approval_audit
                    (request_id, trace_id, user_id, domain, tool_name, finding_code, approved, scope, reused,
                     approval_key, expires_at, created_at)
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, current_timestamp(6))
                    """,
                    (
                        state.request_id,
                        state.trace_id,
                        state.user_id,
                        "FINANCE",
                        decision.get("tool_name", ""),
                        decision.get("finding_code", ""),
                        1 if decision.get("approved") else 0,
                        decision.get("scope", ""),
                        1 if decision.get("reused") else 0,
                        decision.get("approval_key"),
                        decision.get("expires_at"),
                    ),
                )


def build_audit_store() -> AuditStore:
    if settings.audit_backend.lower() == "mysql":
        try:
            return MySqlAuditStore()
        except Exception:
            return FileAuditStore(settings.audit_dir)
    return FileAuditStore(settings.audit_dir)

