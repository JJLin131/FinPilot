from __future__ import annotations

import copy
import time
from typing import Protocol

from finpilot.evals.models import (
    BaseEvalCase,
    EndToEndTaskCase,
    EvalObservation,
    EvalStatus,
    MultiTurnMemoryCase,
    QueryRewriteRerankerCase,
    RagRetrievalCase,
)
from finpilot.observability.capture import capture_observability
from finpilot.usage import capture_usage


class EvalBackendError(RuntimeError):
    def __init__(self, status: EvalStatus, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


class EvalFixture(Protocol):
    def execute(self, case: BaseEvalCase) -> EvalObservation: ...


class SequenceFixture:
    def __init__(self, observations: list[EvalObservation]):
        self._observations = list(observations)
        self.call_count = 0

    def execute(self, case: BaseEvalCase) -> EvalObservation:
        del case
        if self.call_count >= len(self._observations):
            raise EvalBackendError(
                EvalStatus.FIXTURE_UNAVAILABLE,
                "FIXTURE_SEQUENCE_EXHAUSTED",
                "Controlled fixture has no remaining observations.",
            )
        observation = copy.deepcopy(self._observations[self.call_count])
        self.call_count += 1
        observation.attempts = self.call_count
        return observation


class ControlledBackend:
    def __init__(self, fixtures: dict[str, EvalFixture]):
        self.fixtures = dict(fixtures)

    def execute(self, case: BaseEvalCase) -> EvalObservation:
        inline_observation = case.fixtures.get("observation")
        if isinstance(inline_observation, dict):
            return EvalObservation.model_validate(copy.deepcopy(inline_observation))
        fixture_name = str(case.fixtures.get("fixture") or "")
        fixture = self.fixtures.get(fixture_name)
        if fixture is None:
            raise EvalBackendError(
                EvalStatus.FIXTURE_UNAVAILABLE,
                "FIXTURE_UNAVAILABLE",
                f"Controlled fixture is not registered: {fixture_name or '<empty>'}",
            )
        return fixture.execute(case)


class LiveBackend:
    def __init__(self, agent_service, *, user_id: str, run_id: str):
        self.agent_service = agent_service
        self.user_id = user_id
        self.run_id = run_id

    def execute(self, case: BaseEvalCase) -> EvalObservation:
        with capture_usage() as usage, capture_observability() as telemetry:
            observation = self._execute(case)
        return observation.model_copy(
            update={
                "token_usage": usage.token_usage,
                "cost": usage.cost,
                "traces": telemetry.spans,
                "scores": telemetry.scores,
                "audit_events": telemetry.audit_events,
            }
        )

    def _execute(self, case: BaseEvalCase) -> EvalObservation:
        if isinstance(case, RagRetrievalCase):
            return self._execute_retrieval(case)
        if isinstance(case, QueryRewriteRerankerCase):
            return self._execute_rewrite_reranker(case)
        messages = self._messages(case)
        if not messages:
            raise EvalBackendError(EvalStatus.EVALUATOR_ERROR, "CASE_INPUT_MISSING", "Case has no executable input.")
        chat_id = f"eval-{self.run_id}-{case.case_id}"
        response_payload: dict = {}
        started = time.perf_counter()
        for message in messages:
            response = self.agent_service.chat(self.user_id, chat_id, message)
            response_payload = response.model_dump(mode="json")
        duration_ms = round((time.perf_counter() - started) * 1000, 3)
        final_state = self._response_final_state(case, chat_id, response_payload)
        return EvalObservation(
            status=str(response_payload.get("status") or "FAILED"),
            response=response_payload,
            plan=dict(response_payload.get("plan") or {}),
            tool_calls=list(response_payload.get("tool_calls") or []),
            final_state=final_state,
            duration_ms=duration_ms,
        )

    def _response_final_state(self, case: BaseEvalCase, chat_id: str, response_payload: dict) -> dict:
        final_state: dict = {"status": response_payload.get("status")}
        tool_calls = response_payload.get("tool_calls") or []
        final_state["receipt_downloaded"] = any(
            call.get("tool_name") == "download_receipt" and call.get("status") == "SUCCEEDED"
            for call in tool_calls
            if isinstance(call, dict)
        )
        if not isinstance(case, MultiTurnMemoryCase):
            return final_state
        manager = self.agent_service.memory_manager
        wait_for_pending = getattr(manager, "wait_for_pending", None)
        if callable(wait_for_pending):
            wait_for_pending()
        context = manager.load(self.user_id, chat_id)
        isolation = manager.load(case.isolation_user_id, chat_id) if case.isolation_user_id else None
        final_state["memories"] = self._memory_values(context)
        final_state["isolation_memories"] = self._memory_values(isolation) if isolation else {}
        return final_state

    @staticmethod
    def _memory_values(context) -> dict:
        values = dict(context.structured_memory)
        values.update({item.memory_key: item.memory_value for item in context.semantic_memory})
        return values

    def _execute_retrieval(self, case: RagRetrievalCase) -> EvalObservation:
        started = time.perf_counter()
        matches = self.agent_service.rag_service.search(case.query, limit=max(case.k_values))
        return EvalObservation(
            status="COMPLETED",
            retrieved_documents=[self._dump_item(match) for match in matches],
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _execute_rewrite_reranker(self, case: QueryRewriteRerankerCase) -> EvalObservation:
        started = time.perf_counter()
        trace = self.agent_service.rag_service.search_trace(case.query, limit=5)
        rewritten_queries = [str(item) for item in trace.get("rewritten_queries") or []]
        return EvalObservation(
            status="COMPLETED",
            rewritten_query="\n".join(rewritten_queries),
            retrieved_documents=[self._dump_item(item) for item in trace.get("candidates") or []],
            ranked_documents=[self._dump_item(item) for item in trace.get("reranked") or []],
            duration_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @staticmethod
    def _dump_item(item) -> dict:
        if isinstance(item, dict):
            return dict(item)
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        raise TypeError(f"Unsupported evaluation observation item: {type(item).__name__}")

    @staticmethod
    def _messages(case: BaseEvalCase) -> list[str]:
        if isinstance(case, (EndToEndTaskCase, MultiTurnMemoryCase)):
            return [turn.content for turn in case.turns if turn.role == "user"]
        for field in ("user_message", "prompt", "question", "query"):
            value = getattr(case, field, None)
            if isinstance(value, str) and value.strip():
                return [value]
        return []
