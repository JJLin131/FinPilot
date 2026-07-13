from __future__ import annotations

import pytest

from finpilot.evals.backends import ControlledBackend, EvalBackendError, LiveBackend, SequenceFixture
from finpilot.evals.models import (
    EvalObservation,
    EvalStatus,
    QueryRewriteRerankerCase,
    RagRetrievalCase,
    ResilienceDegradationCase,
    ToolCallingCase,
)


def test_controlled_backend_uses_named_fixture_and_records_attempts():
    case = ResilienceDegradationCase(
        suite="resilience_degradation",
        case_id="retry-001",
        name="首次限流后恢复",
        prompt="查询余额",
        fixtures={"fixture": "planner-rate-limit"},
        fault={"component": "planner", "behavior": "rate_limit_once"},
        expected_status="SUCCEEDED",
        max_attempts=2,
    )
    fixture = SequenceFixture(
        [
            EvalObservation(status="FAILED", errors=[{"code": "RATE_LIMIT", "message": "429"}]),
            EvalObservation(status="SUCCEEDED", final_state={"balance": 1000}),
        ]
    )
    backend = ControlledBackend({"planner-rate-limit": fixture})

    first = backend.execute(case)
    second = backend.execute(case)

    assert first.status == "FAILED"
    assert second.status == "SUCCEEDED"
    assert fixture.call_count == 2


def test_controlled_backend_missing_fixture_is_structured_failure():
    case = ResilienceDegradationCase(
        suite="resilience_degradation",
        case_id="missing-001",
        name="缺少替身",
        prompt="查询余额",
        fault={"component": "planner", "behavior": "timeout"},
        expected_status="FAILED",
    )

    with pytest.raises(EvalBackendError) as captured:
        ControlledBackend({}).execute(case)

    assert captured.value.status is EvalStatus.FIXTURE_UNAVAILABLE
    assert captured.value.code == "FIXTURE_UNAVAILABLE"


def test_controlled_backend_accepts_inline_observation_fixture():
    case = ToolCallingCase(
        suite="tool_calling",
        case_id="inline-001",
        name="内联可控替身",
        execution_mode="controlled",
        fixtures={
            "observation": {
                "status": "REQUIRE_APPROVAL",
                "tool_calls": [{"tool_name": "transfer_mock_funds", "arguments": {}}],
                "side_effect_count": 0,
            }
        },
        user_message="执行模拟转账",
        expected_calls=[{"tool_name": "transfer_mock_funds", "arguments": {}}],
    )

    observation = ControlledBackend({}).execute(case)

    assert observation.status == "REQUIRE_APPROVAL"
    assert observation.tool_calls[0]["tool_name"] == "transfer_mock_funds"


def test_live_backend_runs_all_conversation_turns_with_isolated_identity():
    calls: list[tuple[str, str, str]] = []

    class Service:
        def chat(self, user_id: str, chat_id: str, content: str):
            calls.append((user_id, chat_id, content))
            return type(
                "Response",
                (),
                {"model_dump": lambda self, mode="json": {"status": "SUCCEEDED", "answer": f"answer:{content}"}},
            )()

    case = ToolCallingCase(
        suite="tool_calling",
        case_id="live-001",
        name="真实余额查询",
        execution_mode="live",
        user_message="查询 ACC-001 余额",
    )
    backend = LiveBackend(Service(), user_id="eval-user", run_id="run-123")

    observation = backend.execute(case)

    assert calls == [("eval-user", "eval-run-123-live-001", "查询 ACC-001 余额")]
    assert observation.status == "SUCCEEDED"
    assert observation.response["answer"] == "answer:查询 ACC-001 余额"


def test_live_backend_runs_retrieval_directly_and_exposes_ranked_documents():
    class Match:
        def model_dump(self, mode="json"):
            del mode
            return {"document_id": "doc-a", "text": "工资发放规则", "score": 0.9}

    class RagService:
        def search(self, query: str, limit: int):
            assert query == "工资规则"
            assert limit == 5
            return [Match()]

    service = type("Service", (), {"rag_service": RagService()})()
    case = RagRetrievalCase(
        suite="rag_retrieval",
        case_id="retrieval-live",
        name="实时检索",
        execution_mode="live",
        query="工资规则",
        relevant_document_ids=["doc-a"],
        k_values=[1, 5],
    )

    observation = LiveBackend(service, user_id="eval-user", run_id="run-1").execute(case)

    assert observation.status == "COMPLETED"
    assert observation.retrieved_documents[0]["document_id"] == "doc-a"


def test_live_backend_exposes_query_rewrite_and_reranker_trace():
    class RagService:
        def search_trace(self, query: str, limit: int):
            assert query == "工资咋发"
            assert limit == 5
            return {
                "rewritten_queries": ["企业工资发放规则"],
                "candidates": [{"document_id": "doc-b", "text": "其他"}],
                "reranked": [{"document_id": "doc-a", "text": "工资发放 审批规则"}],
            }

    service = type("Service", (), {"rag_service": RagService()})()
    case = QueryRewriteRerankerCase(
        suite="query_rewrite_reranker",
        case_id="rewrite-live",
        name="实时改写重排",
        execution_mode="live",
        query="工资咋发",
        reference_rewrite="企业工资发放规则",
        relevant_document_ids=["doc-a"],
        subtopics=["工资发放", "审批规则"],
    )

    observation = LiveBackend(service, user_id="eval-user", run_id="run-1").execute(case)

    assert observation.rewritten_query == "企业工资发放规则"
    assert observation.retrieved_documents[0]["document_id"] == "doc-b"
    assert observation.ranked_documents[0]["document_id"] == "doc-a"
