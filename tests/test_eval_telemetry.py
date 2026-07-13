from __future__ import annotations

from finpilot.evals.backends import LiveBackend
from finpilot.evals.models import PerformanceCostCase
from finpilot.config import settings
from finpilot.llm import DeepSeekChatClient, OllamaClient
from finpilot.observability.capture import capture_observability, record_audit_event, record_score, record_span
from finpilot.usage import capture_usage, record_usage


def test_usage_capture_aggregates_provider_tokens_and_cost():
    with capture_usage() as usage:
        record_usage(prompt_tokens=100, completion_tokens=25, cost=0.0125)
        record_usage(prompt_tokens=50, completion_tokens=10, cost=0.005)

    assert usage.token_usage == {"prompt_tokens": 150, "completion_tokens": 35, "total_tokens": 185}
    assert usage.cost == 0.0175


def test_observability_capture_records_actual_runtime_events():
    with capture_observability() as captured:
        record_span("agent.chat", {"status": "ok"})
        record_score("runtime.tool_success", 1.0)
        record_audit_event("tool_call", {"tool_name": "query_account_balance"})

    assert captured.spans == [{"name": "agent.chat", "attributes": {"status": "ok"}}]
    assert captured.scores == [{"name": "runtime.tool_success", "value": 1.0}]
    assert captured.audit_events[0]["type"] == "tool_call"


def test_live_backend_attaches_usage_and_observability_to_observation():
    class Response:
        def model_dump(self, mode="json"):
            del mode
            return {"status": "COMPLETED", "answer": "ok"}

    class Service:
        def chat(self, user_id, chat_id, content):
            del user_id, chat_id, content
            record_usage(prompt_tokens=10, completion_tokens=5, cost=0.01)
            record_span("agent.chat")
            record_score("runtime.tool_success", 1.0)
            record_audit_event("tool_call", {"tool_name": "query_account_balance"})
            return Response()

    case = PerformanceCostCase(
        suite="performance_cost",
        case_id="perf-live",
        name="采集性能数据",
        execution_mode="live",
        prompt="查询余额",
        max_p95_ms=1000,
        max_total_tokens=100,
        max_cost=1,
    )

    observation = LiveBackend(Service(), user_id="eval-user", run_id="run-1").execute(case)

    assert observation.token_usage["total_tokens"] == 15
    assert observation.cost == 0.01
    assert observation.traces[0]["name"] == "agent.chat"
    assert observation.scores[0]["name"] == "runtime.tool_success"
    assert observation.audit_events[0]["type"] == "tool_call"


def test_llm_clients_record_provider_reported_usage(monkeypatch):
    payloads = [
        {
            "choices": [{"message": {"content": "deepseek answer"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        },
        {"response": "ollama answer", "prompt_eval_count": 30, "eval_count": 10},
    ]

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Client:
        def __init__(self, timeout):
            del timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            del args, kwargs
            return Response(payloads.pop(0))

    monkeypatch.setattr("finpilot.llm.httpx.Client", Client)
    monkeypatch.setattr(settings, "ai_input_cost_per_million", 2.0)
    monkeypatch.setattr(settings, "ai_output_cost_per_million", 4.0)

    with capture_usage() as usage:
        assert DeepSeekChatClient(api_key="test-key").generate("prompt") == "deepseek answer"
        assert OllamaClient().generate("prompt") == "ollama answer"

    assert usage.token_usage == {"prompt_tokens": 130, "completion_tokens": 30, "total_tokens": 160}
    assert usage.cost == 0.00028
