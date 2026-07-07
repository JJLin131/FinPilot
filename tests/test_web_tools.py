from __future__ import annotations

import httpx

from finpilot.agent.tools import ToolRegistry
from finpilot.config import settings
from finpilot.models import GraphState
from finpilot.safety.service import SafetyReviewService


class EmptyRagService:
    def search(self, query: str, limit: int = 3):
        return []


class FakeResponse:
    def __init__(self, payload=None, text="", headers=None):
        self._payload = payload or {}
        self.text = text
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


def _state() -> GraphState:
    return GraphState(
        request_id="req-1",
        trace_id="trace-1",
        user_id="user-1",
        chat_id="chat-1",
        memory_id="chat:user-1:chat-1",
        user_message="latest finance regulation",
        normalized_intent="GENERAL_KNOWLEDGE_QA",
        target_agent="QueryAgent",
    )


def test_web_search_parses_brave_results(monkeypatch):
    def fake_get(url, *, headers=None, params=None, timeout=None, follow_redirects=False):
        assert url == "https://api.search.brave.com/res/v1/web/search"
        assert headers["X-Subscription-Token"] == "test-key"
        assert params["q"] == "payroll policy"
        return FakeResponse(
            {
                "web": {
                    "results": [
                        {
                            "title": "Payroll Policy",
                            "url": "https://example.com/payroll",
                            "description": "Policy summary",
                            "extra_snippets": ["Extra context"],
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr(settings, "brave_search_api_key", "test-key", raising=False)
    monkeypatch.setattr(httpx, "get", fake_get)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "web_search", query="payroll policy", count=1)

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["content"][0]["source"] == "https://example.com/payroll"
    assert invocation.output["content"][0]["title"] == "Payroll Policy"
    assert "Policy summary" in invocation.output["content"][0]["text"]
    assert "Extra context" in invocation.output["content"][0]["text"]


def test_fetch_url_rejects_local_network_targets(monkeypatch):
    monkeypatch.setattr(settings, "web_fetch_max_chars", 1000, raising=False)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "fetch_url", url="http://127.0.0.1:8099/healthz")

    assert invocation.status == "FAILED"
    assert "not allowed" in invocation.output["error"]


def test_fetch_url_extracts_text_and_truncates(monkeypatch):
    def fake_get(url, *, headers=None, params=None, timeout=None, follow_redirects=False):
        assert url == "https://example.com/report"
        return FakeResponse(text="<html><body><h1>Report</h1><p>abcdef</p></body></html>")

    monkeypatch.setattr(settings, "web_fetch_max_chars", 10, raising=False)
    monkeypatch.setattr(httpx, "get", fake_get)
    registry = ToolRegistry(EmptyRagService(), safety=SafetyReviewService())

    invocation = registry.invoke(_state(), "fetch_url", url="https://example.com/report")

    assert invocation.status == "SUCCEEDED"
    assert invocation.output["content"][0]["source"] == "https://example.com/report"
    assert invocation.output["content"][0]["text"] == "Report abc"
    assert invocation.output["content"][0]["metadata"]["truncated"] is True

