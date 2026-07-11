from __future__ import annotations

import json

from finpilot.context.compression import CompressionRule, ContextBuilder, ContextPolicy, ContextSegment


class RecordingClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, model_name: str | None = None, system_prompt: str | None = None) -> str:
        del model_name, system_prompt
        self.prompts.append(prompt)
        return self.response


def _summary_json() -> str:
    return json.dumps(
        {
            "summary": "工资审批需要双人复核。",
            "key_facts": ["付款前需要双人复核"],
            "constraints": ["不得由同一人发起并审批"],
            "evidence_refs": [
                {"document_id": "doc-1", "source": "manual", "title": "Payroll"},
            ],
        },
        ensure_ascii=False,
    )


def test_context_summarizer_validates_and_caches_by_content_hash():
    from finpilot.context.summarization import ContextSummarizationService

    client = RecordingClient(_summary_json())
    service = ContextSummarizationService(client=client, model_name="test-model")
    cache: dict[str, object] = {}

    first = service.summarize(
        query="工资审批规则是什么？",
        path="evidence",
        value={"document_id": "doc-1", "text": "付款前需要双人复核。"},
        target_tokens=200,
        chunk_token_budget=8000,
        cache=cache,
    )
    second = service.summarize(
        query="工资审批规则是什么？",
        path="evidence",
        value={"document_id": "doc-1", "text": "付款前需要双人复核。"},
        target_tokens=200,
        chunk_token_budget=8000,
        cache=cache,
    )

    assert first.summary == "工资审批需要双人复核。"
    assert first.evidence_refs[0].document_id == "doc-1"
    assert second == first
    assert len(client.prompts) == 1


def test_context_summarizer_chunks_large_input_and_reduces_hierarchically():
    from finpilot.context.summarization import ContextSummarizationService
    from finpilot.context.compression import estimate_tokens

    client = RecordingClient(_summary_json())
    service = ContextSummarizationService(client=client, model_name="test-model")

    service.summarize(
        query="总结规则",
        path="session.recent_messages",
        value="第一段。\n\n" + ("很长的历史内容。" * 400),
        target_tokens=200,
        chunk_token_budget=200,
        cache={},
    )

    assert len(client.prompts) > 2
    assert "合并" in client.prompts[-1]
    reduce_prompts = [prompt for prompt in client.prompts if "合并以下分块摘要" in prompt]
    assert len(reduce_prompts) >= 2
    prompt_contents = [
        prompt.split("<context_data>\n", 1)[1].split("\n</context_data>", 1)[0]
        for prompt in client.prompts
    ]
    assert all(estimate_tokens(content) <= 200 for content in prompt_contents)


class FailingSummarizer:
    def summarize(self, **kwargs):
        del kwargs
        raise ValueError("invalid summary json")


class OversizedSummarizer:
    def summarize(self, **kwargs):
        del kwargs
        from finpilot.context.summarization import SemanticSummary

        return SemanticSummary(summary="x" * 10_000)


class StepAwareSummarizer:
    def summarize(self, **kwargs):
        from finpilot.context.summarization import EvidenceReference, SemanticSummary

        document_id = kwargs["value"][0]["document_id"]
        return SemanticSummary(
            summary=f"summary for {document_id}",
            evidence_refs=[EvidenceReference(document_id=document_id, source="manual")],
        )


def test_builder_records_llm_failure_and_uses_deterministic_fallback():
    builder = ContextBuilder(
        policies={
            "history": ContextPolicy(
                token_budget=600,
                trigger_ratio=0.2,
                compression_rules=[
                    CompressionRule(
                        name="recent_messages",
                        path="session.recent_messages",
                        method="llm",
                        target_tokens=100,
                        preserve_last=1,
                    )
                ],
                protected_paths=["session.user_message"],
            )
        },
        summarizer=FailingSummarizer(),
    )

    bundle = builder.build(
        "history",
        [
            ContextSegment(
                name="session",
                value={
                    "user_message": "current",
                    "recent_messages": [
                        {"role": "user", "content": "old" * 2000},
                        {"role": "assistant", "content": "latest response"},
                    ],
                },
            )
        ],
        stage="decision",
        query="current",
        summary_cache={},
    )

    assert bundle.status == "ready"
    assert bundle.payload["session"]["user_message"] == "current"
    assert any(event["action"] == "summarization_failed" for event in bundle.compression_events)
    assert any(event.get("fallback_reason") == "invalid summary json" for event in bundle.compression_events)


def test_builder_rejects_oversized_llm_summary_and_falls_back():
    builder = ContextBuilder(
        policies={
            "evidence_policy": ContextPolicy(
                token_budget=800,
                trigger_ratio=0.2,
                compression_rules=[
                    CompressionRule(
                        name="evidence",
                        path="evidence",
                        method="llm",
                        target_tokens=100,
                    )
                ],
            )
        },
        summarizer=OversizedSummarizer(),
    )

    bundle = builder.build(
        "evidence_policy",
        [ContextSegment(name="evidence", value=[{"text": "source" * 2_000}])],
        stage="answer",
        query="current",
    )

    assert "x" * 10_000 not in str(bundle.payload)
    assert any(event["action"] == "summarization_failed" for event in bundle.compression_events)
    assert any("target" in event.get("fallback_reason", "") for event in bundle.compression_events)


def test_tool_documents_are_summarized_per_observation_without_duplication():
    builder = ContextBuilder(
        policies={
            "tool_policy": ContextPolicy(
                token_budget=5_000,
                trigger_ratio=0.1,
                compression_rules=[
                    CompressionRule(
                        name="tool_outputs",
                        path="loop.tool_outputs",
                        method="llm",
                        target_tokens=400,
                    )
                ],
            )
        },
        summarizer=StepAwareSummarizer(),
    )
    history = [
        {
            "step_index": index,
            "observation": {
                "output": {
                    "documents": [
                        {
                            "document_id": f"doc-{index}",
                            "source": "manual",
                            "text": "evidence" * 2_000,
                        }
                    ]
                }
            },
        }
        for index in (1, 2)
    ]

    bundle = builder.build(
        "tool_policy",
        [ContextSegment(name="loop", value={"step_history": history})],
        stage="decision",
        query="current",
        summary_cache={},
    )

    outputs = [step["observation"]["output"] for step in bundle.payload["loop"]["step_history"]]
    assert outputs[0]["documents_summary"][0]["summary"] == "summary for doc-1"
    assert outputs[1]["documents_summary"][0]["summary"] == "summary for doc-2"
    llm_events = [event for event in bundle.compression_events if event["action"] == "llm_summarized"]
    assert len(llm_events) == 2
    assert llm_events[0]["path"] != llm_events[1]["path"]
    assert all(event["after_tokens"] > 0 for event in llm_events)
