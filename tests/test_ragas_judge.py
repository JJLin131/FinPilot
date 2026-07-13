from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from finpilot.evals.judges.ragas_judge import RagasJudge
from finpilot.evals.models import EvalObservation, RagGenerationCase


def test_ragas_judge_requests_faithfulness_correctness_and_relevance(monkeypatch):
    requested_metrics = []

    class Dataset:
        @staticmethod
        def from_dict(payload):
            return payload

    class Frame:
        iloc = None

        def __init__(self):
            self.iloc = self

        def __getitem__(self, index):
            assert index == 0
            return {
                "faithfulness": 0.91,
                "answer_correctness": 0.88,
                "answer_relevancy": 0.93,
            }

    def evaluate(dataset, metrics, llm, embeddings, show_progress):
        assert dataset["question"] == ["工资规则？"]
        assert llm is judge_llm
        assert embeddings is judge_embeddings
        assert show_progress is False
        requested_metrics.extend(metrics)
        return SimpleNamespace(to_pandas=lambda: Frame())

    datasets_module = ModuleType("datasets")
    datasets_module.Dataset = Dataset
    ragas_module = ModuleType("ragas")
    ragas_module.evaluate = evaluate
    metrics_module = ModuleType("ragas.metrics")
    metrics_module.faithfulness = object()
    metrics_module.answer_correctness = object()
    metrics_module.answer_relevancy = SimpleNamespace(strictness=3)
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "ragas", ragas_module)
    monkeypatch.setitem(sys.modules, "ragas.metrics", metrics_module)

    case = RagGenerationCase(
        suite="rag_generation",
        case_id="ragas-1",
        name="真实指标",
        question="工资规则？",
        reference_answer="需要审批",
        contexts=[{"document_id": "doc-1", "text": "工资需要审批"}],
    )
    observation = EvalObservation(status="COMPLETED", response={"answer": "需要审批"})

    judge_llm = object()
    judge_embeddings = object()
    scores = RagasJudge(llm=judge_llm, embeddings=judge_embeddings).score(case, observation)

    assert requested_metrics[:2] == [metrics_module.faithfulness, metrics_module.answer_correctness]
    assert requested_metrics[2] is not metrics_module.answer_relevancy
    assert requested_metrics[2].strictness == 1
    assert metrics_module.answer_relevancy.strictness == 3
    assert scores == {"faithfulness": 0.91, "answer_correctness": 0.88, "answer_relevance": 0.93}


def test_ragas_embeddings_send_text_to_openai_compatible_endpoint(monkeypatch):
    captured = {}

    class OpenAIEmbeddings:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    langchain_openai = ModuleType("langchain_openai")
    langchain_openai.OpenAIEmbeddings = OpenAIEmbeddings
    monkeypatch.setitem(sys.modules, "langchain_openai", langchain_openai)

    RagasJudge._build_embeddings()

    assert captured["check_embedding_ctx_length"] is False
