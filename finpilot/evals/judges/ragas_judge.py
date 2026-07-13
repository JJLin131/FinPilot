from __future__ import annotations

import copy

from finpilot.config import settings


class JudgeUnavailable(RuntimeError):
    pass


class RagasJudge:
    def __init__(self, *, llm=None, embeddings=None):
        self.llm = llm
        self.embeddings = embeddings

    def score(self, case, observation) -> dict[str, float]:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import answer_correctness, answer_relevancy, faithfulness
        except ImportError as exc:
            raise JudgeUnavailable("Ragas evaluation dependencies are not installed.") from exc
        llm = self.llm or self._build_llm()
        embeddings = self.embeddings or self._build_embeddings()
        answer = str(observation.response.get("answer") or "")
        dataset = Dataset.from_dict(
            {
                "question": [case.question],
                "answer": [answer],
                "contexts": [[context.text for context in case.contexts]],
                "ground_truth": [case.reference_answer],
            }
        )
        # 当前 OpenAI 兼容服务只支持 n=1；RAGAS 的默认 strictness=3 会发送 n=3。
        answer_relevancy_metric = copy.deepcopy(answer_relevancy)
        answer_relevancy_metric.strictness = 1
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_correctness, answer_relevancy_metric],
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
        ).to_pandas().iloc[0]
        return {
            "faithfulness": float(scores["faithfulness"]),
            "answer_correctness": float(scores["answer_correctness"]),
            "answer_relevance": float(scores["answer_relevancy"]),
        }

    @staticmethod
    def _build_llm():
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise JudgeUnavailable("langchain-openai is required for Ragas evaluation.") from exc
        if settings.ai_provider.lower() == "deepseek":
            if not settings.deepseek_api_key:
                raise JudgeUnavailable("DEEPSEEK_API_KEY is required for Ragas evaluation.")
            return ChatOpenAI(
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                model=settings.ai_model_name,
                temperature=0,
            )
        return ChatOpenAI(
            api_key="ollama",
            base_url=RagasJudge._openai_base_url(settings.ollama_base_url),
            model=settings.ai_model_name,
            temperature=0,
        )

    @staticmethod
    def _build_embeddings():
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise JudgeUnavailable("langchain-openai is required for Ragas embeddings.") from exc
        return OpenAIEmbeddings(
            api_key="ollama",
            base_url=RagasJudge._openai_base_url(settings.embedding_base_url),
            model=settings.embedding_model_name,
            check_embedding_ctx_length=False,
        )

    @staticmethod
    def _openai_base_url(base_url: str) -> str:
        normalized = base_url.rstrip("/")
        return normalized if normalized.endswith("/v1") else f"{normalized}/v1"
