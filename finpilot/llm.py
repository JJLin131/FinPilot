from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from finpilot.config import settings
from finpilot.issues import dependency_degraded_issue
from finpilot.models import AgentIssue
from finpilot.usage import record_usage

logger = logging.getLogger(__name__)


class DeepSeekChatClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int = 30,
    ):
        self.base_url = (base_url or settings.deepseek_base_url).rstrip("/")
        self.api_key = api_key or settings.deepseek_api_key
        self.model_name = model_name or settings.ai_model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, model_name: str | None = None, system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name or self.model_name,
                    "temperature": 0.1,
                    "messages": [
                        *([{"role": "system", "content": system_prompt}] if system_prompt else []),
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            response.raise_for_status()
            payload = response.json()
            usage = payload.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            input_rate = settings.ai_input_cost_per_million or 0.0
            output_rate = settings.ai_output_cost_per_million or 0.0
            record_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost=(prompt_tokens * input_rate + completion_tokens * output_rate) / 1_000_000,
            )
            choices = payload.get("choices", [])
            if not choices:
                raise ValueError("DeepSeek returned no choices.")
            text = choices[0].get("message", {}).get("content", "")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("DeepSeek returned an empty response.")
            return text.strip()


class OllamaClient:
    def __init__(self, base_url: str | None = None, model_name: str | None = None, timeout_seconds: int = 30):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.model_name = model_name or settings.query_rewriter_model_name
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str, *, model_name: str | None = None, system_prompt: str | None = None) -> str:
        merged_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name or self.model_name,
                    "prompt": merged_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            record_usage(
                prompt_tokens=int(payload.get("prompt_eval_count") or 0),
                completion_tokens=int(payload.get("eval_count") or 0),
                cost=0.0,
            )
            text = payload.get("response")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Ollama returned an empty response.")
            return text.strip()


class QueryRewriteService:
    PROMPT = """
You rewrite retrieval queries for a finance knowledge base.
Generate {max_rewrites} semantically complementary search queries for the user's question.
Keep the rewritten queries in the same language as the user whenever possible.
Output only the rewritten queries, one per line, with no numbering or explanation.

User question:
{query}
"""

    def __init__(self, client: OllamaClient | None = None):
        self.client = client or OllamaClient(
            base_url=settings.query_rewriter_base_url,
            model_name=settings.query_rewriter_model_name,
            timeout_seconds=settings.query_rewriter_timeout_seconds,
        )

    def rewrite(self, query: str) -> list[str]:
        if not settings.query_rewriter_enabled or not query.strip():
            return [query]
        try:
            raw = self.client.generate(
                self.PROMPT.format(max_rewrites=max(settings.query_rewriter_max_rewrites, 1), query=query),
                model_name=settings.query_rewriter_model_name,
            )
            values: list[str] = [query]
            for line in raw.splitlines():
                cleaned = line.strip().lstrip("-*").strip()
                if cleaned and cleaned not in values:
                    values.append(cleaned)
                if len(values) >= max(settings.query_rewriter_max_rewrites, 1) + 1:
                    break
            return values
        except Exception as exc:
            logger.warning("Query rewrite via remote Ollama failed, using original query: %s", exc)
            return [query]


class FinanceAnsweringService:
    PROMPT = """
You are a finance support assistant.
Answer the user's question using only the provided context.
If the context is insufficient, say so briefly.

User question:
{query}

Context:
{context}
"""

    def __init__(self, client=None):
        self.client = client or self._build_client()

    def _build_client(self):
        if settings.ai_provider.lower() == "deepseek":
            return DeepSeekChatClient(
                model_name=settings.ai_model_name,
                timeout_seconds=settings.ai_timeout_seconds,
            )
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.ai_model_name,
            timeout_seconds=settings.ai_timeout_seconds,
        )

    def answer_with_context(self, prompt_context: dict) -> str:
        self._issues = []
        session_payload = prompt_context.get("session") if isinstance(prompt_context.get("session"), dict) else {}
        evidence = prompt_context.get("evidence") if isinstance(prompt_context.get("evidence"), list) else []
        loop = prompt_context.get("loop") if isinstance(prompt_context.get("loop"), dict) else {}
        if not evidence and not any(session_payload.values()) and not loop.get("working_notes"):
            return "当前知识库没有命中足够相关的规则，请补充更具体的问题。"
        try:
            return self.client.generate(
                render_finance_answer_prompt(prompt_context),
                model_name=settings.ai_model_name,
            )
        except Exception as exc:
            logger.warning("Answer generation via configured LLM failed, using fallback synthesis: %s", exc)
            self._issues.append(
                dependency_degraded_issue(
                    code="ANSWER_LLM_DEGRADED",
                    component="answer_llm",
                    message="Answer generation model failed; using retrieved evidence fallback.",
                    exc=exc,
                )
            )
            if evidence:
                first = evidence[0] if isinstance(evidence[0], str) else json.dumps(evidence[0], ensure_ascii=False)
                return f"根据当前知识库，优先参考以下内容：{first}"
            return "当前上下文已保留，但回答模型暂时不可用，请稍后重试。"

    def consume_issues(self) -> list[AgentIssue]:
        issues = list(self._issues)
        self._issues = []
        return issues


def render_finance_answer_prompt(prompt_context: dict[str, Any]) -> str:
    session = prompt_context.get("session") if isinstance(prompt_context.get("session"), dict) else {}
    return FinanceAnsweringService.PROMPT.format(
        query=session.get("user_message", ""),
        context=json.dumps(prompt_context, ensure_ascii=False, indent=2),
    )

