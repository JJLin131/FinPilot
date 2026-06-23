from __future__ import annotations

import json
import logging

import httpx

from python_agent_service.config import settings
from python_agent_service.intents import INTENT_ORDER

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


class IntentClassificationService:
    PROMPT = """
You are a finance intent classifier.
Classify the user's request into exactly one of these intents:
{intents}

Return JSON only with this schema:
{{"intent":"INTENT_NAME","reason":"short explanation"}}

If the request is outside the supported domain, return UNKNOWN.

User question:
{query}
"""

    def __init__(self, client=None):
        self.client = client or self._build_client()

    def _build_client(self):
        if settings.routing_provider.lower() == "deepseek":
            return DeepSeekChatClient(
                model_name=settings.routing_model_name,
                timeout_seconds=settings.query_rewriter_timeout_seconds,
            )
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.routing_model_name,
            timeout_seconds=settings.query_rewriter_timeout_seconds,
        )

    def classify(self, query: str) -> tuple[str, str]:
        if not settings.routing_llm_enabled:
            raise RuntimeError("Routing LLM is disabled.")
        raw = self.client.generate(
            self.PROMPT.format(intents=", ".join(INTENT_ORDER), query=query),
            model_name=settings.routing_model_name,
        )
        payload = self._extract_json(raw)
        intent = str(payload.get("intent", "UNKNOWN")).strip().upper()
        reason = str(payload.get("reason", "")).strip() or "LLM classifier returned no reason."
        if intent not in INTENT_ORDER:
            intent = "UNKNOWN"
        return intent, reason

    def _extract_json(self, raw: str) -> dict:
        text = raw.strip()
        if "```" in text:
            text = text.replace("```json", "```")
            parts = [part.strip() for part in text.split("```") if part.strip()]
            for part in parts:
                if part.startswith("{") and part.endswith("}"):
                    return json.loads(part)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)


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
                timeout_seconds=settings.query_rewriter_timeout_seconds,
            )
        return OllamaClient(
            base_url=settings.ollama_base_url,
            model_name=settings.ai_model_name,
            timeout_seconds=settings.query_rewriter_timeout_seconds,
        )

    def answer_with_context(self, query: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "当前知识库没有命中足够相关的规则，请补充更具体的问题。"
        try:
            return self.client.generate(
                self.PROMPT.format(query=query, context="\n\n".join(context_chunks[:3])),
                model_name=settings.ai_model_name,
            )
        except Exception as exc:
            logger.warning("Answer generation via configured LLM failed, using fallback synthesis: %s", exc)
            return f"根据当前知识库，优先参考以下内容：{context_chunks[0]}"
