from __future__ import annotations

import json

from finpilot.config import settings
from finpilot.llm import DeepSeekChatClient, OllamaClient
from finpilot.models import AgentDecision


class AgentDecisionService:
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

    def decide(self, prompt: str) -> AgentDecision:
        raw = self.client.generate(prompt, model_name=settings.ai_model_name)
        return AgentDecision.model_validate(self._extract_json(raw))

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

