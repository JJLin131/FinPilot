from __future__ import annotations

import json
import logging

from FinanceAgent.config import settings
from FinanceAgent.llm import DeepSeekChatClient, OllamaClient
from FinanceAgent.memory.definitions import render_memory_key_definitions
from FinanceAgent.memory.models import ExtractedMemory
from FinanceAgent.models import RouteDecision

logger = logging.getLogger(__name__)


class MemoryExtractor:
    PROMPT = """
Extract durable user memory from the completed finance support conversation.
Use only the allowed field names and memory keys listed below. Do not invent keys.
Return JSON only with this schema:
{{
  "structuredMemories": {{
    "city": "杭州",
    "job": "银行从业者"
  }},
  "semanticMemories": [
    {{
      "memoryKey": "userSalary",
      "memoryValue": "用户每个月12号会发工资到工资卡。",
      "confidence": 0.9,
      "evidence": "用户说：我每个月12号都会发工资到这张卡"
    }}
  ]
}}

If nothing is worth remembering, return:
{{"structuredMemories": {{}}, "semanticMemories": []}}

Allowed memory definitions:
{definitions}

User id: {user_id}
Chat id: {chat_id}
Route intent: {intent}

User message:
{user_message}

Assistant answer:
{assistant_answer}
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

    def build_prompt(
        self,
        *,
        user_id: str,
        chat_id: str,
        user_message: str,
        assistant_answer: str,
        route: RouteDecision,
    ) -> str:
        return self.PROMPT.format(
            definitions=render_memory_key_definitions(),
            user_id=user_id,
            chat_id=chat_id,
            intent=route.normalized_intent,
            user_message=user_message,
            assistant_answer=assistant_answer,
        )

    def extract(
        self,
        *,
        user_id: str,
        chat_id: str,
        user_message: str,
        assistant_answer: str,
        route: RouteDecision,
    ) -> ExtractedMemory:
        if route.normalized_intent == "UNKNOWN":
            return ExtractedMemory()
        raw = self.client.generate(
            self.build_prompt(
                user_id=user_id,
                chat_id=chat_id,
                user_message=user_message,
                assistant_answer=assistant_answer,
                route=route,
            ),
            model_name=settings.ai_model_name,
        )
        return ExtractedMemory.model_validate(self._extract_json(raw))

    def _extract_json(self, raw: str) -> dict:
        text = raw.strip()
        if "```" in text:
            text = text.replace("```json", "```")
            for part in [part.strip() for part in text.split("```") if part.strip()]:
                if part.startswith("{") and part.endswith("}"):
                    return json.loads(part)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        return json.loads(text)
