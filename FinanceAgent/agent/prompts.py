from __future__ import annotations

import json
from typing import Any


def build_agent_decision_prompt(prompt_context: dict[str, Any]) -> str:
    return f"""
You are deciding the next step for a controlled sub-agent loop.
Use only the allowed tools listed in sub_agent.allowed_tools.
Return JSON only with this schema:
{{
  "decision": "act|answer|fallback|stop",
  "reason": "short reason",
  "tool_name": "tool name when decision is act, otherwise null",
  "tool_args": {{"query": "search query when needed"}},
  "enough_information": true,
  "draft_answer": "optional answer when decision is answer or fallback"
}}

Context:
{json.dumps(prompt_context, ensure_ascii=False, indent=2)}
"""


def build_answer_prompt_context_text(prompt_context: dict[str, Any]) -> str:
    return f"""
Compose the final finance support answer from the available evidence.
If evidence_sufficient is false, state briefly that the knowledge base does not contain enough information.

Context:
{json.dumps(prompt_context, ensure_ascii=False, indent=2)}
"""
