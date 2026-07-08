from __future__ import annotations

from finpilot.memory.extractor import MemoryExtractor
from finpilot.models import RouteDecision


def test_memory_extractor_prompt_uses_readable_utf8_examples():
    prompt = MemoryExtractor(client=object()).build_prompt(
        user_id="user-1",
        chat_id="chat-1",
        user_message="我每个月12号会发工资到工资卡。",
        assistant_answer="已记录你的发薪日偏好。",
        route=RouteDecision(
            raw_intent_json="{}",
            normalized_intent="FINANCE_QA",
            reason="test",
            confidence=1.0,
            valid=True,
            target_agent="QueryAgent",
            classifier_intent="FINANCE_QA",
        ),
    )

    assert '"city": "杭州"' in prompt
    assert '"occupation": "财务经理"' in prompt
    assert '"memoryValue": "用户每个月12号会发工资到工资卡。"' in prompt
    assert '"evidence": "用户说：我每个月12号都会发工资到这张卡"' in prompt
    for mojibake in ("閺夘厼", "闁炬儼", "閻", "鐢ㄦ埛"):
        assert mojibake not in prompt
