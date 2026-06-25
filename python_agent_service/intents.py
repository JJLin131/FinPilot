from __future__ import annotations

INTENT_ORDER = [
    "KNOWLEDGE_QA",
    "GENERAL_FINANCE",
    "UNKNOWN",
]

INTENT_KEYWORDS = {
    "KNOWLEDGE_QA": ["规则", "制度", "审批", "状态码", "要求", "合规", "bank", "rule"],
    "GENERAL_FINANCE": ["财务", "资金", "金融", "finance", "处理"],
}

UNKNOWN_INTENT_ANSWER = "系统暂不支持这类请求"
