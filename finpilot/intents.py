from __future__ import annotations

INTENT_ORDER = [
    "FINANCE_KNOWLEDGE_QA",
    "GENERAL_KNOWLEDGE_QA",
    "UNKNOWN",
]

INTENT_KEYWORDS = {
    "FINANCE_KNOWLEDGE_QA": [
        "规则",
        "制度",
        "审批",
        "状态码",
        "要求",
        "合规",
        "银行",
        "转账",
        "回单",
        "资金用途",
        "资金审批",
        "bank",
        "rule",
        "policy",
        "approval",
    ],
    "GENERAL_KNOWLEDGE_QA": [
        "财务",
        "金融",
        "资金问题",
        "咨询",
        "分析",
        "处理",
        "finance",
    ],
}

UNKNOWN_INTENT_ANSWER = "系统暂不支持这类请求"
