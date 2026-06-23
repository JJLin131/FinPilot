from __future__ import annotations

INTENT_ORDER = [
    "CANCEL_TRANSFER",
    "TRANSFER_STATUS",
    "TRANSFER_RECEIPT",
    "ACCOUNT_BALANCE",
    "KNOWLEDGE_QA",
    "GENERAL_FINANCE",
    "UNKNOWN",
]

INTENT_KEYWORDS = {
    "CANCEL_TRANSFER": ["取消", "撤销", "停止", "终止", "cancel"],
    "TRANSFER_STATUS": ["状态", "进度", "到账", "到帐", "status", "成功了吗"],
    "TRANSFER_RECEIPT": ["回单", "凭证", "receipt", "证明", "电子回单"],
    "ACCOUNT_BALANCE": ["余额", "账户", "可用金额", "流水", "balance"],
    "KNOWLEDGE_QA": ["规则", "制度", "审批", "状态码", "要求", "合规", "bank", "rule"],
    "GENERAL_FINANCE": ["财务", "资金", "分析", "finance", "处理"],
}

UNKNOWN_INTENT_ANSWER = "系统暂不支持这类请求"

