from __future__ import annotations


STRUCTURED_MEMORY_FIELDS: dict[str, str] = {
    "city": "用户所在城市。",
    "gender": "用户性别。",
    "age": "用户年龄。",
    "occupation": "用户职业或行业。",
}


SEMANTIC_MEMORY_KEYS: dict[str, str] = {
    "userBank": "用户常用银行、开户行、主要银行关系等信息。",
    "userCard": "用户银行卡、卡用途、工资卡、还款卡、常用卡等信息。",
    "userSalary": "用户工资发放时间、发薪卡、收入到账习惯等信息。",
    "userFund": "用户基金关注、持仓偏好、赎回习惯等信息。",
    "userLoan": "用户贷款、还款、房贷、车贷、授信等信息。",
    "userInsurance": "用户保险产品、保障偏好、保单相关信息。",
    "userPayment": "用户支付习惯、扣款方式、自动还款等信息。",
    "userRiskConcern": "用户风险顾虑、资金安全、流动性焦虑等信息。",
    "userRecentFocus": "用户近期反复关注的金融问题或业务主题。",
}


def render_memory_key_definitions() -> str:
    structured = "\n".join(f"- {key}: {description}" for key, description in STRUCTURED_MEMORY_FIELDS.items())
    semantic = "\n".join(f"- {key}: {description}" for key, description in SEMANTIC_MEMORY_KEYS.items())
    return f"""Structured MySQL fields:
{structured}

Semantic Chroma memory keys:
{semantic}"""
