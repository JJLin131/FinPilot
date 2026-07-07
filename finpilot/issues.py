from __future__ import annotations

import json

import httpx

from finpilot.models import AgentIssue, ToolInvocation

ROUTING_COMPONENT = "routing_llm"

ROUTING_MESSAGES = {
    "ROUTING_LLM_DISABLED": "路由侧 LLM 未启用，请检查路由配置。",
    "ROUTING_LLM_CONFIG_MISSING": "路由侧 LLM 配置缺失，请检查 API Key 或模型配置。",
    "ROUTING_LLM_AUTH_FAILED": "路由侧 LLM 鉴权失败，请检查凭证配置。",
    "ROUTING_LLM_MODEL_UNAVAILABLE": "路由侧 LLM 模型不可用，请检查路由模型配置或稍后重试。",
    "ROUTING_LLM_UNAVAILABLE": "路由侧 LLM 服务不可用，请稍后重试。",
    "ROUTING_LLM_RATE_LIMITED": "路由侧 LLM 当前限流，请稍后重试。",
    "ROUTING_LLM_SERVICE_ERROR": "路由侧 LLM 服务异常，请稍后重试。",
    "ROUTING_LLM_INVALID_RESPONSE": "路由侧 LLM 返回格式异常，请检查模型输出配置。",
}


def routing_llm_issue_from_exception(exc: Exception) -> AgentIssue | None:
    # 只把可预期的路由侧依赖问题转成用户可见诊断，避免掩盖真实代码缺陷。
    code = _routing_issue_code(exc)
    if code is None:
        return None
    return AgentIssue(
        code=code,
        component=ROUTING_COMPONENT,
        message=ROUTING_MESSAGES[code],
        severity="error",
        retryable=code
        in {
            "ROUTING_LLM_MODEL_UNAVAILABLE",
            "ROUTING_LLM_UNAVAILABLE",
            "ROUTING_LLM_RATE_LIMITED",
            "ROUTING_LLM_SERVICE_ERROR",
        },
        detail=_safe_exception_detail(exc),
    )


def issue_from_tool_failure(invocation: ToolInvocation) -> AgentIssue:
    return AgentIssue(
        code="TOOL_INVOCATION_FAILED",
        component=f"tool:{invocation.tool_name}",
        message="工具调用失败，请稍后重试或联系管理员查看请求编号。",
        severity="error",
        retryable=True,
        detail=_truncate(str(invocation.output.get("error") or invocation.observation_summary or "tool failed")),
    )


def dependency_degraded_issue(*, code: str, component: str, message: str, exc: Exception) -> AgentIssue:
    return AgentIssue(
        code=code,
        component=component,
        message=message,
        severity="warning",
        retryable=True,
        detail=_safe_exception_detail(exc),
    )


def scrub_issue_details(issues: list[AgentIssue]) -> list[AgentIssue]:
    # 普通响应不返回内部异常细节，调用方仍可用 code/component 做判断。
    return [issue.model_copy(update={"detail": None}) for issue in issues]


def _routing_issue_code(exc: Exception) -> str | None:
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        body = _response_text(exc)
        lowered = body.lower()
        if status_code in {401, 403}:
            return "ROUTING_LLM_AUTH_FAILED"
        if status_code == 404 or _looks_like_model_error(lowered):
            return "ROUTING_LLM_MODEL_UNAVAILABLE"
        if status_code == 429:
            return "ROUTING_LLM_RATE_LIMITED"
        if status_code >= 500:
            return "ROUTING_LLM_SERVICE_ERROR"
        return "ROUTING_LLM_INVALID_RESPONSE"
    if isinstance(exc, httpx.TimeoutException | httpx.ConnectError | httpx.NetworkError):
        return "ROUTING_LLM_UNAVAILABLE"
    if isinstance(exc, json.JSONDecodeError):
        return "ROUTING_LLM_INVALID_RESPONSE"
    if isinstance(exc, RuntimeError):
        message = str(exc)
        if "Routing LLM is disabled" in message:
            return "ROUTING_LLM_DISABLED"
        if "DEEPSEEK_API_KEY" in message or "api key" in message.lower():
            return "ROUTING_LLM_CONFIG_MISSING"
    if isinstance(exc, ValueError) and _looks_like_invalid_response(str(exc)):
        return "ROUTING_LLM_INVALID_RESPONSE"
    return None


def _looks_like_model_error(text: str) -> bool:
    return "model" in text and any(token in text for token in ("not found", "not exist", "unavailable", "invalid"))


def _looks_like_invalid_response(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ("empty response", "no choices", "invalid", "json", "response"))


def _safe_exception_detail(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        body = _response_text(exc)
        return _truncate(f"{type(exc).__name__}: HTTP {exc.response.status_code}; {body}")
    return _truncate(f"{type(exc).__name__}: {exc}")


def _response_text(exc: httpx.HTTPStatusError) -> str:
    try:
        return _truncate(exc.response.text)
    except Exception:
        return ""


def _truncate(value: str, limit: int = 240) -> str:
    cleaned = " ".join(value.replace("\r", " ").replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
