from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from finpilot.config import settings
from finpilot.mysql import connect_runtime_mysql

ReadinessStatus = Literal["ok", "degraded", "failed"]


class ReadinessCheck(BaseModel):
    name: str
    status: ReadinessStatus
    detail: str


class RuntimeReadiness(BaseModel):
    status: ReadinessStatus
    checks: list[ReadinessCheck]


def check_runtime_readiness() -> RuntimeReadiness:
    checks = [
        _check_mysql(),
        _check_bm25(),
        _check_optional_http("chroma", settings.vector_enabled, settings.chroma_base_url, ["/api/v1/heartbeat", "/api/v2/heartbeat", "/"]),
        _check_optional_http("embedding", settings.vector_enabled, settings.embedding_base_url, ["/api/tags", "/"]),
        _check_optional_http("reranker", settings.reranker_enabled, settings.reranker_base_url, ["/healthz", "/"]),
        _check_llm_config(),
    ]
    return RuntimeReadiness(status=_overall_status(checks), checks=checks)


def _check_mysql() -> ReadinessCheck:
    try:
        with connect_runtime_mysql() as connection:
            with connection.cursor() as cursor:
                cursor.execute("select 1")
        return ReadinessCheck(name="mysql", status="ok", detail="select 1 succeeded")
    except Exception as exc:
        return ReadinessCheck(name="mysql", status="failed", detail=_exception_detail(exc))


def _check_bm25() -> ReadinessCheck:
    path = settings.bm25_index_path
    if path.exists():
        return ReadinessCheck(name="bm25", status="ok", detail=str(path))
    return ReadinessCheck(name="bm25", status="degraded", detail=f"index not found: {path}")


def _check_llm_config() -> ReadinessCheck:
    deepseek_features = []
    if settings.ai_provider.lower() == "deepseek":
        deepseek_features.append("ai")
    if settings.routing_llm_enabled and settings.routing_provider.lower() == "deepseek":
        deepseek_features.append("routing")
    if settings.rag_curation_enabled and settings.rag_curation_provider.lower() == "deepseek":
        deepseek_features.append("rag_curation")
    if settings.safety_response_llm_enabled and settings.safety_response_provider.lower() == "deepseek":
        deepseek_features.append("safety_response")
    if not deepseek_features:
        return ReadinessCheck(name="llm_config", status="ok", detail="no DeepSeek-backed feature enabled")
    if settings.deepseek_api_key:
        return ReadinessCheck(name="llm_config", status="ok", detail=f"DeepSeek API key configured for {','.join(deepseek_features)}")
    return ReadinessCheck(
        name="llm_config",
        status="failed",
        detail=f"DEEPSEEK_API_KEY is not configured for {','.join(deepseek_features)}",
    )


def _check_optional_http(name: str, enabled: bool, base_url: str, paths: list[str]) -> ReadinessCheck:
    if not enabled:
        return ReadinessCheck(name=name, status="ok", detail="disabled")
    return _probe_http(name, base_url, paths)


def _probe_http(name: str, base_url: str, paths: list[str]) -> ReadinessCheck:
    base = base_url.rstrip("/")
    last_error = ""
    with httpx.Client(timeout=2.0) as client:
        for path in paths:
            url = f"{base}{path}"
            try:
                response = client.get(url)
                if response.status_code < 500:
                    return ReadinessCheck(name=name, status="ok", detail=f"{url} returned {response.status_code}")
                last_error = f"{url} returned {response.status_code}"
            except Exception as exc:
                last_error = _exception_detail(exc)
    return ReadinessCheck(name=name, status="degraded", detail=last_error or f"{base} unavailable")


def _overall_status(checks: list[ReadinessCheck]) -> ReadinessStatus:
    if any(check.status == "failed" for check in checks):
        return "failed"
    if any(check.status == "degraded" for check in checks):
        return "degraded"
    return "ok"


def _exception_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"
