from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from finpilot.config import settings
from finpilot.memory.crypto import MemoryCipher
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
        _check_model_context_window(),
        _check_memory_encryption(),
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


def _check_model_context_window() -> ReadinessCheck:
    model_label = f"{settings.ai_provider}:{settings.ai_model_name}"
    windows = settings.model_context_windows or {}
    value = windows.get(model_label) or windows.get(settings.ai_model_name)
    if value is None:
        return ReadinessCheck(
            name="model_context_window",
            status="degraded",
            detail=f"MODEL_CONTEXT_WINDOWS is not configured for {model_label}; context policy budget will be used.",
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = 0
    if parsed <= 0:
        return ReadinessCheck(
            name="model_context_window",
            status="degraded",
            detail=f"MODEL_CONTEXT_WINDOWS has an invalid value for {model_label}: {value}",
        )
    return ReadinessCheck(
        name="model_context_window",
        status="ok",
        detail=f"{model_label} context window is {parsed} tokens",
    )


def _check_memory_encryption() -> ReadinessCheck:
    try:
        cipher = MemoryCipher.from_base64_key(settings.memory_encryption_key)
    except ValueError as exc:
        return ReadinessCheck(name="memory_encryption", status="failed", detail=str(exc))
    try:
        with connect_runtime_mysql() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    create table if not exists memory_encryption_key_check (
                        id tinyint not null primary key,
                        key_fingerprint char(64) not null,
                        created_at datetime(6) not null
                    )
                    """
                )
                cursor.execute("select key_fingerprint from memory_encryption_key_check where id = 1")
                row = cursor.fetchone()
                if row and str(row[0]) != cipher.key_fingerprint:
                    return ReadinessCheck(
                        name="memory_encryption",
                        status="failed",
                        detail="MEMORY_ENCRYPTION_KEY does not match the persisted key fingerprint.",
                    )
                if not row:
                    cursor.execute(
                        "insert into memory_encryption_key_check(id, key_fingerprint, created_at) "
                        "values (1, %s, current_timestamp(6))",
                        (cipher.key_fingerprint,),
                    )
    except Exception as exc:
        return ReadinessCheck(name="memory_encryption", status="failed", detail=_exception_detail(exc))
    return ReadinessCheck(name="memory_encryption", status="ok", detail="AES-256-GCM key verified")


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
