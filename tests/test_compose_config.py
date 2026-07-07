from __future__ import annotations

from pathlib import Path


def test_compose_requires_sensitive_environment_variables():
    compose = Path("compose.yaml").read_text(encoding="utf-8")

    for name in [
        "DEEPSEEK_API_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_POSTGRES_PASSWORD",
        "LANGFUSE_REDIS_AUTH",
        "MINIO_ROOT_PASSWORD",
        "MYSQL_PASSWORD",
    ]:
        assert f"${{{name}:?" in compose
