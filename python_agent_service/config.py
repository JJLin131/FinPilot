from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    app_port: int = 8099

    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str | None = None
    ai_provider: str = "deepseek"
    ai_model_name: str = "deepseek-v4-pro"
    ai_debug_trace: bool = False

    routing_llm_enabled: bool = True
    routing_provider: str = "deepseek"
    routing_model_name: str = "deepseek-v4-pro"

    audit_backend: str = "mysql"
    audit_dir: Path = Field(default=Path("./data/python-audit"))
    knowledge_dir: Path = Field(default=Path("./src/main/resources"))

    ollama_base_url: str = "http://100.92.110.54:11434"
    query_rewriter_base_url: str = "http://100.92.110.54:11434"
    query_rewriter_model_name: str = "qwen2.5:3b"
    query_rewriter_enabled: bool = True
    query_rewriter_timeout_seconds: int = 30
    query_rewriter_max_rewrites: int = 3

    reranker_base_url: str = "http://100.92.110.54:8081"
    reranker_enabled: bool = True
    reranker_timeout_seconds: int = 30

    otel_enabled: bool = True
    otel_exporter_otlp_endpoint: str | None = "http://localhost:4318"

    langfuse_enabled: bool = False
    langfuse_host: str | None = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "work_memory"


settings = Settings()
