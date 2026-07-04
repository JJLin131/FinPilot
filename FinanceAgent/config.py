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
    rag_bootstrap_on_startup: bool = False
    bm25_index_path: Path = Field(default=Path("./data/bm25/knowledge.json"))
    rrf_k: int = 60

    ollama_base_url: str = "http://100.92.110.54:11434"
    embedding_base_url: str = "http://100.92.110.54:11434"
    embedding_model_name: str = "bge-m3"
    embedding_timeout_seconds: int = 60
    chroma_base_url: str = "http://100.92.110.54:8000"
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"
    chroma_finance_collection: str = "finance-knowledge-bge-m3-v1"
    chroma_memory_collection: str = "finance-user-memory-bge-m3-v1"
    vector_min_score: float = 0.45
    vector_enabled: bool = True
    memory_extraction_enabled: bool = True
    memory_min_confidence: float = 0.7
    memory_semantic_search_enabled: bool = True
    memory_semantic_search_limit: int = 5
    memory_semantic_min_score: float = 0.45
    memory_chroma_timeout_seconds: int = 3
    title_conflict_enabled: bool = True
    title_conflict_min_score: float = 0.82
    title_conflict_search_limit: int = 8

    rag_curation_enabled: bool = True
    rag_curation_provider: str = "deepseek"
    rag_curation_model_name: str = "deepseek-v4-pro"
    rag_curation_timeout_seconds: int = 60

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
