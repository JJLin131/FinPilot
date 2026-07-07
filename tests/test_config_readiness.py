from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from finpilot.config import Settings, settings


def test_local_settings_allow_development_defaults():
    settings = Settings(_env_file=None)

    assert settings.app_env == "local"
    assert settings.mysql_user == "root"


def test_non_local_settings_reject_default_mysql_password():
    with pytest.raises(ValidationError, match="MYSQL_PASSWORD"):
        Settings(
            _env_file=None,
            app_env="docker",
            deepseek_api_key="sk-test",
            mysql_user="root",
            mysql_password="123456",
            langfuse_enabled=False,
        )


def test_non_local_settings_reject_missing_deepseek_key():
    with pytest.raises(ValidationError, match="DEEPSEEK_API_KEY"):
        Settings(
            _env_file=None,
            app_env="docker",
            mysql_user="finpilot",
            mysql_password="not-default-password",
            langfuse_enabled=False,
        )


def test_non_local_settings_reject_missing_deepseek_key_for_all_deepseek_features():
    for feature, overrides in {
        "routing_provider": {"ai_provider": "ollama", "routing_provider": "deepseek", "routing_llm_enabled": True},
        "rag_curation_provider": {"ai_provider": "ollama", "rag_curation_provider": "deepseek", "rag_curation_enabled": True},
        "safety_response_provider": {
            "ai_provider": "ollama",
            "safety_response_provider": "deepseek",
            "safety_response_llm_enabled": True,
        },
    }.items():
        with pytest.raises(ValidationError, match=feature):
            Settings(
                _env_file=None,
                app_env="docker",
                mysql_user="finpilot",
                mysql_password="not-default-password",
                langfuse_enabled=False,
                **overrides,
            )


def test_non_local_settings_reject_default_langfuse_secret_when_enabled():
    with pytest.raises(ValidationError, match="LANGFUSE_SECRET_KEY"):
        Settings(
            _env_file=None,
            app_env="docker",
            deepseek_api_key="sk-test",
            mysql_user="finpilot",
            mysql_password="not-default-password",
            langfuse_enabled=True,
            langfuse_secret_key="sk-lf-local",
        )


def test_readyz_returns_runtime_readiness(monkeypatch):
    import importlib

    monkeypatch.setattr(settings, "otel_enabled", False)
    monkeypatch.setattr(settings, "otel_exporter_otlp_endpoint", None)
    api = importlib.import_module("finpilot.api")
    from finpilot.readiness import ReadinessCheck, RuntimeReadiness

    class FakeService:
        audit_store = object()
        rag_service = object()

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(api, "FinPilotService", FakeService)
    monkeypatch.setattr(api, "EvalRunner", lambda service, audit_store: object())
    monkeypatch.setattr(api, "KnowledgeLifecycleService", lambda rag_service: object())
    monkeypatch.setattr(
        api,
        "check_runtime_readiness",
        lambda: RuntimeReadiness(
            status="degraded",
            checks=[
                ReadinessCheck(
                    name="mysql",
                    status="failed",
                    detail="connection refused",
                )
            ],
        ),
    )

    app = api.create_app()
    response = TestClient(app).get("/readyz")

    assert response.status_code == 200
    assert response.json() == {
        "status": "degraded",
        "checks": [{"name": "mysql", "status": "failed", "detail": "connection refused"}],
    }


def test_check_runtime_readiness_combines_component_statuses(monkeypatch, tmp_path):
    import finpilot.readiness as readiness

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql: str) -> None:
            self.sql = sql

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    index_path = tmp_path / "knowledge.json"
    index_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(readiness, "connect_runtime_mysql", lambda: FakeConnection())
    monkeypatch.setattr(
        readiness,
        "_probe_http",
        lambda name, base_url, paths: readiness.ReadinessCheck(name=name, status="ok", detail=base_url),
    )
    monkeypatch.setattr(readiness.settings, "bm25_index_path", index_path)
    monkeypatch.setattr(readiness.settings, "deepseek_api_key", "sk-test")

    result = readiness.check_runtime_readiness()

    assert result.status == "ok"
    assert {check.name: check.status for check in result.checks} == {
        "mysql": "ok",
        "bm25": "ok",
        "chroma": "ok",
        "embedding": "ok",
        "reranker": "ok",
        "llm_config": "ok",
    }


def test_check_runtime_readiness_marks_disabled_optional_dependencies_ok(monkeypatch, tmp_path):
    import finpilot.readiness as readiness

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql: str) -> None:
            self.sql = sql

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    index_path = tmp_path / "knowledge.json"
    index_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(readiness, "connect_runtime_mysql", lambda: FakeConnection())
    monkeypatch.setattr(readiness.settings, "bm25_index_path", index_path)
    monkeypatch.setattr(readiness.settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(readiness.settings, "vector_enabled", False)
    monkeypatch.setattr(readiness.settings, "reranker_enabled", False)

    result = readiness.check_runtime_readiness()

    assert result.status == "ok"
    assert next(check for check in result.checks if check.name == "chroma").detail == "disabled"
    assert next(check for check in result.checks if check.name == "reranker").detail == "disabled"


def test_check_runtime_readiness_marks_failed_for_required_dependency(monkeypatch):
    import finpilot.readiness as readiness

    monkeypatch.setattr(readiness, "connect_runtime_mysql", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(
        readiness,
        "_probe_http",
        lambda name, base_url, paths: readiness.ReadinessCheck(name=name, status="ok", detail=base_url),
    )
    monkeypatch.setattr(readiness.settings, "deepseek_api_key", "sk-test")

    result = readiness.check_runtime_readiness()

    assert result.status == "failed"
    assert next(check for check in result.checks if check.name == "mysql").detail == "RuntimeError: down"
