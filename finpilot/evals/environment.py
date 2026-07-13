from __future__ import annotations

import importlib.util
from collections.abc import Callable

from finpilot.config import settings
from finpilot.readiness import RuntimeReadiness, check_runtime_readiness


class RuntimeEnvironmentChecker:
    """把运行时 readiness 与评测 suite 的能力依赖映射为阻断项。"""

    _LLM_ALIASES = {"llm", "agent", "planner", "query_rewriter"}

    def __init__(
        self,
        *,
        readiness_provider: Callable[[], RuntimeReadiness] = check_runtime_readiness,
        feature_flags: dict[str, bool] | None = None,
        package_checker: Callable[[str], bool] | None = None,
    ):
        self.readiness_provider = readiness_provider
        self.feature_flags = feature_flags or {
            "langfuse": bool(
                settings.langfuse_enabled
                and settings.langfuse_host
                and settings.langfuse_public_key
                and settings.langfuse_secret_key
            ),
            "otel": bool(settings.otel_enabled and settings.otel_exporter_otlp_endpoint),
            "audit": bool(settings.audit_backend),
            "cost_pricing": (
                settings.ai_provider.lower() != "deepseek"
                or (
                    settings.ai_input_cost_per_million is not None
                    and settings.ai_output_cost_per_million is not None
                )
            ),
        }
        self.package_checker = package_checker or (lambda package: importlib.util.find_spec(package) is not None)
        self._checks: dict[str, str] | None = None

    def __call__(self, requirements: tuple[str, ...]) -> list[str]:
        checks = self._readiness_checks()
        missing: list[str] = []
        for requirement in requirements:
            if requirement == "controlled_backend":
                continue
            if requirement == "ragas":
                available = self.package_checker("ragas") and self.package_checker("datasets")
            elif requirement == "runtime_service":
                available = checks.get("mysql") == "ok" and checks.get("memory_encryption") == "ok"
            elif requirement in self._LLM_ALIASES:
                available = checks.get("llm_config") == "ok"
            elif requirement in self.feature_flags:
                available = self.feature_flags[requirement]
            else:
                available = checks.get(requirement) == "ok"
            if not available:
                missing.append(requirement)
        return missing

    def _readiness_checks(self) -> dict[str, str]:
        if self._checks is None:
            readiness = self.readiness_provider()
            self._checks = {check.name: check.status for check in readiness.checks}
        return self._checks
