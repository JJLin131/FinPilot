from __future__ import annotations

from finpilot.evals.environment import RuntimeEnvironmentChecker
from finpilot.readiness import ReadinessCheck, RuntimeReadiness


def test_environment_checker_maps_runtime_dependencies_and_caches_probe():
    calls = 0

    def readiness_provider():
        nonlocal calls
        calls += 1
        return RuntimeReadiness(
            status="degraded",
            checks=[
                ReadinessCheck(name="bm25", status="ok", detail="ready"),
                ReadinessCheck(name="chroma", status="degraded", detail="connection refused"),
                ReadinessCheck(name="embedding", status="ok", detail="ready"),
                ReadinessCheck(name="reranker", status="degraded", detail="connection refused"),
                ReadinessCheck(name="mysql", status="failed", detail="connection refused"),
                ReadinessCheck(name="llm_config", status="ok", detail="configured"),
            ],
        )

    checker = RuntimeEnvironmentChecker(
        readiness_provider=readiness_provider,
        feature_flags={"langfuse": False, "otel": False, "audit": True},
        package_checker=lambda package: package != "ragas",
    )

    first = checker(("bm25", "chroma", "embedding", "reranker", "llm", "mysql", "ragas"))
    second = checker(("planner", "langfuse", "otel", "audit"))

    assert first == ["chroma", "reranker", "mysql", "ragas"]
    assert second == ["langfuse", "otel"]
    assert calls == 1


def test_environment_checker_treats_agent_planner_and_query_rewriter_as_llm_dependencies():
    readiness = RuntimeReadiness(
        status="failed",
        checks=[ReadinessCheck(name="llm_config", status="failed", detail="missing key")],
    )
    checker = RuntimeEnvironmentChecker(
        readiness_provider=lambda: readiness,
        feature_flags={"langfuse": True, "otel": True, "audit": True},
        package_checker=lambda package: True,
    )

    assert checker(("agent", "planner", "query_rewriter")) == ["agent", "planner", "query_rewriter"]
