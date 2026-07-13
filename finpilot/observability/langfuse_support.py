from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from finpilot.config import settings
from finpilot.observability.capture import record_score

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_langfuse_client():
    if not settings.langfuse_enabled:
        return None
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        logger.warning("Langfuse is enabled but API keys are missing.")
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            environment=settings.app_env,
        )
    except Exception as exc:  # pragma: no cover - defensive init
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        return None


def ensure_dataset(name: str, description: str, metadata: dict[str, Any] | None = None) -> None:
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.create_dataset(name=name, description=description, metadata=metadata or {})
    except Exception:
        return


def upsert_dataset_item(
    dataset_name: str,
    *,
    case_name: str,
    payload_input: Any,
    expected_output: Any,
    metadata: dict[str, Any] | None = None,
) -> str:
    client = get_langfuse_client()
    item_id = hashlib.sha1(f"{dataset_name}:{case_name}".encode("utf-8")).hexdigest()[:24]
    if client is None:
        return item_id
    try:
        client.create_dataset_item(
            id=item_id,
            dataset_name=dataset_name,
            input=payload_input,
            expected_output=expected_output,
            metadata=metadata or {},
        )
    except Exception:
        return item_id
    return item_id


def score_trace(trace_id: str, *, name: str, value: float | str, comment: str | None = None, metadata: dict[str, Any] | None = None) -> None:
    record_score(name, value, metadata)
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=value,
            comment=comment,
            metadata=metadata or {},
        )
    except Exception:
        return


def publish_eval_suite_scores(suite: Any, *, run_id: str) -> None:
    """Publish aggregate evaluation metrics to one deterministic Langfuse run trace."""
    client = get_langfuse_client()
    if client is None:
        return
    try:
        trace_id = client.create_trace_id(seed=f"finpilot-eval:{run_id}")
        metadata = {
            "schema_version": int(getattr(suite, "schema_version", 2)),
            "suite": str(suite.suite),
            "mode": str(suite.mode),
            "status": str(getattr(suite.status, "value", suite.status)),
            "total_cases": int(suite.total_cases),
            "passed_cases": int(suite.passed_cases),
        }
        with client.start_as_current_span(
            trace_context={"trace_id": trace_id},
            name=f"eval.{suite.suite}",
            input={"suite": suite.suite, "mode": suite.mode},
            output={"status": metadata["status"], "metrics": dict(suite.metrics)},
            metadata=metadata,
        ):
            pass_rate = suite.passed_cases / suite.total_cases if suite.total_cases else 0.0
            scores = {"pass_rate": pass_rate, **dict(suite.metrics)}
            for metric_name, value in scores.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                client.create_score(
                    trace_id=trace_id,
                    name=f"eval.{suite.suite}.{metric_name}",
                    value=float(value),
                    metadata=metadata,
                )
    except Exception as exc:
        logger.warning("Failed to publish evaluation scores to Langfuse: %s", exc)


def sync_local_datasets(root: Path) -> None:
    client = get_langfuse_client()
    if client is None or not root.exists():
        return
    for dataset_path in sorted(root.glob("*.jsonl")):
        dataset_name = dataset_path.stem
        ensure_dataset(
            dataset_name,
            description=f"Local regression dataset synchronized from {dataset_path.name}",
            metadata={"source": "git", "path": str(dataset_path)},
        )
        with dataset_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                input_fields = {
                    key: payload[key]
                    for key in ("query", "question", "user_message", "turns", "prompt")
                    if key in payload
                }
                common_fields = {
                    "suite", "case_id", "name", "description", "tags", "execution_mode",
                    "severity", "timeout_seconds", "repetitions", "fixtures",
                }
                upsert_dataset_item(
                    dataset_name,
                    case_name=payload["name"],
                    payload_input={"case_id": payload["case_id"], **input_fields},
                    expected_output={
                        key: value
                        for key, value in payload.items()
                        if key not in common_fields and key not in input_fields
                    },
                    metadata={
                        "schema_version": 2,
                        "suite": payload["suite"],
                        "execution_mode": payload.get("execution_mode", "controlled"),
                        "severity": payload.get("severity", "medium"),
                    },
                )


def run_experiment(
    experiment_name: str,
    *,
    run_name: str,
    description: str,
    cases: list[dict[str, Any]],
    task,
    evaluators: list,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    client = get_langfuse_client()
    if client is None:
        return None
    try:
        return client.run_experiment(
            name=experiment_name,
            run_name=run_name,
            description=description,
            data=cases,
            task=task,
            evaluators=evaluators,
            metadata={k: str(v) for k, v in (metadata or {}).items()},
        )
    except Exception as exc:
        logger.warning("Langfuse experiment run failed: %s", exc)
        return None


def build_run_name(suite: str) -> str:
    return f"{suite}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"

