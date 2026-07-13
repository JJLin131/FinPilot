from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservabilityCapture:
    spans: list[dict[str, Any]] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)


_CURRENT_CAPTURE: ContextVar[ObservabilityCapture | None] = ContextVar(
    "finpilot_observability_capture", default=None
)


@contextmanager
def capture_observability():
    captured = ObservabilityCapture()
    token = _CURRENT_CAPTURE.set(captured)
    try:
        yield captured
    finally:
        _CURRENT_CAPTURE.reset(token)


def record_span(name: str, attributes: dict[str, Any] | None = None) -> None:
    captured = _CURRENT_CAPTURE.get()
    if captured is not None:
        captured.spans.append({"name": name, "attributes": dict(attributes or {})})


def record_score(name: str, value: float | str, metadata: dict[str, Any] | None = None) -> None:
    captured = _CURRENT_CAPTURE.get()
    if captured is not None:
        item: dict[str, Any] = {"name": name, "value": value}
        if metadata:
            item["metadata"] = dict(metadata)
        captured.scores.append(item)


def record_audit_event(event_type: str, detail: dict[str, Any] | None = None) -> None:
    captured = _CURRENT_CAPTURE.get()
    if captured is not None:
        captured.audit_events.append({"type": event_type, "detail": dict(detail or {})})
