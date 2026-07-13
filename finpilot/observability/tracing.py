from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from finpilot.config import settings
from finpilot.observability.capture import record_span

logger = logging.getLogger(__name__)

_TRACER_NAME = "finpilot"
_INITIALIZED = False


def setup_tracing() -> None:
    global _INITIALIZED
    if _INITIALIZED or not settings.otel_enabled:
        return

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": "FinPilot",
                "service.version": "0.1.0",
                "deployment.environment": settings.app_env,
            }
        )
    )
    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _INITIALIZED = True


def tracer():
    setup_tracing()
    return trace.get_tracer(_TRACER_NAME)


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None):
    trace_attributes = attributes or {}
    try:
        with tracer().start_as_current_span(name) as current_span:
            for key, value in trace_attributes.items():
                if value is not None:
                    current_span.set_attribute(key, value)
            yield current_span
    finally:
        record_span(name, trace_attributes)


def current_trace_id() -> str:
    current = trace.get_current_span()
    span_context = current.get_span_context()
    if not span_context or not span_context.is_valid:
        return "local-trace-unavailable"
    return format(span_context.trace_id, "032x")

