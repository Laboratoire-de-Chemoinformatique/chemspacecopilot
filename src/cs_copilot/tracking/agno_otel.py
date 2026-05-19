"""Optional OpenTelemetry tracing for Agno via OpenInference.

This is the higher-fidelity sibling of :mod:`cs_copilot.tracking.agno_logging`.
When activated it installs OpenInference's ``AgnoInstrumentor``, which
auto-captures spans for every Agent run, Team run, model invocation, and tool
call — including the raw HTTP request_params and provider response that Agno
sends/receives over the wire.

Activation
----------
Two env vars are involved:

* ``CS_COPILOT_OTEL=1`` toggles initialisation.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` (optional): if set, spans are exported via
  OTLP/HTTP to that endpoint (Langfuse, Phoenix, OpenLIT collector, etc.).
  Use the matching ``OTEL_EXPORTER_OTLP_HEADERS`` for auth.
* If no OTLP endpoint is set, spans are written as JSON lines to
  ``$CS_COPILOT_AGNO_LOG_DIR/spans.jsonl`` (default ``./logs/agno/spans.jsonl``).

Dependencies
------------
The optional ``[otel]`` extra of ``cs_copilot`` installs the required
``openinference-instrumentation-agno`` and ``opentelemetry-*`` packages. This
module degrades gracefully (logs a warning, returns) if they aren't available.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_initialised = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    return _env_flag("CS_COPILOT_OTEL", default=False)


def _spans_file() -> Path:
    log_dir = Path(os.getenv("CS_COPILOT_AGNO_LOG_DIR", "./logs/agno")).expanduser()
    return log_dir / "spans.jsonl"


class _JsonlSpanExporter:
    """Tiny OTel SpanExporter that writes one JSON line per span.

    Used as a default sink when no OTLP endpoint is configured, so the OTel
    layer is useful even in fully offline / docker-only setups.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")

    def export(self, spans):  # type: ignore[no-untyped-def]
        try:
            from opentelemetry.sdk.trace.export import SpanExportResult
        except ImportError:
            return None
        for span in spans:
            try:
                attrs = dict(span.attributes) if span.attributes else {}
                payload = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_id": (
                        format(span.parent.span_id, "016x") if span.parent else None
                    ),
                    "start_ns": span.start_time,
                    "end_ns": span.end_time,
                    "status": str(span.status.status_code),
                    "attributes": {str(k): _coerce(v) for k, v in attrs.items()},
                }
                self._fh.write(json.dumps(payload, ensure_ascii=False, default=str))
                self._fh.write("\n")
            except Exception as exc:
                logger.warning("OTel JSONL exporter failed: %s", exc)
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self):  # type: ignore[no-untyped-def]
        try:
            self._fh.close()
        except Exception:
            pass

    def force_flush(self, timeout_millis: int = 0):  # type: ignore[no-untyped-def]
        try:
            self._fh.flush()
        except Exception:
            pass
        return True


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return str(value)


def init_otel() -> Optional[bool]:
    """Install OpenInference instrumentation for Agno (idempotent).

    Returns ``True`` if instrumentation was installed, ``False`` if disabled,
    ``None`` if the optional dependencies are missing.
    """
    global _initialised
    if not is_enabled():
        return False
    if _initialised:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
    except ImportError as exc:
        logger.warning(
            "CS_COPILOT_OTEL=1 but opentelemetry packages are missing; "
            "install the 'otel' extra: uv sync --extra otel. (%s)",
            exc,
        )
        return None

    try:
        from openinference.instrumentation.agno import AgnoInstrumentor
    except ImportError as exc:
        logger.warning(
            "CS_COPILOT_OTEL=1 but openinference-instrumentation-agno is "
            "missing; install the 'otel' extra: uv sync --extra otel. (%s)",
            exc,
        )
        return None

    resource = Resource.create({"service.name": "cs_copilot"})
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)

    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as exc:
            logger.warning(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but OTLP HTTP exporter "
                "is missing; falling back to file exporter. (%s)",
                exc,
            )
            provider.add_span_processor(SimpleSpanProcessor(_JsonlSpanExporter(_spans_file())))
        else:
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(_JsonlSpanExporter(_spans_file())))

    AgnoInstrumentor().instrument(tracer_provider=provider)
    _initialised = True
    logger.info(
        "OpenInference Agno instrumentation enabled (exporter=%s)",
        "otlp" if otlp_endpoint else f"jsonl:{_spans_file()}",
    )
    return True
