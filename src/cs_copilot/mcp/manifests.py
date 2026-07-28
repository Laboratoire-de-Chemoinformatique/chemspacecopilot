"""Record MCP tool observations in the authoritative workflow event stream."""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Callable, Mapping, Sequence

from cs_copilot.storage import OUTPUT_CONTEXT_KEY, S3

logger = logging.getLogger(__name__)

_SECRET_RE = re.compile(r"(token|secret|password|api[_-]?key|authorization)", re.I)
_ACTIVE_TOOL_SPANS: set[str] = set()
_ACTIVE_TOOL_SPANS_LOCK = threading.Lock()
_TERMINAL_PROGRESS_STAGES = frozenset(
    {
        "abandoned",
        "cache_hit",
        "cancelled",
        "completed",
        "failed",
        "result_accepted",
    }
)


def is_tool_span_active(span_id: str) -> bool:
    """Return whether this server process is still executing ``span_id``."""

    with _ACTIVE_TOOL_SPANS_LOCK:
        return str(span_id) in _ACTIVE_TOOL_SPANS


def record_tool_call(
    *,
    ctx: Any,
    tool_name: str,
    public_args: Mapping[str, Any],
    forced_args: Mapping[str, Any],
    status: str,
    duration_ms: float,
    result: Any = None,
    error: str | None = None,
    execution_scope: Mapping[str, Any] | None = None,
) -> str | None:
    """Append one redacted ``tool_call_recorded`` v2 workflow event.

    Recording is observational and must never turn a successful scientific
    tool call into a failure. Context-free unit calls are therefore ignored,
    and persistence failures are logged.
    """

    resolved = _resolve_run_context(ctx)
    if resolved is None:
        return None
    run_context, output_context = resolved
    task_id, role, profile = _resolved_execution_scope(
        ctx,
        run_context,
        tool_name,
        execution_scope,
    )
    metrics = _result_metrics(result)

    payload: dict[str, Any] = {
        "runtime": "mcp",
        "session_id": str(output_context.get("session_id") or ""),
        "run_id": str(output_context.get("run_id") or ""),
        "workflow_slug": str(output_context.get("workflow_slug") or ""),
        "trace_id": _trace_value(result, "trace_id") or output_context.get("trace_id"),
        "span_id": _trace_value(result, "span_id") or output_context.get("span_id"),
        "parent_span_id": (
            _trace_value(result, "parent_span_id") or output_context.get("parent_span_id")
        ),
        "tool_name": tool_name,
        "task_id": task_id,
        "role": role,
        "profile": profile,
        "status": status,
        "duration_ms": round(float(duration_ms), 3),
        "attempts": int(metrics.get("attempts") or 0),
        "retries": int(metrics.get("retries") or 0),
        "cached": bool(metrics.get("cached", False)),
        "public_args": _redact(public_args),
        "forced_args": _redact(forced_args),
        "output_summary": _summarize(result),
    }
    idempotency_fingerprint = public_args.get("idempotency_key")
    if idempotency_fingerprint:
        payload["idempotency_fingerprint"] = str(idempotency_fingerprint)
    if error:
        payload["error"] = _short_text(error, max_chars=1000)

    try:
        event = run_context.append_event("tool_call_recorded", payload)
        return S3.path(run_context.layout.event_rel_path(event.event_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record MCP tool event for %s: %s", tool_name, exc)
        return None


def record_tool_progress(
    *,
    ctx: Any,
    tool_name: str,
    trace: Mapping[str, Any],
    stage: str,
    attempt: int,
    max_attempts: int,
    message: str | None = None,
    cached: bool = False,
    execution_scope: Mapping[str, Any] | None = None,
    precondition: Callable[[Any, Sequence[Any]], None] | None = None,
    required: bool = False,
) -> str | None:
    """Append an observational execution-progress event for an MCP tool."""

    resolved = _resolve_run_context(ctx)
    if resolved is None:
        return None
    run_context, output_context = resolved
    task_id, role, profile = _resolved_execution_scope(
        ctx,
        run_context,
        tool_name,
        execution_scope,
    )
    payload: dict[str, Any] = {
        "runtime": "mcp",
        "session_id": str(output_context.get("session_id") or ""),
        "run_id": str(output_context.get("run_id") or ""),
        "workflow_slug": str(output_context.get("workflow_slug") or ""),
        "trace_id": trace.get("trace_id") or output_context.get("trace_id"),
        "span_id": trace.get("span_id") or output_context.get("span_id"),
        "parent_span_id": (trace.get("parent_span_id") or output_context.get("parent_span_id")),
        "tool_name": tool_name,
        "task_id": task_id,
        "role": role,
        "profile": profile,
        "stage": str(stage),
        "attempt": max(0, int(attempt)),
        "max_attempts": max(1, int(max_attempts)),
        "cached": bool(cached),
    }
    if execution_scope is not None:
        payload["task_attempt"] = execution_scope.get("task_attempt")
        payload["handoff_id"] = execution_scope.get("handoff_id")
    if message:
        payload["message"] = _short_text(message, max_chars=1000)

    span_id = str(payload.get("span_id") or "")
    try:
        event = run_context.append_event(
            "tool_progress",
            payload,
            precondition=precondition,
        )
        if span_id:
            with _ACTIVE_TOOL_SPANS_LOCK:
                if stage == "started":
                    _ACTIVE_TOOL_SPANS.add(span_id)
                elif stage in _TERMINAL_PROGRESS_STAGES:
                    _ACTIVE_TOOL_SPANS.discard(span_id)
        return S3.path(run_context.layout.event_rel_path(event.event_id))
    except Exception as exc:  # noqa: BLE001
        if span_id and stage in _TERMINAL_PROGRESS_STAGES:
            # The scientific execution has ended even when its terminal event
            # could not be persisted. A supervisor may now reconcile the
            # orphaned durable "started" event explicitly.
            with _ACTIVE_TOOL_SPANS_LOCK:
                _ACTIVE_TOOL_SPANS.discard(span_id)
        if required:
            raise
        logger.warning("Failed to record MCP progress event for %s: %s", tool_name, exc)
        return None


def _resolve_run_context(ctx: Any):
    session_state = getattr(ctx, "session_state", None)
    if not isinstance(session_state, dict):
        return None
    output_context = session_state.get(OUTPUT_CONTEXT_KEY)
    if not isinstance(output_context, dict):
        return None
    try:
        run_context = getattr(ctx, "run_context", None)
        if run_context is None:
            run_context = _load_or_create_run_context(session_state, output_context)
            ctx.run_context = run_context
        return run_context, output_context
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to resolve MCP run context: %s", exc)
        return None


def _execution_scope(ctx: Any, run_context: Any, tool_name: str):
    state = getattr(ctx, "session_state", None)
    if not isinstance(state, dict):
        state = {}
    task_id = _optional_text(state.get("active_task_id"))
    role = _optional_text(state.get("active_role"))
    profile = _optional_text(state.get("active_profile"))
    run = getattr(run_context, "run", None)
    task = run.tasks.get(task_id) if run is not None and task_id is not None else None
    if task is not None:
        # Persist the immutable task assignment, not mutable session-state
        # mirrors that a toolkit may have changed.
        role = _optional_text(task.role)
        profile = _optional_text(task.profile)
    if tool_name.startswith("workflow_"):
        role = "supervisor"
        profile = _optional_text(state.get("mcp_profile")) or profile
    else:
        profile = profile or _optional_text(state.get("mcp_profile"))
    return task_id, role, profile


def _resolved_execution_scope(
    ctx: Any,
    run_context: Any,
    tool_name: str,
    supplied: Mapping[str, Any] | None,
):
    if supplied is None:
        return _execution_scope(ctx, run_context, tool_name)
    return (
        _optional_text(supplied.get("task_id")),
        _optional_text(supplied.get("role")),
        _optional_text(supplied.get("profile")),
    )


def _result_metrics(result: Any) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    metrics = result.get("metrics")
    return metrics if isinstance(metrics, Mapping) else {}


def _optional_text(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None


def _load_or_create_run_context(
    session_state: dict[str, Any],
    output_context: Mapping[str, Any],
):
    from cs_copilot.workflows import EventReplayError, RunContext

    try:
        return RunContext.from_session_state(session_state)
    except (FileNotFoundError, EventReplayError):
        return RunContext.create(
            str(output_context.get("workflow_slug") or "mcp-session"),
            session_state=session_state,
            session_id=str(output_context.get("session_id") or "") or None,
            run_id=str(output_context.get("run_id") or "") or None,
            trace_id=str(output_context.get("trace_id") or "") or None,
        )


def _trace_value(result: Any, key: str) -> str | None:
    if not isinstance(result, Mapping):
        return None
    trace = result.get("trace")
    if not isinstance(trace, Mapping):
        return None
    value = trace.get(key)
    return str(value) if value not in (None, "") else None


def _redact(value: Any) -> Any:
    return _json_safe(value, redact=True)


def _summarize(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "none"}
    if isinstance(value, Mapping):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": [str(key) for key in list(value)[:20]],
        }
        for key in (
            "status",
            "schema_version",
            "artifact_ids",
            "warnings",
            "error",
            "metrics",
        ):
            if key in value:
                summary[key] = _json_safe(value[key])
        data = value.get("data")
        if isinstance(data, Mapping):
            summary["data_keys"] = [str(key) for key in list(data)[:20]]
        return summary
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "length": len(value)}
    if isinstance(value, str):
        return {"type": "str", "preview": _short_text(value)}
    return {"type": type(value).__name__, "preview": _short_text(value)}


def _json_safe(value: Any, *, redact: bool = False, key: str | None = None, depth: int = 0) -> Any:
    if redact and key and _SECRET_RE.search(key):
        return "<redacted>"
    if depth > 4:
        return _short_text(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= 50:
                out["_truncated"] = True
                break
            str_key = str(item_key)
            out[str_key] = _json_safe(
                item_value,
                redact=redact,
                key=str_key,
                depth=depth + 1,
            )
        return out
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        out = [_json_safe(item, redact=redact, depth=depth + 1) for item in values[:50]]
        if len(values) > 50:
            out.append({"_truncated": len(values) - 50})
        return out
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return _short_text(value)


def _short_text(value: Any, *, max_chars: int = 240) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
