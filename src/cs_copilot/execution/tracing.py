"""Trace and span identifiers for one tool invocation."""

from __future__ import annotations

import uuid
from typing import Mapping

from .context import ExecutionContext, _optional_str


def _new_tool_trace(ctx: ExecutionContext) -> dict[str, str | None]:
    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, dict):
        output_context = {}
    trace_id = str(output_context.get("trace_id") or state.get("trace_id") or uuid.uuid4().hex)
    if not output_context.get("trace_id"):
        output_context["trace_id"] = trace_id
    return {
        "run_id": _optional_str(output_context.get("run_id")),
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex,
        "parent_span_id": _optional_str(
            output_context.get("span_id") or output_context.get("parent_span_id")
        ),
    }


def _refresh_tool_trace(
    ctx: ExecutionContext,
    trace: Mapping[str, str | None],
) -> dict[str, str | None]:
    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, Mapping):
        return dict(trace)
    return {
        "run_id": _optional_str(output_context.get("run_id")) or trace.get("run_id"),
        "trace_id": (_optional_str(output_context.get("trace_id")) or trace.get("trace_id")),
        "span_id": trace.get("span_id"),
        "parent_span_id": trace.get("parent_span_id"),
    }
