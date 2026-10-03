"""Helpers shared by execution-kernel tests."""

from __future__ import annotations

from typing import Any

from cs_copilot.execution.spec import ToolSpec


def make_spec(method: str, toolkit: type, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        mcp_name=f"kernel_{method}",
        toolkit_factory=toolkit,
        method=method,
        summary=f"Kernel {method}",
        **kwargs,
    )


def tool_events(ctx: Any) -> list[tuple[str, str | None]]:
    pairs = []
    for event in ctx.run_context.events:
        if event.event_type == "tool_progress":
            pairs.append((event.event_type, event.payload["stage"]))
        elif event.event_type == "tool_call_recorded":
            pairs.append((event.event_type, event.payload["status"]))
        elif event.event_type == "artifact_registered":
            pairs.append((event.event_type, None))
    return pairs


def payloads(ctx: Any, event_type: str) -> list[dict[str, Any]]:
    return [event.payload for event in ctx.run_context.events if event.event_type == event_type]
