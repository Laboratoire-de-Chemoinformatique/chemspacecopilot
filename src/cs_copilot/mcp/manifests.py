"""MCP entry points for recording tool observations.

The recorder is shared by every runtime and lives in
:mod:`cs_copilot.execution.events`. MCP contexts declare no runtime profile, so
their events are labelled ``runtime="mcp"`` and a lazily created ad-hoc run uses
the ``mcp-session`` workflow slug.
"""

from __future__ import annotations

from cs_copilot.execution.events import (
    TERMINAL_PROGRESS_STAGES,
    is_tool_span_active,
    record_tool_call,
    record_tool_progress,
)

__all__ = [
    "TERMINAL_PROGRESS_STAGES",
    "is_tool_span_active",
    "record_tool_call",
    "record_tool_progress",
]
