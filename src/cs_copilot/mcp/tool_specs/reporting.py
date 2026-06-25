"""Report-generation MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.reporting import report_facade
from ..tool_adapter import ToolSpec

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="report_save_markdown",
        toolkit_factory=report_facade,
        method="save_markdown",
        summary="Save a markdown report into the session-scoped storage layout.",
    ),
    ToolSpec(
        mcp_name="report_save_rich",
        toolkit_factory=report_facade,
        method="save_rich",
        summary="Save an image-rich (HTML/PDF/Markdown) report into the session layout.",
    ),
]
