"""PointerPandas MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.pandas import pointer_pandas_facade
from ..tool_adapter import ToolSpec

_METHODS = [
    (
        "pandas_load_dataframe_from_session",
        "load_dataframe_from_session",
        "Load a session DataFrame or CSV artifact into the MCP pandas registry.",
        False,
    ),
    (
        "pandas_create_dataframe",
        "create_dataframe",
        "Create a DataFrame and store it in the MCP pandas registry.",
        False,
    ),
    (
        "pandas_run_operation",
        "run_operation",
        "Run a pandas operation against a registered DataFrame.",
        False,
    ),
    (
        "pandas_normalize_for_analysis",
        "normalize_for_analysis",
        "Normalize a DataFrame for downstream cs_copilot analysis workflows.",
        False,
    ),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=pointer_pandas_facade,
        method=method,
        summary=summary,
        read_only=read_only,
    )
    for mcp_name, method, summary, read_only in _METHODS
]
