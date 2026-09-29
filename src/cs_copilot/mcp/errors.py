"""MCP names for the shared tool error taxonomy.

The taxonomy is defined once in :mod:`cs_copilot.execution.errors` and shared
with the workflow runtime and the in-process Agno runtime. These aliases are the
same objects, so ``isinstance`` checks and ``pytest.raises(MCPToolError)`` keep
working.
"""

from __future__ import annotations

from cs_copilot.execution.errors import NormalizedToolError as NormalizedMCPError
from cs_copilot.execution.errors import ToolErrorCode as MCPErrorCode
from cs_copilot.execution.errors import ToolExecutionError as MCPToolError
from cs_copilot.execution.errors import normalize_error

__all__ = ["MCPErrorCode", "MCPToolError", "NormalizedMCPError", "normalize_error"]
