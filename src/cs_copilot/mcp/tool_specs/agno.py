"""Opt-in MCP delegation into the in-process Agno team."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="agno_team_run",
        toolkit_factory=factory("cs_copilot.mcp.agno_delegate:AgnoTeamFacade"),
        method="run",
        summary=(
            "Private trusted-client escape hatch: delegate one prompt to the "
            "cs_copilot Agno team, which reasons with the MCP server's own model "
            "(--llm-policy agno-model) inside this session's ad-hoc run. Every "
            "tool the team calls is confined, recorded, and registered like an "
            "MCP tool call. Prefer fine-grained MCP skills and tools."
        ),
        open_world=True,
        requires_network=True,
        timeout_s=1800,
        delegates_execution=True,
    ),
]
