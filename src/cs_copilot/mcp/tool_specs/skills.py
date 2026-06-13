"""Skill catalog MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.skills import skill_facade
from ..tool_adapter import ToolSpec

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="skill_list",
        toolkit_factory=skill_facade,
        method="list",
        summary="List reusable cs_copilot workflow skills from the local skill catalog.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="skill_search",
        toolkit_factory=skill_facade,
        method="search",
        summary="Search reusable cs_copilot workflow skills by metadata and tool names.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="skill_fetch",
        toolkit_factory=skill_facade,
        method="fetch",
        summary="Fetch one reusable cs_copilot workflow skill, including SKILL.md content.",
        read_only=True,
    ),
]
