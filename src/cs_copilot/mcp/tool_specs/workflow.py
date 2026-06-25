"""Workflow policy and catalog MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.bootstrap import mcp_bootstrap_facade
from ..facades.workflows import workflow_catalog_facade, workflow_policy_facade
from ..tool_adapter import ToolSpec

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="mcp_bootstrap",
        toolkit_factory=mcp_bootstrap_facade,
        method="bootstrap",
        summary=(
            "Recommend the MCP-native prompt, workflow contract, skills, "
            "preflight tools, and ordered action plan for a user request."
        ),
        read_only=True,
    ),
    ToolSpec(
        mcp_name="chemspace_plan_analysis",
        toolkit_factory=workflow_policy_facade,
        method="plan_chemical_space_analysis",
        summary=(
            "Validate a chemical-space analysis plan before ChEMBL, GTM, "
            "chemotype, or report tools. You classify the analysis_intents and "
            "dataset_source; this read-only gate checks both are present and maps "
            "intents to recommended tools."
        ),
        read_only=True,
    ),
    ToolSpec(
        mcp_name="workflow_list",
        toolkit_factory=workflow_catalog_facade,
        method="list",
        summary="List reusable cs_copilot workflow contracts from the local catalog.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="workflow_search",
        toolkit_factory=workflow_catalog_facade,
        method="search",
        summary="Search reusable cs_copilot workflow contracts by metadata and tool names.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="workflow_fetch",
        toolkit_factory=workflow_catalog_facade,
        method="fetch",
        summary="Fetch one reusable cs_copilot workflow contract, including its markdown body.",
        read_only=True,
    ),
]
