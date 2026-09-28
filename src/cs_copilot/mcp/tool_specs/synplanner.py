"""SynPlanner MCP tool specs."""

from __future__ import annotations

from typing import List

from ..tool_adapter import ToolSpec
from .common import factory

_SYNPLANNER = factory("cs_copilot.tools.chemistry.synplanner_toolkit:SynPlannerToolkit")

_METHODS = [
    (
        "synplanner_identify_input",
        "identify_input",
        "Identify whether a retrosynthesis query is a SMILES string or molecule name.",
        True,
        True,
    ),
    (
        "synplanner_convert_name_to_smiles",
        "convert_name_to_smiles",
        "Convert a molecule name to canonical SMILES for SynPlanner input.",
        True,
        True,
    ),
    (
        "synplanner_plan_synthesis",
        "plan_synthesis",
        "Run SynPlanner retrosynthesis planning for a SMILES string or molecule name.",
        False,
        True,
    ),
    (
        "synplanner_describe_plan",
        "describe_plan",
        "Return a human-readable description of the latest SynPlanner plan.",
        True,
        False,
    ),
    (
        "synplanner_get_route_visualizations",
        "get_route_visualizations",
        "Generate or fetch route visualization artifacts for a SynPlanner plan.",
        False,
        True,
    ),
]

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name=mcp_name,
        toolkit_factory=_SYNPLANNER,
        method=method,
        summary=summary,
        read_only=read_only,
        requires_network=requires_network,
    )
    for mcp_name, method, summary, read_only, requires_network in _METHODS
]
