"""Workflow policy and catalog MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.bootstrap import mcp_bootstrap_facade
from ..facades.workflows import (
    workflow_catalog_facade,
    workflow_policy_facade,
    workflow_runtime_facade,
)
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
            "dataset_source; this contract-recording gate checks both are present, "
            "maps intents to recommended tools, and persists the plan artifact."
        ),
        write_scope="session",
        result_artifact_type="gtm_plan",
        roles=(
            "supervisor",
            "gtm_agent",
            "chemoinformatician",
            "molecular_designer",
            "peptide_designer",
            "single_agent",
        ),
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
    ToolSpec(
        mcp_name="workflow_start_run",
        toolkit_factory=workflow_runtime_facade,
        method="start_run",
        summary=(
            "Start or idempotently reuse a profile-compatible catalog workflow run "
            "without abandoning active work, while pinning supplied root workflow inputs."
        ),
        idempotent=True,
    ),
    ToolSpec(
        mcp_name="workflow_get_run",
        toolkit_factory=workflow_runtime_facade,
        method="get_run",
        summary="Read the active v2 workflow-run manifest snapshot.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="workflow_abandon_tool_invocation",
        toolkit_factory=workflow_runtime_facade,
        method="abandon_tool_invocation",
        summary=(
            "Supervisor crash recovery: mark one orphaned domain-tool span abandoned "
            "only after explicitly confirming no process or worker is still running it."
        ),
        destructive=True,
        risk="high",
        roles=("supervisor",),
    ),
    ToolSpec(
        mcp_name="workflow_add_task",
        toolkit_factory=workflow_runtime_facade,
        method="add_task",
        summary="Add one role/profile-scoped task to the active workflow run.",
    ),
    ToolSpec(
        mcp_name="workflow_transition_run",
        toolkit_factory=workflow_runtime_facade,
        method="transition_run",
        summary="Apply a validated lifecycle transition to the active workflow run.",
    ),
    ToolSpec(
        mcp_name="workflow_transition_task",
        toolkit_factory=workflow_runtime_facade,
        method="transition_task",
        summary="Apply a validated lifecycle transition to one workflow task.",
    ),
    ToolSpec(
        mcp_name="workflow_record_handoff",
        toolkit_factory=workflow_runtime_facade,
        method="record_handoff",
        summary="Validate and persist a minimal structured handoff between roles.",
    ),
    ToolSpec(
        mcp_name="workflow_register_artifact",
        toolkit_factory=workflow_runtime_facade,
        method="register_artifact",
        summary=(
            "Checksum and register one run-contained artifact with provenance "
            "and trust metadata."
        ),
    ),
    ToolSpec(
        mcp_name="workflow_verify_artifact",
        toolkit_factory=workflow_runtime_facade,
        method="verify_artifact",
        summary="Verify the checksum and size of one registered run artifact.",
        read_only=True,
    ),
    ToolSpec(
        mcp_name="workflow_complete_run",
        toolkit_factory=workflow_runtime_facade,
        method="complete_run",
        summary=(
            "Complete the active run, or mark it partial when required tasks "
            "or artifacts are missing."
        ),
    ),
]
