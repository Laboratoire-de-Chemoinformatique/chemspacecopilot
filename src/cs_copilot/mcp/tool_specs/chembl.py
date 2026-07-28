"""ChEMBL MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.chembl import chembl_mcp_facade
from ..facades.workflows import workflow_policy_facade
from ..tool_adapter import ToolSpec
from .common import factory

_CHEMBL = factory("cs_copilot.tools.databases.chembl:ChemblToolkit")
_CHEMBL_MCP = chembl_mcp_facade

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=workflow_policy_facade,
        method="prepare_chembl_retrieval",
        summary=(
            "Contract-recording preflight gate that validates a ChEMBL retrieval before "
            "chembl_fetch_compounds. You extract the target (gene symbol, protein name, "
            "ChEMBL id, or organism), organism, assay_types, and mechanism from the "
            "request and pass them in; the gate checks completeness and returns "
            "clarifying questions for anything missing. Do not infer fields just to pass."
        ),
        write_scope="session",
        result_artifact_type="retrieval_plan",
    ),
    ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=_CHEMBL_MCP,
        method="fetch_compounds",
        summary=(
            "Low-level execution tool that fetches ChEMBL bioactivity data for "
            "one or more keyword targets. For vague user requests, call "
            "chembl_prepare_retrieval first and fetch only after can_proceed=true. "
            "MCP LLM policy controls LLM-as-judge behavior: external creates "
            "client-side judge tasks, agno-model uses the configured model "
            "in-process, and disabled skips LLM judging."
        ),
        run_in_worker_process=True,
        worker_timeout_s=900,
        requires_network=True,
    ),
    ToolSpec(
        mcp_name="chembl_create_external_judge_task",
        toolkit_factory=_CHEMBL_MCP,
        method="create_external_judge_task",
        summary=(
            "Create a pending ChEMBL LLM-as-judge task using the same retrieval "
            "or metadata prompt templates as the in-process judge."
        ),
    ),
    ToolSpec(
        mcp_name="chembl_submit_external_judge_result",
        toolkit_factory=_CHEMBL_MCP,
        method="submit_external_judge_result",
        summary=(
            "Validate and submit external ChEMBL LLM-as-judge decisions for a "
            "pending MCP LLM task."
        ),
    ),
    ToolSpec(
        mcp_name="chembl_describe_dataset",
        toolkit_factory=_CHEMBL,
        method="describe_dataset",
        summary="Return a structural summary of a previously fetched ChEMBL dataset by path.",
        read_only=True,
        read_artifact_fields=("path_to_dataset",),
    ),
    ToolSpec(
        mcp_name="chembl_convert_to_chembl_query",
        toolkit_factory=_CHEMBL,
        method="convert_to_chembl_query",
        summary=(
            "Rewrite a free-form natural language query into the canonical "
            "ChEMBL keyword form accepted by chembl_fetch_compounds."
        ),
        read_only=True,
    ),
]
