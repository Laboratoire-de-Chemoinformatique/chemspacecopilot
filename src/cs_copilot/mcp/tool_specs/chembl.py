"""ChEMBL MCP tool specs."""

from __future__ import annotations

from typing import List

from ..facades.workflows import workflow_policy_facade
from ..tool_adapter import ToolSpec
from .common import factory

_CHEMBL = factory("cs_copilot.tools.databases.chembl:ChemblToolkit")

SPECS: List[ToolSpec] = [
    ToolSpec(
        mcp_name="chembl_prepare_retrieval",
        toolkit_factory=workflow_policy_facade,
        method="prepare_chembl_retrieval",
        summary=(
            "Preflight a ChEMBL retrieval request. Use this read-only workflow gate "
            "before chembl_fetch_compounds to identify missing target, organism, "
            "assay-type, or mechanism clarification."
        ),
        read_only=True,
    ),
    ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=_CHEMBL,
        method="fetch_compounds",
        summary=(
            "Low-level execution tool that fetches ChEMBL bioactivity data for "
            "one or more keyword targets. For vague user requests, call "
            "chembl_prepare_retrieval first and fetch only after can_proceed=true. "
            "In MCP mode the in-process LLM-as-judge filtering is disabled; "
            "use the chembl_retrieval_judge / chembl_metadata_judge prompts "
            "if you want to perform equivalent filtering with this client."
        ),
        forces={"enable_retrieval_judge": False, "enable_metadata_judge": False},
    ),
    ToolSpec(
        mcp_name="chembl_describe_dataset",
        toolkit_factory=_CHEMBL,
        method="describe_dataset",
        summary="Return a structural summary of a previously fetched ChEMBL dataset by path.",
        read_only=True,
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
