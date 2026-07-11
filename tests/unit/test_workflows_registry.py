"""Tests for reusable workflow contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.workflows import (
    WorkflowRegistry,
    get_workflow,
    list_workflows,
    search_workflows,
)


def test_default_registry_loads_workflow_catalog():
    slugs = {workflow.slug for workflow in list_workflows()}

    assert {
        "chembl-target-retrieval",
        "gtm-density-landscape",
        "gtm-activity-landscape",
        "chembl-to-gtm-report",
        "candidate-design-to-gtm",
        "retrosynthesis-for-candidates",
        "dataset-normalization",
        "robustness-report",
    }.issubset(slugs)

    workflow = get_workflow("chembl-to-gtm-report")
    assert workflow.recommended_prompt == "cs_copilot_workflow"
    assert "report_save_rich" in workflow.required_tools
    assert "# ChEMBL To GTM Report" in workflow.workflow_md


def test_default_registry_searches_metadata_and_tools():
    results = search_workflows("gtm activity")
    assert results[0].slug in {"gtm-activity-landscape", "chembl-to-gtm-report"}

    results = search_workflows("gtm density")
    assert results[0].slug == "gtm-density-landscape"

    results = search_workflows("robustness_export_analysis_report")
    assert results[0].slug == "robustness-report"


def test_default_workflow_tool_references_exist_in_mcp_registry():
    tool_names = {spec.mcp_name for spec in all_specs()}

    missing = {}
    for workflow in list_workflows():
        referenced = (
            set(workflow.preflight_tools)
            | set(workflow.required_tools)
            | set(workflow.optional_tools)
        )
        absent = sorted(tool for tool in referenced if tool not in tool_names)
        if absent:
            missing[workflow.slug] = absent

    assert not missing


def test_workflow_as_dict_can_include_content():
    workflow = get_workflow("candidate-design-to-gtm")

    without_content = workflow.as_dict(include_content=False)
    with_content = workflow.as_dict(include_content=True)

    assert "workflow_md" not in without_content
    assert "workflow_md" in with_content
    assert "gtm_project_data" in with_content["workflow_md"]


def test_custom_registry_rejects_missing_workflow_md(tmp_path: Path):
    (tmp_path / "broken").mkdir()

    registry = WorkflowRegistry(tmp_path)
    with pytest.raises(FileNotFoundError, match="WORKFLOW.md"):
        registry.list_workflows()


def test_custom_registry_rejects_workflow_md_without_frontmatter(tmp_path: Path):
    workflow_dir = tmp_path / "plain"
    workflow_dir.mkdir()
    (workflow_dir / "WORKFLOW.md").write_text("# Plain\n", encoding="utf-8")

    registry = WorkflowRegistry(tmp_path)
    with pytest.raises(ValueError, match="frontmatter"):
        registry.list_workflows()


def test_custom_registry_rejects_slug_directory_mismatch(tmp_path: Path):
    workflow_dir = tmp_path / "actual"
    workflow_dir.mkdir()
    (workflow_dir / "WORKFLOW.md").write_text(
        "---\nname: other\ndescription: Other workflow\n---\n\n# Other\n",
        encoding="utf-8",
    )

    registry = WorkflowRegistry(tmp_path)
    with pytest.raises(ValueError, match="must match directory"):
        registry.list_workflows()
