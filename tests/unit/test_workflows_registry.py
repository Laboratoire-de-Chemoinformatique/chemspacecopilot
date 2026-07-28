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
    assert workflow.version == "2.0.0"
    assert workflow.profiles == ("standard",)
    assert [task.task_id for task in workflow.tasks] == [
        "chembl-preflight",
        "chembl-retrieval",
        "gtm-preflight",
        "gtm-model",
        "gtm-landscapes",
        "report",
    ]
    assert workflow.tasks[-1].depends_on == ("chembl-retrieval", "gtm-landscapes")
    assert "html_report_path" in workflow.tasks[-1].output_artifacts
    assert all(item.version == "2.0.0" for item in list_workflows())


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


def test_explicit_missing_catalog_does_not_fall_back_to_bundled_workflows(
    tmp_path: Path,
    monkeypatch,
):
    missing = tmp_path / "missing-catalog"
    monkeypatch.setenv("CS_COPILOT_WORKFLOWS_DIR", str(missing))

    with pytest.raises(FileNotFoundError, match=str(missing)):
        WorkflowRegistry().list_workflows()


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


def _write_workflow(
    root: Path,
    slug: str,
    *,
    version: str = "1.0.0",
    depends_on=(),
    tasks: str = "",
):
    directory = root / slug
    directory.mkdir()
    dependencies = "\n".join(f"    - {item}" for item in depends_on) or "[]"
    if dependencies != "[]":
        dependencies = "\n" + dependencies
    (directory / "WORKFLOW.md").write_text(
        "---\n"
        f"name: {slug}\n"
        f"description: {slug} description\n"
        "metadata:\n"
        f"  version: {version}\n"
        f"  depends_on: {dependencies}\n"
        "  profiles:\n"
        "    - standard\n"
        "  permissions:\n"
        "    - artifact:read\n"
        "  input_artifacts:\n"
        "    - name: source\n"
        "      kind: dataset\n"
        "      required: true\n"
        "  output_artifacts:\n"
        "    - name: result\n"
        "      kind: report\n"
        "      required: true\n"
        f"{tasks}"
        "---\n\n"
        f"# {slug}\n",
        encoding="utf-8",
    )


def test_registry_rejects_invalid_semver(tmp_path: Path):
    _write_workflow(tmp_path, "invalid-version", version="1.0")

    with pytest.raises(ValueError, match="semantic versioning"):
        WorkflowRegistry(tmp_path).list_workflows()


def test_registry_rejects_missing_and_cyclic_dependencies(tmp_path: Path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    _write_workflow(missing_root, "consumer", depends_on=("absent",))
    with pytest.raises(ValueError, match="unknown workflow dependencies"):
        WorkflowRegistry(missing_root).list_workflows()

    cycle_root = tmp_path / "cycle"
    cycle_root.mkdir()
    _write_workflow(cycle_root, "alpha", depends_on=("beta",))
    _write_workflow(cycle_root, "beta", depends_on=("alpha",))
    with pytest.raises(ValueError, match="Workflow dependency cycle"):
        WorkflowRegistry(cycle_root).list_workflows()

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    _write_workflow(duplicate_root, "base")
    _write_workflow(
        duplicate_root,
        "duplicate-consumer",
        depends_on=("base", "base"),
    )
    with pytest.raises(ValueError, match="duplicate workflow dependencies"):
        WorkflowRegistry(duplicate_root).list_workflows()


def test_registry_rejects_task_artifacts_without_upstream_lineage(tmp_path: Path):
    _write_workflow(
        tmp_path,
        "invalid-lineage",
        tasks=(
            "  tasks:\n"
            "    - id: consume-own-output\n"
            "      role: report_generator\n"
            "      profile: reporting\n"
            "      depends_on: []\n"
            "      required_tools: []\n"
            "      input_artifacts:\n"
            "        - result\n"
            "      output_artifacts:\n"
            "        - result\n"
            "      acceptance_criteria:\n"
            "        - Result is supported by an upstream artifact.\n"
        ),
    )

    with pytest.raises(ValueError, match="without an upstream producer"):
        WorkflowRegistry(tmp_path).list_workflows()


def test_stdlib_yaml_fallback_parses_pilot_task_contract():
    from cs_copilot.workflows.registry import _parse_yaml_block, _split_frontmatter

    path = get_workflow("chembl-to-gtm-report").path / "WORKFLOW.md"
    frontmatter, _body = _split_frontmatter(path.read_text(encoding="utf-8"), path)
    data = _parse_yaml_block(frontmatter.splitlines(), path)

    assert data["metadata"]["permissions"][0] == "network:read"
    first_task = data["metadata"]["tasks"][0]
    assert first_task["id"] == "chembl-preflight"
    assert first_task["depends_on"] == []
    assert first_task["input_artifacts"] == ["retrieval_request"]
