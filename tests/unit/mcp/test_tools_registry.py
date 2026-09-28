"""Tests for the MCP tools registry — naming and uniqueness invariants."""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from cs_copilot.mcp.tools_registry import (
    all_specs,
    required_permissions_for_spec,
    validate_workflow_permissions,
)
from cs_copilot.workflows import get_workflow, list_workflows

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_every_spec_has_valid_mcp_name():
    bad = [spec.mcp_name for spec in all_specs() if not _NAME_RE.match(spec.mcp_name)]
    assert not bad, f"Invalid MCP tool names: {bad!r}"


def test_mcp_names_are_unique():
    names = [spec.mcp_name for spec in all_specs()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"Duplicate MCP tool names: {duplicates!r}"


def test_chembl_fetch_uses_policy_facade_without_static_judge_forces():
    spec = next(s for s in all_specs() if s.mcp_name == "chembl_fetch_compounds")
    assert spec.forces == {}
    assert spec.method == "fetch_compounds"
    assert spec.run_in_worker_process is True
    assert "chembl_prepare_retrieval" in spec.summary


def test_gtm_density_plot_tool_is_exposed_for_mcp():
    spec = next(s for s in all_specs() if s.mcp_name == "gtm_save_density_plot")
    assert spec.method == "save_density_plot"
    assert spec.read_only is False
    assert "density landscape plot" in spec.summary


def test_every_method_resolves_against_its_toolkit():
    for spec in all_specs():
        instance = spec.toolkit_factory()
        bound = getattr(instance, spec.method, None)
        assert callable(bound), f"{spec.mcp_name}: missing method {spec.method!r}"


def test_every_spec_has_explicit_safety_hints():
    for spec in all_specs():
        assert isinstance(spec.read_only, bool), spec.mcp_name
        assert isinstance(spec.destructive, bool), spec.mcp_name
        assert isinstance(spec.open_world, bool), spec.mcp_name
        assert isinstance(spec.run_in_worker_process, bool), spec.mcp_name


def test_result_artifacts_are_declared_as_session_writes():
    for spec in all_specs():
        if spec.result_artifact_type is None:
            continue
        assert spec.read_only is False, spec.mcp_name
        assert spec.write_scope == "session", spec.mcp_name


def test_review_sensitive_tools_are_classified_conservatively():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert specs["chembl_describe_dataset"].read_only is True
    assert specs["chembl_prepare_retrieval"].read_only is False
    assert specs["chembl_prepare_retrieval"].write_scope == "session"
    assert specs["chemspace_plan_analysis"].read_only is False
    assert specs["chemspace_plan_analysis"].write_scope == "session"
    assert specs["chem_calculate_tanimoto_similarity"].read_only is True
    assert specs["session_resolve_candidate_set"].read_only is True
    assert specs["robustness_generate_insights"].read_only is True

    assert specs["chembl_fetch_compounds"].read_only is False
    assert specs["gtm_optimization"].read_only is False
    assert specs["gtm_sample_nodes"].read_only is False
    assert specs["session_select_session_object"].read_only is False
    assert specs["session_summarize_session_memory"].read_only is False
    assert specs["report_save_markdown"].read_only is False
    assert specs["robustness_export_analysis_report"].read_only is False


def test_direct_mcp_parity_tool_names_are_present():
    names = {spec.mcp_name for spec in all_specs()}

    expected = {
        "skill_list",
        "skill_search",
        "skill_fetch",
        "mcp_bootstrap",
        "llm_create_task",
        "llm_list_pending_tasks",
        "llm_get_task",
        "llm_submit_task_result",
        "llm_cancel_task",
        "pandas_load_dataframe_from_session",
        "pandas_create_dataframe",
        "pandas_run_operation",
        "pandas_normalize_for_analysis",
        "mol_list_design_engines",
        "mol_design_molecules",
        "mol_generate_analogs",
        "mol_interpolate_molecules",
        "mol_validate_design_candidates",
        "mol_rank_design_candidates",
        "mol_register_design_candidates",
        "peptide_list_design_engines",
        "peptide_design_peptides",
        "peptide_generate_analogs",
        "peptide_design_interpolation",
        "peptide_validate_design_candidates",
        "peptide_rank_design_candidates",
        "peptide_load_design_candidates",
        "peptide_validate_model_loaded",
        "peptide_get_latent_dimension",
        "peptide_encode_peptides",
        "peptide_decode_latent",
        "peptide_sample_peptides",
        "peptide_interpolate_peptides",
        "peptide_reconstruct_sequence",
        "peptide_explore_latent_neighborhood",
        "peptide_get_model_info",
        "synplanner_identify_input",
        "synplanner_convert_name_to_smiles",
        "synplanner_plan_synthesis",
        "synplanner_describe_plan",
        "synplanner_get_route_visualizations",
        "chembl_prepare_retrieval",
        "chembl_create_external_judge_task",
        "chembl_submit_external_judge_result",
        "chemspace_plan_analysis",
        "gtm_save_density_plot",
        "workflow_list",
        "workflow_search",
        "workflow_fetch",
    }

    assert expected.issubset(names)


def test_every_spec_has_discoverability_group():
    expected_groups = {
        "chembl",
        "gtm",
        "chem",
        "session",
        "report",
        "robustness",
        "skills",
        "llm",
        "pandas",
        "molecular_design",
        "peptide_design",
        "synplanner",
        "workflow",
    }

    groups = {spec.group for spec in all_specs()}

    assert groups == expected_groups


def test_control_and_shared_tool_roles_include_supervisor():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert "supervisor" in specs["workflow_start_run"].roles
    assert "supervisor" in specs["workflow_transition_task"].roles
    assert "supervisor" in specs["skill_fetch"].roles
    assert "gtm_agent" in specs["skill_fetch"].roles
    assert "chembl_downloader" in specs["llm_get_task"].roles
    assert "supervisor" in specs["session_list_session_objects"].roles


def test_new_direct_tool_safety_hints_are_classified():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert specs["skill_fetch"].read_only is True
    assert specs["workflow_list"].read_only is True
    assert specs["workflow_search"].read_only is True
    assert specs["workflow_fetch"].read_only is True
    assert specs["mcp_bootstrap"].read_only is True
    assert specs["llm_list_pending_tasks"].read_only is True
    assert specs["llm_get_task"].read_only is True
    assert specs["mol_list_design_engines"].read_only is True
    assert specs["mol_validate_design_candidates"].read_only is True
    assert specs["mol_rank_design_candidates"].read_only is True
    assert specs["peptide_validate_design_candidates"].read_only is True
    assert specs["peptide_rank_design_candidates"].read_only is True
    assert specs["peptide_encode_peptides"].read_only is True
    assert specs["peptide_decode_latent"].read_only is True
    assert specs["synplanner_identify_input"].read_only is True
    assert specs["synplanner_describe_plan"].read_only is True

    assert specs["pandas_create_dataframe"].read_only is False
    assert specs["pandas_run_operation"].read_only is False
    assert specs["llm_create_task"].read_only is False
    assert specs["llm_submit_task_result"].read_only is False
    assert specs["llm_cancel_task"].read_only is False
    assert specs["chembl_create_external_judge_task"].read_only is False
    assert specs["chembl_submit_external_judge_result"].read_only is False
    assert specs["mol_design_molecules"].read_only is False
    assert specs["mol_generate_analogs"].read_only is False
    assert specs["mol_register_design_candidates"].read_only is False
    assert specs["peptide_design_peptides"].read_only is False
    assert specs["peptide_sample_peptides"].read_only is False
    assert specs["synplanner_plan_synthesis"].read_only is False
    assert specs["synplanner_get_route_visualizations"].read_only is False


def test_network_capabilities_are_open_world_and_high_risk():
    network_specs = [spec for spec in all_specs() if spec.requires_network]

    assert network_specs
    assert all(spec.open_world for spec in network_specs)
    assert all(spec.risk == "high" for spec in network_specs)


def test_tool_capabilities_map_to_workflow_permissions():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert required_permissions_for_spec(specs["chembl_fetch_compounds"]) == {
        "network:read",
        "artifact:write",
    }
    assert required_permissions_for_spec(specs["gtm_project_data"]) == {
        "network:read",
        "compute:execute",
        "artifact:read",
        "artifact:write",
    }
    assert required_permissions_for_spec(specs["chem_calculate_tanimoto_similarity"]) == {
        "compute:execute",
    }


def test_workflow_permission_contracts_cover_declared_capabilities():
    validate_workflow_permissions(all_specs(), workflows=list_workflows())


@pytest.mark.parametrize(
    ("workflow_slug", "permission"),
    [
        ("candidate-design-to-gtm", "network:read"),
        ("candidate-design-to-gtm", "compute:execute"),
        ("chembl-target-retrieval", "artifact:read"),
        ("chembl-target-retrieval", "artifact:write"),
    ],
)
def test_workflow_permission_validation_rejects_missing_capability(
    workflow_slug,
    permission,
):
    workflow = get_workflow(workflow_slug)
    weakened = replace(
        workflow,
        permissions=tuple(item for item in workflow.permissions if item != permission),
    )

    with pytest.raises(ValueError, match=rf"{re.escape(workflow_slug)}.*{re.escape(permission)}"):
        validate_workflow_permissions(all_specs(), workflows=[weakened])


def test_internal_source_tool_arguments_are_forced_and_hidden():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert specs["mol_design_molecules"].forces == {"_source_tool": "design_molecules"}
    assert specs["peptide_design_peptides"].forces == {"_source_tool": "design_peptides"}
