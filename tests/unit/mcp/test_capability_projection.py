"""MCP tool metadata is a projection of the shared capability table."""

from __future__ import annotations

from cs_copilot import capabilities
from cs_copilot.mcp import profiles as mcp_profiles
from cs_copilot.mcp.facades.bootstrap import _task_actions
from cs_copilot.mcp.tools_registry import OPT_IN_GROUPS, all_specs
from cs_copilot.tracking.replay import golden_for_workflow
from cs_copilot.workflows import list_workflows


def test_spec_groups_roles_and_profiles_come_from_the_table():
    specs = all_specs(opt_in_groups=OPT_IN_GROUPS)
    assert {spec.group for spec in specs} == set(capabilities.GROUPS)
    for spec in specs:
        assert capabilities.group_for_tool(spec.mcp_name) == spec.group, spec.mcp_name
        assert spec.roles == capabilities.mcp_roles_for_tool(spec.mcp_name, spec.group)
        assert spec.profiles == capabilities.profiles_for_tool(spec.mcp_name, spec.group)


def test_every_role_can_use_its_tools_within_its_own_profile():
    for spec in all_specs(opt_in_groups=OPT_IN_GROUPS):
        for role in spec.roles:
            profile = capabilities.ROLES[capabilities.canonical_role(role)].profile
            assert profile in spec.profiles, (spec.mcp_name, role, profile)


def test_function_twins_are_registered_tools():
    names = {spec.mcp_name: spec for spec in all_specs()}
    for function, tool_name in capabilities.AGNO_FUNCTION_TWINS.items():
        assert tool_name in names, (function, tool_name)


def test_mcp_profiles_are_the_table_profiles():
    assert mcp_profiles.PROFILES is capabilities.PROFILES
    assert mcp_profiles.MCPProfile is capabilities.CapabilityProfile


def test_intended_grant_changes():
    by_name = {spec.mcp_name: spec for spec in all_specs()}
    assert {"chemoinformatician", "molecular_designer", "peptide_designer"} <= set(
        by_name["gtm_optimization"].roles
    )
    assert "report_generator" in by_name["gtm_save_density_plot"].roles
    assert "report_generator" not in by_name["gtm_optimization"].roles
    assert "chembl_downloader" in by_name["pandas_create_dataframe"].roles
    robustness = [spec for spec in all_specs() if spec.group == "robustness"]
    assert robustness and all("single_agent" not in spec.roles for spec in robustness)
    for profile, tool in (
        ("chembl-retrieval", "pandas_create_dataframe"),
        ("chemoinformatics", "gtm_optimization"),
        ("reporting", "gtm_save_density_plot"),
        ("robustness", "pandas_create_dataframe"),
    ):
        assert profile in by_name[tool].profiles
    assert "reporting" not in by_name["gtm_optimization"].profiles


def test_replay_and_bootstrap_allowlists_agree():
    for workflow in list_workflows():
        if not workflow.tasks:
            continue
        bootstrap: dict[str, set[str]] = {}
        for action in _task_actions(workflow):
            if action.get("type") == "execute_workflow_task":
                bootstrap.setdefault(action["role"], set()).update(action["tool_allowlist"])
        replay = golden_for_workflow(workflow.slug).role_tool_allowlists
        assert bootstrap == {role: set(tools) for role, tools in replay.items()}, workflow.slug
