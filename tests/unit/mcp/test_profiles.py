"""Tests for static MCP profiles and catalog compatibility."""

from __future__ import annotations

import pytest

from cs_copilot.agents.contracts import ROLE_POLICIES
from cs_copilot.mcp.profiles import (
    MCPProfileError,
    get_profile,
    profile_names,
    validate_profile_registry,
    validate_workflow_profile,
)
from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.workflows import list_workflows

EXPECTED_PROFILES = (
    "bootstrap",
    "standard",
    "chembl-retrieval",
    "gtm-analysis",
    "chemoinformatics",
    "reporting",
    "molecular-design",
    "peptide-design",
    "retrosynthesis",
    "robustness",
)


def test_profile_names_are_stable_and_registry_is_valid():
    assert profile_names() == EXPECTED_PROFILES
    validate_profile_registry(all_specs())


def test_bootstrap_is_a_strict_subset_of_standard():
    bootstrap = {spec.mcp_name for spec in all_specs("bootstrap")}
    standard = {spec.mcp_name for spec in all_specs("standard")}

    assert bootstrap
    assert bootstrap < standard
    assert "mcp_bootstrap" in bootstrap
    assert "gtm_optimization" not in bootstrap


def test_unknown_profile_is_rejected():
    with pytest.raises(MCPProfileError, match="Unknown MCP profile"):
        get_profile("unbounded")


def test_each_workflow_accepts_its_declared_least_privilege_profile():
    for workflow in list_workflows():
        for profile in workflow.profiles:
            validate_workflow_profile(profile, workflow.slug)


def test_standard_profile_can_execute_every_catalog_workflow():
    for workflow in list_workflows():
        validate_workflow_profile("standard", workflow.slug)


def test_narrow_profile_mismatch_is_rejected():
    with pytest.raises(MCPProfileError, match="does not permit"):
        validate_workflow_profile("reporting", "chembl-target-retrieval")


def test_task_profiles_expose_every_required_task_tool():
    specs = {spec.mcp_name: spec for spec in all_specs()}
    for workflow in list_workflows():
        for task in workflow.tasks:
            available = {spec.mcp_name for spec in all_specs(task.profile)}
            assert set(task.required_tools) <= available, (
                workflow.slug,
                task.task_id,
                task.profile,
            )
            for tool_name in task.required_tools:
                assert task.profile in specs[tool_name].profiles
                assert task.role in specs[tool_name].roles


def test_tool_roles_match_declared_agent_roles():
    declared_roles = set(ROLE_POLICIES) | {"supervisor"}
    tool_roles = {role for spec in all_specs() for role in spec.roles}

    assert tool_roles <= declared_roles
