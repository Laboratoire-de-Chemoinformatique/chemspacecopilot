"""The shared capability table: self-consistency and the Agno-side projection."""

from __future__ import annotations

import ast
import inspect
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from cs_copilot import capabilities
from cs_copilot.agents import factories as factory_module
from cs_copilot.agents.contracts import ROLE_POLICIES

CAPABILITIES_MODULE = Path(capabilities.__file__)
_TOOLKIT_NAMES = sorted(
    {toolkit for group in capabilities.GROUPS.values() for toolkit in group.agno_toolkits}
)

# Canonical grants (the Agno factory grants). Changing a role's capabilities is
# a deliberate decision that affects both runtimes, so it is pinned here.
EXPECTED_ROLES = {
    "coordinator": ("standard", {"session", "skills"}, set(), {"workflow", "agno"}),
    "chembl_downloader": (
        "chembl-retrieval",
        {"chembl", "pandas", "skills", "session"},
        set(),
        set(),
    ),
    "gtm_agent": (
        "gtm-analysis",
        {"gtm", "pandas", "session", "skills"},
        {"save_gtm_landscape_plot", "save_gtm_plot"},
        set(),
    ),
    "chemoinformatician": (
        "chemoinformatics",
        {"chem", "gtm", "pandas", "skills", "session"},
        set(),
        set(),
    ),
    "report_generator": (
        "reporting",
        {"pandas", "report", "skills", "session"},
        {"save_gtm_landscape_plot", "save_gtm_plot", "save_rich_report", "save_markdown_report"},
        set(),
    ),
    "molecular_designer": (
        "molecular-design",
        {"molecular_design", "gtm", "chem", "pandas", "skills", "session"},
        set(),
        set(),
    ),
    "peptide_designer": (
        "peptide-design",
        {"peptide_design", "gtm", "pandas", "skills", "session"},
        {"save_gtm_landscape_plot", "save_gtm_plot"},
        set(),
    ),
    "synplanner": ("retrosynthesis", {"synplanner", "skills"}, set(), set()),
    "robustness_evaluation": (
        "robustness",
        {"pandas", "robustness", "skills", "session"},
        set(),
        set(),
    ),
    "single_agent": (
        "standard",
        {
            "chembl",
            "gtm",
            "chem",
            "molecular_design",
            "peptide_design",
            "synplanner",
            "session",
            "pandas",
            "report",
            "skills",
        },
        {"save_gtm_landscape_plot", "save_gtm_plot", "save_rich_report", "save_markdown_report"},
        {"workflow"},
    ),
}


def test_capability_module_only_imports_the_standard_library():
    tree = ast.parse(CAPABILITIES_MODULE.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules.add((node.module or "").split(".")[0])
    assert modules <= set(sys.stdlib_module_names) | {"__future__"}


def test_table_validates():
    capabilities.validate_capability_table()


def test_canonical_role_grants_are_pinned():
    actual = {
        name: (role.profile, set(role.groups), set(role.functions), set(role.mcp_groups))
        for name, role in capabilities.ROLES.items()
    }
    assert actual == EXPECTED_ROLES


def test_role_policies_expand_the_table():
    assert list(ROLE_POLICIES) == list(capabilities.ROLES)
    for role, policy in ROLE_POLICIES.items():
        assert policy.profile == capabilities.ROLES[role].profile
        assert policy.allowed_toolkits == capabilities.agno_toolkits_for_role(role)
        assert policy.allowed_functions == capabilities.agno_functions_for_role(role)


def test_every_role_fits_inside_its_profile():
    for role in capabilities.ROLES.values():
        profile = capabilities.PROFILES[role.profile]
        grants = role.groups | role.mcp_groups | capabilities.MCP_UNIVERSAL_GROUPS
        assert grants <= profile.groups, role.role
        for function in role.functions:
            assert profile.allows_tool(capabilities.AGNO_FUNCTION_TWINS[function]), function


def test_role_spellings_round_trip():
    assert capabilities.mcp_role_name("coordinator") == "supervisor"
    assert capabilities.canonical_role("supervisor") == "coordinator"
    for role in capabilities.ROLES:
        assert capabilities.canonical_role(capabilities.mcp_role_name(role)) == role
    assert len(set(capabilities.mcp_role_names())) == len(capabilities.ROLES)
    assert "coordinator" not in capabilities.mcp_role_names()


@pytest.mark.parametrize(
    ("tool_name", "group"),
    [
        ("gtm_optimization", "gtm"),
        ("mol_design_molecules", "molecular_design"),
        ("skill_fetch", "skills"),
        ("mcp_bootstrap", "workflow"),
        ("chemspace_plan_analysis", "workflow"),
        ("agno_team_run", "agno"),
        ("unknown_tool", None),
    ],
)
def test_group_for_tool_uses_prefixes_and_exceptions(tool_name, group):
    assert capabilities.group_for_tool(tool_name) == group


def test_team_delegation_is_a_supervisor_tool_of_the_standard_profile():
    assert capabilities.mcp_roles_for_tool("agno_team_run") == ("supervisor",)
    assert capabilities.profiles_for_tool("agno_team_run") == ("standard",)


def test_computed_tool_role_rules():
    assert capabilities.mcp_roles_for_tool("chemspace_plan_analysis") == (
        "supervisor",
        "gtm_agent",
        "chemoinformatician",
        "molecular_designer",
        "peptide_designer",
        "single_agent",
    )
    assert capabilities.mcp_roles_for_tool("workflow_abandon_tool_invocation") == ("supervisor",)
    assert "report_generator" in capabilities.mcp_roles_for_tool("gtm_save_density_plot")
    assert "report_generator" not in capabilities.mcp_roles_for_tool("gtm_optimization")
    assert set(capabilities.mcp_roles_for_tool("session_list_objects")) == set(
        capabilities.mcp_role_names()
    )


def test_toolkit_classes_are_owned_once_and_exist_in_the_factories():
    owners = [toolkit for group in capabilities.GROUPS.values() for toolkit in group.agno_toolkits]
    assert len(owners) == len(set(owners))
    for toolkit in owners:
        assert hasattr(factory_module, toolkit), toolkit
    for function in capabilities.AGNO_FUNCTION_TWINS:
        assert inspect.isfunction(getattr(factory_module, function)), function


def test_invalid_tables_are_rejected(monkeypatch):
    broken = dict(capabilities.ROLES)
    broken["chembl_downloader"] = replace(
        broken["chembl_downloader"], groups=frozenset({"chembl", "gtm"})
    )
    monkeypatch.setattr(capabilities, "ROLES", MappingProxyType(broken))
    with pytest.raises(capabilities.CapabilityTableError, match="not covered by profile"):
        capabilities.validate_capability_table()


def _named_dummy(name: str):
    return type(name, (), {"__init__": lambda self, *args, **kwargs: None})


def test_factory_tool_lists_equal_their_role_policy(monkeypatch):
    for name in _TOOLKIT_NAMES:
        monkeypatch.setattr(factory_module, name, _named_dummy(name))
    factory_classes = [
        cls
        for cls in vars(factory_module).values()
        if inspect.isclass(cls)
        and issubclass(cls, factory_module.BaseAgentFactory)
        and getattr(cls, "agent_type", None)
    ]
    assert {cls.agent_type for cls in factory_classes} == set(ROLE_POLICIES) - {"coordinator"}
    for cls in factory_classes:
        config = cls().get_agent_config()
        toolkits = {
            tool.__class__.__name__ for tool in config.tools if not inspect.isfunction(tool)
        }
        functions = {tool.__name__ for tool in config.tools if inspect.isfunction(tool)}
        policy = ROLE_POLICIES[cls.agent_type]
        assert toolkits == set(policy.allowed_toolkits), cls.agent_type
        assert functions == set(policy.allowed_functions), cls.agent_type
