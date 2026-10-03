"""One capability table for every runtime.

Role grants and capability profiles used to be declared three times: the Agno
role policies (toolkit class names), the MCP profiles (tool groups), and the
MCP tool-spec roles (group to roles). They drifted apart. This module is now the
single source of truth, and each runtime derives its view from it:

* :data:`cs_copilot.agents.contracts.ROLE_POLICIES` expands each role's groups
  into Agno toolkit class names and bare callables;
* :mod:`cs_copilot.mcp.profiles` exposes :data:`PROFILES` directly;
* :mod:`cs_copilot.mcp.tools_registry` derives every tool's roles and profiles
  from its name and group;
* workflow replay and MCP bootstrap derive optional-tool allowlists with
  :func:`allowed_tools_for`.

Grants are expressed in *groups*. A group owns the MCP tools whose names start
with its prefix (with two listed exceptions) and the Agno toolkits that
implement them. A role may also hold individual Agno callables; each maps to
its MCP twin tool, so holding one callable never grants its whole group.

The canonical grants are the in-process Agno factory grants. On the MCP side
every role additionally receives the universal ``llm`` and ``session`` groups,
and the coordinator (spelled ``supervisor`` on MCP) receives the control-plane
``workflow`` group and the opt-in ``agno`` delegation group. Every role's grants must fit inside its assigned profile;
the table is validated at import time.

This module only depends on the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping


class CapabilityTableError(ValueError):
    """Raised when the capability table is internally inconsistent."""


@dataclass(frozen=True)
class CapabilityGroup:
    """A family of MCP tools (by name prefix) and the Agno toolkits behind them."""

    name: str
    tool_prefix: str
    agno_toolkits: frozenset[str] = frozenset()


@dataclass(frozen=True)
class RoleCapabilities:
    """Canonical grants of one role.

    ``groups`` and ``functions`` are the Agno factory grants. ``mcp_groups``
    are MCP-only control-plane groups that have no Agno implementation.
    """

    role: str
    profile: str
    groups: frozenset[str]
    functions: frozenset[str] = frozenset()
    mcp_groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class CapabilityProfile:
    """One immutable deployment allowlist (MCP server profile)."""

    name: str
    description: str
    groups: frozenset[str] = frozenset()
    tools: frozenset[str] = frozenset()

    def allows_tool(self, tool_name: str, group: str | None = None) -> bool:
        """Return whether this profile exposes ``tool_name``."""

        resolved = group if group is not None else group_for_tool(tool_name)
        return tool_name in self.tools or (resolved or "") in self.groups

    def allows(self, spec: Any) -> bool:
        """Return whether this profile exposes a tool spec."""

        return spec.mcp_name in self.tools or (spec.group or "") in self.groups


def _group(name: str, prefix: str, toolkits: Iterable[str] = ()) -> CapabilityGroup:
    return CapabilityGroup(name=name, tool_prefix=prefix, agno_toolkits=frozenset(toolkits))


GROUPS: Mapping[str, CapabilityGroup] = MappingProxyType(
    {
        group.name: group
        for group in (
            _group("chembl", "chembl_", ("ChemblToolkit",)),
            _group("gtm", "gtm_", ("GTMToolkit",)),
            _group("chem", "chem_", ("ChemicalSimilarityToolkit",)),
            _group("session", "session_", ("SessionMemoryToolkit",)),
            # Agno reaches reports through bare callables (AGNO_FUNCTION_TWINS).
            _group("report", "report_"),
            # MCP-only control plane and LLM task broker.
            _group("workflow", "workflow_"),
            _group("llm", "llm_"),
            # MCP-only, opt-in: delegation into the in-process Agno team.
            _group("agno", "agno_"),
            _group("robustness", "robustness_", ("RobustnessAnalysisToolkit",)),
            _group("skills", "skill_", ("SkillToolkit",)),
            _group("pandas", "pandas_", ("PointerPandasTools",)),
            _group(
                "molecular_design",
                "mol_",
                ("MolecularDesignerToolkit", "AutoencoderToolkit"),
            ),
            _group("peptide_design", "peptide_", ("PeptideDesignerToolkit",)),
            _group("synplanner", "synplanner_", ("SynPlannerToolkit",)),
        )
    }
)
ALL_GROUPS = frozenset(GROUPS)
CORE_GROUPS = frozenset({"workflow", "skills", "session", "llm"})

# MCP tools whose names do not follow their group's prefix.
TOOL_GROUP_EXCEPTIONS: Mapping[str, str] = MappingProxyType(
    {
        "mcp_bootstrap": "workflow",
        "chemspace_plan_analysis": "workflow",
    }
)
# Bare Agno callables and the MCP tool that exposes the same operation.
AGNO_FUNCTION_TWINS: Mapping[str, str] = MappingProxyType(
    {
        "save_gtm_landscape_plot": "gtm_save_landscape_plot",
        "save_gtm_plot": "gtm_save_density_plot",
        "save_rich_report": "report_save_rich",
        "save_markdown_report": "report_save_markdown",
    }
)
# Tools also granted to every role that holds the named group. The chemical
# space preflight is the MCP twin of ``GTMToolkit.plan_analysis``.
TOOL_GROUP_GRANTS: Mapping[str, str] = MappingProxyType({"chemspace_plan_analysis": "gtm"})
# Crash-recovery operations reserved for the supervisor.
SUPERVISOR_ONLY_TOOLS = frozenset({"workflow_abandon_tool_invocation"})

COORDINATOR_ROLE = "coordinator"
SUPERVISOR_ROLE = "supervisor"
MCP_ROLE_ALIASES: Mapping[str, str] = MappingProxyType({COORDINATOR_ROLE: SUPERVISOR_ROLE})
# Groups every role receives on the MCP side.
MCP_UNIVERSAL_GROUPS = frozenset({"llm", "session"})

_GTM_PLOTS = ("save_gtm_landscape_plot", "save_gtm_plot")
_REPORT_WRITERS = ("save_rich_report", "save_markdown_report")


def _role(
    role: str,
    profile: str,
    groups: Iterable[str],
    functions: Iterable[str] = (),
    mcp_groups: Iterable[str] = (),
) -> RoleCapabilities:
    return RoleCapabilities(
        role=role,
        profile=profile,
        groups=frozenset(groups),
        functions=frozenset(functions),
        mcp_groups=frozenset(mcp_groups),
    )


ROLES: Mapping[str, RoleCapabilities] = MappingProxyType(
    {
        role.role: role
        for role in (
            _role(
                COORDINATOR_ROLE,
                "standard",
                ("session", "skills"),
                mcp_groups=("workflow", "agno"),
            ),
            _role(
                "chembl_downloader",
                "chembl-retrieval",
                ("chembl", "pandas", "session", "skills"),
            ),
            _role(
                "gtm_agent",
                "gtm-analysis",
                ("gtm", "pandas", "session", "skills"),
                _GTM_PLOTS,
            ),
            _role(
                "chemoinformatician",
                "chemoinformatics",
                ("chem", "gtm", "pandas", "session", "skills"),
            ),
            _role(
                "report_generator",
                "reporting",
                ("pandas", "report", "session", "skills"),
                _GTM_PLOTS + _REPORT_WRITERS,
            ),
            _role(
                "molecular_designer",
                "molecular-design",
                ("molecular_design", "gtm", "chem", "pandas", "session", "skills"),
            ),
            _role(
                "peptide_designer",
                "peptide-design",
                ("peptide_design", "gtm", "pandas", "session", "skills"),
                _GTM_PLOTS,
            ),
            _role("synplanner", "retrosynthesis", ("synplanner", "skills")),
            _role(
                "robustness_evaluation",
                "robustness",
                ("pandas", "robustness", "session", "skills"),
            ),
            # The flat ablation baseline deliberately excludes robustness tools.
            _role(
                "single_agent",
                "standard",
                (
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
                ),
                _GTM_PLOTS + _REPORT_WRITERS,
                mcp_groups=("workflow",),
            ),
        )
    }
)

DISCOVERY_TOOLS = frozenset(
    {
        "mcp_bootstrap",
        "workflow_list",
        "workflow_search",
        "workflow_fetch",
        "skill_list",
        "skill_search",
        "skill_fetch",
        "chembl_prepare_retrieval",
        "chemspace_plan_analysis",
    }
)


def _profile(
    name: str,
    description: str,
    groups: Iterable[str] = (),
    tools: Iterable[str] = (),
) -> CapabilityProfile:
    return CapabilityProfile(
        name=name,
        description=description,
        groups=frozenset(groups),
        tools=frozenset(tools),
    )


PROFILES: Mapping[str, CapabilityProfile] = MappingProxyType(
    {
        profile.name: profile
        for profile in (
            _profile(
                "bootstrap",
                "Catalog discovery, workflow selection, and plan-artifact-recording "
                "scientific preflight tools.",
                tools=DISCOVERY_TOOLS,
            ),
            _profile(
                "standard",
                "All stable tools required by the published workflow catalog.",
                ALL_GROUPS,
            ),
            _profile(
                "chembl-retrieval",
                "ChEMBL retrieval, external judging, tabular preparation, and session "
                "artifact handling.",
                CORE_GROUPS | {"chembl", "pandas"},
            ),
            _profile(
                "gtm-analysis",
                "GTM fitting, projection, landscapes, tabular preparation, and reporting.",
                CORE_GROUPS | {"gtm", "pandas", "report"},
            ),
            _profile(
                "chemoinformatics",
                "Similarity analysis, GTM-backed chemotype analysis, tabular "
                "normalization, session data, and reporting.",
                CORE_GROUPS | {"chem", "gtm", "pandas", "report"},
            ),
            _profile(
                "reporting",
                "Session inspection, tabular data, GTM figures, and report artifact generation.",
                CORE_GROUPS | {"pandas", "report"},
                (
                    AGNO_FUNCTION_TWINS["save_gtm_landscape_plot"],
                    AGNO_FUNCTION_TWINS["save_gtm_plot"],
                ),
            ),
            _profile(
                "molecular-design",
                "Small-molecule design, validation, analysis, GTM projection, and artifacts.",
                CORE_GROUPS | {"molecular_design", "chem", "pandas", "gtm", "report"},
            ),
            _profile(
                "peptide-design",
                "Peptide design, validation, latent-space analysis, GTM, and artifacts.",
                CORE_GROUPS | {"peptide_design", "chem", "pandas", "gtm", "report"},
            ),
            _profile(
                "retrosynthesis",
                "Candidate resolution, SynPlanner retrosynthesis, and report artifacts.",
                CORE_GROUPS | {"synplanner", "report"},
            ),
            _profile(
                "robustness",
                "Robustness result analysis, tabular data, and report export.",
                CORE_GROUPS | {"robustness", "pandas", "report"},
            ),
        )
    }
)


def canonical_role(role: str) -> str:
    """Return the canonical role name for an MCP or canonical spelling."""

    for canonical, alias in MCP_ROLE_ALIASES.items():
        if role == alias:
            return canonical
    return role


def mcp_role_name(role: str) -> str:
    """Return the MCP spelling of a canonical role."""

    return MCP_ROLE_ALIASES.get(role, role)


def mcp_role_names() -> tuple[str, ...]:
    """Return every role in its MCP spelling, in table order."""

    return tuple(mcp_role_name(role) for role in ROLES)


def group_for_tool(tool_name: str) -> str | None:
    """Return the group owning an MCP tool name, or ``None`` if none matches."""

    if tool_name in TOOL_GROUP_EXCEPTIONS:
        return TOOL_GROUP_EXCEPTIONS[tool_name]
    prefix = tool_name.split("_", 1)[0] + "_"
    for group in GROUPS.values():
        if group.tool_prefix == prefix:
            return group.name
    return None


def group_for_toolkit(class_name: str) -> str | None:
    """Return the group implemented by an Agno toolkit class name, if any."""

    for group in GROUPS.values():
        if class_name in group.agno_toolkits:
            return group.name
    return None


def _twin_tools(role: RoleCapabilities) -> frozenset[str]:
    return frozenset(AGNO_FUNCTION_TWINS[function] for function in role.functions)


def _mcp_groups_of(role: RoleCapabilities) -> frozenset[str]:
    return role.groups | role.mcp_groups | MCP_UNIVERSAL_GROUPS


def _role_holds_tool(role: RoleCapabilities, tool_name: str, group: str | None) -> bool:
    if tool_name in SUPERVISOR_ONLY_TOOLS:
        return role.role == COORDINATOR_ROLE
    if group is not None and group in _mcp_groups_of(role):
        return True
    extra_group = TOOL_GROUP_GRANTS.get(tool_name)
    if extra_group is not None and extra_group in role.groups:
        return True
    return tool_name in _twin_tools(role)


def roles_for_tool(tool_name: str, group: str | None = None) -> tuple[str, ...]:
    """Return canonical roles allowed to call ``tool_name``, in table order."""

    resolved = group if group is not None else group_for_tool(tool_name)
    return tuple(
        role.role for role in ROLES.values() if _role_holds_tool(role, tool_name, resolved)
    )


def mcp_roles_for_tool(tool_name: str, group: str | None = None) -> tuple[str, ...]:
    """Return MCP-spelled roles allowed to call ``tool_name``, in table order."""

    return tuple(mcp_role_name(role) for role in roles_for_tool(tool_name, group))


def profiles_for_tool(tool_name: str, group: str | None = None) -> tuple[str, ...]:
    """Return every profile exposing ``tool_name``, in table order."""

    resolved = group if group is not None else group_for_tool(tool_name)
    return tuple(
        profile.name for profile in PROFILES.values() if profile.allows_tool(tool_name, resolved)
    )


def tool_allowed(tool_name: str, *, role: str, profile: str, group: str | None = None) -> bool:
    """Return whether ``role`` working under ``profile`` may call ``tool_name``."""

    canonical = canonical_role(role)
    return profile in profiles_for_tool(tool_name, group) and canonical in roles_for_tool(
        tool_name, group
    )


def allowed_tools_for(role: str, profile: str, tool_names: Iterable[str]) -> tuple[str, ...]:
    """Filter ``tool_names`` to those ``role`` may call under ``profile``."""

    return tuple(
        name
        for name in dict.fromkeys(str(item) for item in tool_names)
        if tool_allowed(name, role=role, profile=profile)
    )


def agno_toolkits_for_role(role: str) -> frozenset[str]:
    """Return the Agno toolkit class names a role may be configured with."""

    return frozenset(
        toolkit for group in ROLES[role].groups for toolkit in GROUPS[group].agno_toolkits
    )


def agno_functions_for_role(role: str) -> frozenset[str]:
    """Return the bare Agno callables a role may be configured with."""

    return ROLES[role].functions


def validate_capability_table() -> None:
    """Raise :class:`CapabilityTableError` if the table is inconsistent."""

    problems: list[str] = []

    prefixes = [group.tool_prefix for group in GROUPS.values()]
    for group in GROUPS.values():
        prefix = group.tool_prefix
        if not prefix.endswith("_") or "_" in prefix[:-1] or not prefix[:-1]:
            problems.append(f"group {group.name!r} needs a single-token prefix, got {prefix!r}")
    if len(set(prefixes)) != len(prefixes):
        problems.append("group tool prefixes must be unique")
    for tool_name, group in TOOL_GROUP_EXCEPTIONS.items():
        if group not in GROUPS:
            problems.append(f"exception {tool_name!r} names unknown group {group!r}")
        if tool_name.split("_", 1)[0] + "_" in prefixes:
            problems.append(f"exception {tool_name!r} collides with a group prefix")

    owners: dict[str, str] = {}
    for group in GROUPS.values():
        for toolkit in group.agno_toolkits:
            if toolkit in owners:
                problems.append(
                    f"toolkit {toolkit!r} belongs to {owners[toolkit]!r} and {group.name!r}"
                )
            owners[toolkit] = group.name
    twin_groups = {group_for_tool(tool) for tool in AGNO_FUNCTION_TWINS.values()}
    if None in twin_groups:
        problems.append("every Agno function twin must name a tool with a known group")
    mcp_only = {
        name
        for name, group in GROUPS.items()
        if not group.agno_toolkits and name not in twin_groups
    }

    aliases = set(MCP_ROLE_ALIASES.values())
    if aliases & set(ROLES):
        problems.append("an MCP role alias must not equal a canonical role")
    if len(set(mcp_role_names())) != len(ROLES):
        problems.append("MCP role spellings must be unique")
    for group in MCP_UNIVERSAL_GROUPS | set(TOOL_GROUP_GRANTS.values()):
        if group not in GROUPS:
            problems.append(f"unknown group {group!r} in MCP grant rules")

    for role in ROLES.values():
        profile = PROFILES.get(role.profile)
        if profile is None:
            problems.append(f"role {role.role!r} uses unknown profile {role.profile!r}")
            continue
        unknown = sorted((role.groups | role.mcp_groups) - ALL_GROUPS)
        if unknown:
            problems.append(f"role {role.role!r} references unknown groups {unknown}")
        unknown_functions = sorted(role.functions - set(AGNO_FUNCTION_TWINS))
        if unknown_functions:
            problems.append(f"role {role.role!r} references unknown functions {unknown_functions}")
        misplaced = sorted(role.mcp_groups - mcp_only)
        if misplaced:
            problems.append(f"role {role.role!r} lists Agno-backed groups as MCP-only: {misplaced}")
        missing_groups = sorted(_mcp_groups_of(role) - profile.groups)
        missing_tools = sorted(tool for tool in _twin_tools(role) if not profile.allows_tool(tool))
        if missing_groups or missing_tools:
            problems.append(
                f"role {role.role!r} is not covered by profile {profile.name!r}: "
                f"groups {missing_groups}, tools {missing_tools}"
            )

    if problems:
        raise CapabilityTableError("Invalid capability table:\n  - " + "\n  - ".join(problems))


validate_capability_table()
