"""Explicit, profile-aware registry of cs_copilot MCP capabilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, List, Sequence

from cs_copilot import capabilities

from .profiles import (
    MCPProfile,
    get_profile,
    profiles_for_spec,
    validate_profile_registry,
)
from .tool_adapter import ToolSpec
from .tool_specs import (
    agno_twins,
    chembl,
    chemistry,
    design,
    gtm,
    llm,
    pandas,
    reporting,
    robustness,
    session,
    skills,
    synplanner,
    workflow,
)

_COMPUTE_GROUPS = frozenset({"chem", "gtm", "pandas", "robustness"})
_ARTIFACT_READ_PERMISSION = "artifact:read"
_ARTIFACT_WRITE_PERMISSION = "artifact:write"
_COMPUTE_PERMISSION = "compute:execute"
_NETWORK_PERMISSION = "network:read"


def _with_group(specs: Iterable[ToolSpec], group: str) -> Iterable[ToolSpec]:
    for spec in specs:
        yield replace(spec, group=spec.group or group)


def _base_specs() -> Iterable[ToolSpec]:
    yield from _with_group(chembl.SPECS, "chembl")
    yield from _with_group(gtm.SPECS, "gtm")
    yield from _with_group(chemistry.SPECS, "chem")
    yield from _with_group(session.SPECS, "session")
    yield from _with_group(reporting.SPECS, "report")
    yield from _with_group(workflow.SPECS, "workflow")
    yield from _with_group(llm.SPECS, "llm")
    yield from _with_group(robustness.SPECS, "robustness")
    yield from _with_group(skills.SPECS, "skills")
    yield from _with_group(pandas.SPECS, "pandas")
    yield from _with_group(design.MOLECULAR_SPECS, "molecular_design")
    yield from _with_group(design.PEPTIDE_SPECS, "peptide_design")
    yield from _with_group(synplanner.SPECS, "synplanner")


def _enrich(spec: ToolSpec) -> ToolSpec:
    """Fill capability policy defaults without duplicating tool declarations."""

    if spec.roles or spec.profiles:
        raise ValueError(
            f"{spec.mcp_name}: declare role and profile grants in cs_copilot/capabilities.py, "
            "not on the tool spec"
        )
    profiles = profiles_for_spec(spec)
    roles = capabilities.mcp_roles_for_tool(spec.mcp_name, spec.group)
    agno_bindings = _agno_bindings(spec)
    write_scope = spec.write_scope
    if write_scope == "none" and not spec.read_only:
        write_scope = "session"
    open_world = spec.open_world or spec.requires_network
    risk = spec.risk
    if spec.destructive or open_world or write_scope == "external":
        risk = "high"
    elif not spec.read_only and risk == "low":
        risk = "medium"
    return replace(
        spec,
        idempotent=spec.idempotent or spec.read_only,
        open_world=open_world,
        risk=risk,
        roles=tuple(dict.fromkeys(roles)),
        profiles=tuple(dict.fromkeys(profiles)),
        write_scope=write_scope,
        agno_bindings=agno_bindings,
    )


def _agno_bindings(spec: ToolSpec) -> tuple[str, ...]:
    """Name the in-process implementations of one MCP operation."""

    bindings = list(spec.agno_bindings)
    toolkit_path = getattr(spec.toolkit_factory, "toolkit_import_path", None)
    if isinstance(toolkit_path, str) and not toolkit_path.startswith("cs_copilot.mcp."):
        bindings.append(f"{toolkit_path}.{spec.method}")
    group_toolkit = agno_twins.FACADE_GROUP_TOOLKITS.get(spec.group or "")
    if group_toolkit is not None:
        bindings.append(f"{group_toolkit}.{spec.method}")
    bindings.extend(agno_twins.AGNO_TWINS.get(spec.mcp_name, ()))
    return tuple(dict.fromkeys(bindings))


def _materialized_specs() -> tuple[ToolSpec, ...]:
    specs = tuple(_enrich(spec) for spec in _base_specs())
    validate_registry(specs)
    validate_workflow_permissions(specs)
    return specs


def iter_specs(profile: str | MCPProfile | None = None) -> Iterable[ToolSpec]:
    """Yield tools, optionally restricted to one strict static profile."""

    specs = _materialized_specs()
    if profile is None:
        yield from specs
        return
    selected = get_profile(profile)
    yield from (spec for spec in specs if selected.name in spec.profiles)


def all_specs(profile: str | MCPProfile | None = None) -> List[ToolSpec]:
    """Return registered tools, optionally restricted to ``profile``."""

    return list(iter_specs(profile=profile))


def validate_registry(specs: Iterable[ToolSpec] | None = None) -> None:
    """Validate names, execution metadata, and profile references."""

    materialized = tuple(specs if specs is not None else (_enrich(s) for s in _base_specs()))
    names = [spec.mcp_name for spec in materialized]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate MCP tool names: {', '.join(duplicates)}")
    for spec in materialized:
        if not spec.group:
            raise ValueError(f"{spec.mcp_name}: group is required")
        if capabilities.group_for_tool(spec.mcp_name) != spec.group:
            raise ValueError(
                f"{spec.mcp_name}: tool names must start with their group's prefix "
                f"(group {spec.group!r}); see cs_copilot/capabilities.py"
            )
        if not spec.roles:
            raise ValueError(f"{spec.mcp_name}: at least one role is required")
        if not spec.profiles:
            raise ValueError(f"{spec.mcp_name}: at least one profile is required")
        if spec.read_only and spec.write_scope != "none":
            raise ValueError(f"{spec.mcp_name}: read-only tool cannot require write access")
    validate_profile_registry(materialized)


def required_permissions_for_spec(spec: ToolSpec) -> frozenset[str]:
    """Return workflow permissions implied by one executable tool contract."""

    permissions: set[str] = set()
    if spec.requires_network:
        permissions.add(_NETWORK_PERMISSION)
    if spec.read_artifact_fields:
        permissions.add(_ARTIFACT_READ_PERMISSION)
    if spec.write_scope == "session" or spec.result_artifact_type is not None:
        permissions.add(_ARTIFACT_WRITE_PERMISSION)
    if spec.group in _COMPUTE_GROUPS:
        permissions.add(_COMPUTE_PERMISSION)
    return frozenset(permissions)


def validate_workflow_permissions(
    specs: Iterable[ToolSpec] | None = None,
    *,
    workflows: Sequence[Any] | None = None,
) -> None:
    """Reject workflow permission metadata weaker than its declared capabilities."""

    materialized = tuple(specs if specs is not None else (_enrich(s) for s in _base_specs()))
    by_name = {spec.mcp_name: spec for spec in materialized}
    if workflows is None:
        from cs_copilot.workflows import list_workflows

        workflows = list_workflows()

    for workflow_spec in workflows:
        declared_tool_names = tuple(
            dict.fromkeys(
                str(tool_name)
                for field in ("preflight_tools", "required_tools", "optional_tools")
                for tool_name in getattr(workflow_spec, field, ())
            )
        )
        unknown_tools = sorted(set(declared_tool_names) - set(by_name))
        if unknown_tools:
            raise ValueError(
                f"Workflow {workflow_spec.slug!r} references unknown MCP tools: "
                + ", ".join(unknown_tools)
            )

        required: set[str] = set()
        for tool_name in declared_tool_names:
            required.update(required_permissions_for_spec(by_name[tool_name]))
        if getattr(workflow_spec, "input_artifacts", ()):
            required.add(_ARTIFACT_READ_PERMISSION)
        if getattr(workflow_spec, "output_artifacts", ()):
            required.add(_ARTIFACT_WRITE_PERMISSION)

        declared = {str(permission) for permission in workflow_spec.permissions}
        missing = sorted(required - declared)
        if missing:
            raise ValueError(
                f"Workflow {workflow_spec.slug!r} is missing permissions required by "
                f"its declared tools and artifact contracts: {', '.join(missing)}"
            )
