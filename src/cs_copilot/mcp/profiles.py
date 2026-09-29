"""Static MCP capability profiles and workflow compatibility checks.

Profiles are declared in the shared capability table
(:mod:`cs_copilot.capabilities`), together with the role grants that both
runtimes derive from. They are an execution boundary, not a second scientific
workflow catalog: a profile only decides which deterministic tools a client
may discover and invoke. Workflow procedures and their required tools remain
owned by ``workflow_catalog``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

from cs_copilot.capabilities import (
    ALL_GROUPS,
    CORE_GROUPS,
    DISCOVERY_TOOLS,
    PROFILES,
    CapabilityProfile,
    profiles_for_tool,
)

if TYPE_CHECKING:
    from .tool_adapter import ToolSpec


class MCPProfileError(ValueError):
    """Raised when a profile or workflow/profile selection is invalid."""


# Profiles are declared once in the shared capability table; MCPProfile keeps
# its historical name for callers and isinstance checks.
MCPProfile = CapabilityProfile
_DISCOVERY_TOOLS = DISCOVERY_TOOLS
_CORE_GROUPS = CORE_GROUPS
_ALL_GROUPS = ALL_GROUPS


def profile_names() -> tuple[str, ...]:
    """Return profile names in their stable CLI display order."""

    return tuple(PROFILES)


def get_profile(profile: str | MCPProfile) -> MCPProfile:
    """Resolve a profile name and reject unknown values."""

    if isinstance(profile, MCPProfile):
        return profile
    normalized = str(profile or "").strip().lower()
    try:
        return PROFILES[normalized]
    except KeyError as exc:
        available = ", ".join(profile_names())
        raise MCPProfileError(
            f"Unknown MCP profile {profile!r}. Available profiles: {available}"
        ) from exc


def profiles_for_spec(spec: "ToolSpec") -> tuple[str, ...]:
    """Return every static profile that exposes ``spec``."""

    return profiles_for_tool(spec.mcp_name, spec.group)


def validate_profile_registry(specs: Iterable["ToolSpec"]) -> None:
    """Validate profile references, group names, and non-empty allowlists."""

    materialized = tuple(specs)
    known_names = {spec.mcp_name for spec in materialized}
    known_groups = {spec.group for spec in materialized if spec.group}
    known_profiles = set(PROFILES)

    for profile in PROFILES.values():
        unknown_groups = sorted(profile.groups - known_groups)
        unknown_tools = sorted(profile.tools - known_names)
        if unknown_groups or unknown_tools:
            details = []
            if unknown_groups:
                details.append(f"unknown groups: {', '.join(unknown_groups)}")
            if unknown_tools:
                details.append(f"unknown tools: {', '.join(unknown_tools)}")
            raise MCPProfileError(f"Invalid MCP profile {profile.name!r}: {'; '.join(details)}")
        if not any(profile.allows(spec) for spec in materialized):
            raise MCPProfileError(f"MCP profile {profile.name!r} exposes no tools")

    for spec in materialized:
        unknown = sorted(set(spec.profiles) - known_profiles)
        if unknown:
            raise MCPProfileError(
                f"Tool {spec.mcp_name!r} references unknown profiles: {', '.join(unknown)}"
            )


def validate_workflow_profile(
    profile: str | MCPProfile,
    workflow_slug: str,
    *,
    specs: Iterable["ToolSpec"] | None = None,
) -> None:
    """Fail when ``profile`` cannot execute a selected workflow contract.

    Compatibility is deliberately strict for preflight and required tools.
    Optional tools may be absent; the workflow must already describe how to
    continue without them.
    """

    selected = get_profile(profile)
    try:
        from cs_copilot.workflows import get_workflow

        workflow = get_workflow(workflow_slug)
    except (KeyError, FileNotFoundError, ValueError) as exc:
        raise MCPProfileError(str(exc)) from exc

    declared_profiles = tuple(getattr(workflow, "profiles", ()) or ())
    # ``standard`` is the intentional all-capabilities superset. Catalog
    # metadata names the least-privilege profile(s) recommended for a
    # workflow; it must not make the standard profile less capable.
    if declared_profiles and selected.name != "standard" and selected.name not in declared_profiles:
        raise MCPProfileError(
            f"Workflow {workflow.slug!r} does not permit MCP profile {selected.name!r}; "
            f"declared profiles: {', '.join(declared_profiles)}"
        )

    if specs is None:
        from .tools_registry import iter_specs

        specs = iter_specs(profile=selected.name)
    available = {spec.mcp_name for spec in specs}
    required = set(workflow.preflight_tools) | set(workflow.required_tools)
    missing = sorted(required - available)
    if missing:
        raise MCPProfileError(
            f"MCP profile {selected.name!r} is incompatible with workflow "
            f"{workflow.slug!r}; missing required tools: {', '.join(missing)}"
        )


def validate_pinned_workflow_profile(
    profile: str | MCPProfile,
    workflow_contract: Mapping[str, object],
    *,
    specs: Iterable["ToolSpec"] | None = None,
) -> None:
    """Validate a profile against an immutable workflow contract snapshot."""

    selected = get_profile(profile)
    if not workflow_contract:
        raise MCPProfileError(
            "The active catalog run has no pinned workflow contract and cannot be resumed."
        )
    slug = str(workflow_contract.get("slug") or "")
    declared_profiles = tuple(str(item) for item in workflow_contract.get("profiles", ()) or ())
    if declared_profiles and selected.name != "standard" and selected.name not in declared_profiles:
        raise MCPProfileError(
            f"Workflow {slug!r} does not permit MCP profile {selected.name!r}; "
            f"declared profiles: {', '.join(declared_profiles)}"
        )

    required = {
        str(tool_name)
        for field in ("preflight_tools", "required_tools")
        for tool_name in workflow_contract.get(field, ()) or ()
    }
    for task in workflow_contract.get("tasks", ()) or ():
        if isinstance(task, Mapping):
            required.update(str(item) for item in task.get("required_tools", ()) or ())
    if specs is None:
        from .tools_registry import iter_specs

        specs = iter_specs(profile=selected.name)
    available = {spec.mcp_name for spec in specs}
    missing = sorted(required - available)
    if missing:
        raise MCPProfileError(
            f"MCP profile {selected.name!r} is incompatible with pinned workflow "
            f"{slug!r}; missing required tools: {', '.join(missing)}"
        )
