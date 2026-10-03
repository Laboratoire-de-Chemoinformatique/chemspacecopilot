"""Execution contexts and runtime profiles shared by the tool kernel.

The kernel never needs a concrete agent class. It works with any object that
exposes a mutable ``session_state`` dict and, optionally, a settable
``run_context`` (the active :class:`cs_copilot.workflows.RunContext`). The MCP
server passes its ``MCPAgentContext`` shim; the in-process runtime passes a
:class:`BasicExecutionContext`. A :class:`RuntimeProfile` labels the durable
tool events a context produces and names the ad-hoc workflow slug used when
no catalog workflow was selected.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping

from .errors import ToolErrorCode, ToolExecutionError


@dataclass(frozen=True)
class RuntimeProfile:
    """Identity of one runtime that drives tools through the kernel."""

    label: str
    ad_hoc_workflow_slug: str


MCP_RUNTIME = RuntimeProfile(label="mcp", ad_hoc_workflow_slug="mcp-session")
AGNO_RUNTIME = RuntimeProfile(label="agno", ad_hoc_workflow_slug="agno-session")
AD_HOC_WORKFLOW_SLUGS = frozenset(
    {MCP_RUNTIME.ad_hoc_workflow_slug, AGNO_RUNTIME.ad_hoc_workflow_slug}
)


@typing.runtime_checkable
class ExecutionContext(typing.Protocol):
    """Minimal state the kernel reads and writes during one invocation."""

    session_state: dict[str, Any]


@dataclass
class BasicExecutionContext:
    """Plain execution context for runtimes without an agent shim of their own."""

    execution_runtime: RuntimeProfile
    session_state: dict[str, Any] = field(default_factory=dict)
    run_context: Any = None


def resolve_runtime(ctx: Any) -> RuntimeProfile:
    """Return the runtime profile of ``ctx`` (MCP for contexts that declare none)."""

    runtime = getattr(ctx, "execution_runtime", None)
    return runtime if isinstance(runtime, RuntimeProfile) else MCP_RUNTIME


def is_ad_hoc_run(run: Any) -> bool:
    """Return whether ``run`` is an ad-hoc session run without a catalog contract."""

    return str(getattr(run, "workflow_slug", "")) in AD_HOC_WORKFLOW_SLUGS


def _active_output_layout(ctx: ExecutionContext):
    from cs_copilot.storage.layout import OutputLayout

    run_context = getattr(ctx, "run_context", None)
    layout = getattr(run_context, "layout", None)
    if isinstance(layout, OutputLayout):
        _validate_storage_session(layout.session_id)
        return layout

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, Mapping):
        return None
    required = ("session_id", "run_id", "workflow_slug")
    if not all(output_context.get(field) for field in required):
        return None
    try:
        layout = OutputLayout(
            session_id=str(output_context["session_id"]),
            run_id=str(output_context["run_id"]),
            workflow_slug=str(output_context["workflow_slug"]),
        )
    except ValueError as exc:
        raise ToolExecutionError(
            f"active output context is invalid: {exc}",
            code=ToolErrorCode.PERMISSION_DENIED,
        ) from exc
    _validate_storage_session(layout.session_id)
    return layout


def _validate_storage_session(session_id: str) -> None:
    from cs_copilot.storage import S3

    active_session = PurePosixPath(S3.current_prefix().strip("/")).name
    if active_session != session_id:
        raise ToolExecutionError(
            "active storage session does not match the workflow run",
            code=ToolErrorCode.PERMISSION_DENIED,
        )


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
