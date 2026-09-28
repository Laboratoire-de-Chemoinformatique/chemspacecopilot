"""Lightweight Agno-agent shim used by the MCP server.

cs_copilot toolkit methods that accept ``agent: Optional[Agent]`` read and
write ``agent.session_state`` to share data across calls. In MCP mode there
is no real Agno agent — the MCP client (Codex / Claude Code) is the reasoning
engine — so we provide a tiny stand-in that exposes the small subset of the
``Agent`` interface those methods actually touch.

By default the shim keeps ``model = None`` so toolkit code cannot accidentally
reach the configured Agno provider. MCP bootstrap may opt into ``llm_policy =
"agno-model"`` to attach only the configured model, without constructing or
running the Agno team.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class MCPAgentContext:
    """Minimal agent shim shared across all MCP tool invocations.

    Attributes
    ----------
    name:
        Identifier exposed to toolkit code paths that record provenance
        (``getattr(agent, "name", None)``).
    session_state:
        Mutable dict that toolkits use as shared workspace; matches the role
        of ``Team.session_state`` (`src/cs_copilot/agents/teams.py:131-153`).
    model:
        ``None`` unless MCP bootstrap is explicitly configured with
        ``llm_policy="agno-model"``.
    llm_policy:
        MCP-wide LLM policy: ``external`` (default), ``agno-model``, or
        ``disabled``.
    llm:
        General broker used by MCP tools to create and complete LLM tasks.
    """

    name: str = "mcp-client"
    session_state: Dict[str, Any] = field(default_factory=dict)
    model: Any = None
    llm_policy: str = "external"
    llm: Any = None


_CTX: contextvars.ContextVar[Optional[MCPAgentContext]] = contextvars.ContextVar(
    "cs_copilot_mcp_agent_context",
    default=None,
)
_ACTIVE_TASK_KEYS = (
    "active_task_id",
    "active_role",
    "active_profile",
    "active_task_attempt",
    "active_handoff_id",
)


def bind_active_task_scope(
    session_state: Dict[str, Any],
    task: Any,
    *,
    run: Any | None = None,
) -> None:
    """Bind one authoritative running task as the MCP execution scope."""

    session_state["active_task_id"] = str(task.task_id)
    session_state["active_role"] = str(task.role)
    session_state["active_profile"] = str(task.profile)
    session_state["active_task_attempt"] = int(task.attempts)
    handoff = next(
        (
            item
            for item in reversed(getattr(run, "handoffs", ()) or ())
            if item.task_id == task.task_id and item.task_attempt == max(0, int(task.attempts) - 1)
        ),
        None,
    )
    if handoff is None:
        session_state.pop("active_handoff_id", None)
    else:
        session_state["active_handoff_id"] = str(handoff.handoff_id)


def clear_active_task_scope(
    session_state: Dict[str, Any],
    *,
    task_id: str | None = None,
) -> bool:
    """Clear the execution scope, optionally only when it names ``task_id``."""

    if task_id is not None and session_state.get("active_task_id") != task_id:
        return False
    for key in _ACTIVE_TASK_KEYS:
        session_state.pop(key, None)
    return True


def restore_active_task_scope(session_state: Dict[str, Any], run: Any) -> Any | None:
    """Restore a valid scope, auto-selecting only an unambiguous running task."""

    tasks = getattr(run, "tasks", {}) or {}
    current_id = session_state.get("active_task_id")
    current = tasks.get(current_id) if current_id is not None else None
    if current is not None and _task_is_running(current):
        bind_active_task_scope(session_state, current, run=run)
        return current

    running = [task for task in tasks.values() if _task_is_running(task)]
    clear_active_task_scope(session_state)
    if len(running) == 1:
        bind_active_task_scope(session_state, running[0], run=run)
        return running[0]
    return None


def _task_is_running(task: Any) -> bool:
    status = getattr(task, "status", None)
    return getattr(status, "value", status) == "running"


def set_current_context(ctx: MCPAgentContext) -> None:
    """Bind ``ctx`` as the active MCP agent context for the current task."""

    _CTX.set(ctx)


def get_current_context() -> Optional[MCPAgentContext]:
    """Return the active MCP agent context, if one has been bound."""

    return _CTX.get()
