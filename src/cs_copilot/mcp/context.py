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


def set_current_context(ctx: MCPAgentContext) -> None:
    """Bind ``ctx`` as the active MCP agent context for the current task."""

    _CTX.set(ctx)


def get_current_context() -> Optional[MCPAgentContext]:
    """Return the active MCP agent context, if one has been bound."""

    return _CTX.get()
