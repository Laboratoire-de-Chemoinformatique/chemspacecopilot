"""Lightweight Agno-agent shim used by the MCP server.

ChemSpace toolkit methods that accept ``agent: Optional[Agent]`` read and
write ``agent.session_state`` to share data across calls. In MCP mode there
is no real Agno agent — the MCP client (Codex / Claude Code) is the reasoning
engine — so we provide a tiny stand-in that exposes the small subset of the
``Agent`` interface those methods actually touch.

The shim deliberately keeps ``model = None`` so any toolkit code path that
attempts an LLM-as-judge call (which checks ``getattr(agent, "model", None)``)
bails out cleanly instead of reaching the configured Agno provider.
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
        Always ``None`` in MCP mode — see module docstring.
    """

    name: str = "mcp-client"
    session_state: Dict[str, Any] = field(default_factory=dict)
    model: Any = None


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
