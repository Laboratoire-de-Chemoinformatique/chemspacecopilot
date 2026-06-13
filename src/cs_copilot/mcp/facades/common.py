"""Shared helpers for MCP facade modules."""

from __future__ import annotations

from typing import Any

from ..errors import MCPToolError


def ensure_llm_engine_available(
    engine: str,
    agent: Any | None,
    *,
    domain: str,
    fallback_engine: str,
) -> None:
    """Reject LLM-backed design when the default MCP shim has no model."""

    if str(engine or "").strip().lower() != "llm":
        return
    if getattr(agent, "model", None) is not None:
        return
    raise MCPToolError(
        f"LLM-backed {domain} design is unavailable in default MCP because "
        "MCPAgentContext.model is None. Use agno_team_run or the Agno team "
        f"runtime for internal-model design, or choose engine='{fallback_engine}'."
    )


def backend_unavailable(name: str, exc: Exception) -> MCPToolError:
    """Return a consistent backend-unavailable protocol error."""

    return MCPToolError(f"{name} backend is unavailable for this tool call: {exc}")
