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
) -> bool:
    """Return True when an LLM-backed call should become an external task."""

    if str(engine or "").strip().lower() != "llm":
        return False
    if getattr(agent, "model", None) is not None:
        return False
    policy = str(getattr(agent, "llm_policy", "external") or "external").strip().lower()
    if policy == "external" and getattr(agent, "llm", None) is not None:
        return True
    if policy == "disabled":
        raise MCPToolError(
            f"LLM-backed {domain} design is disabled by MCP llm_policy='disabled'. "
            f"Choose engine='{fallback_engine}' for deterministic generation."
        )
    raise MCPToolError(
        f"LLM-backed {domain} design requires MCP llm_policy='external' with "
        f"the LLM broker, or llm_policy='agno-model'. Choose engine='{fallback_engine}' "
        "for deterministic generation."
    )


def backend_unavailable(name: str, exc: Exception) -> MCPToolError:
    """Return a consistent backend-unavailable protocol error."""

    return MCPToolError(f"{name} backend is unavailable for this tool call: {exc}")
