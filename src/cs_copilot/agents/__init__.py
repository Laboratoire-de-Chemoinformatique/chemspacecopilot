#!/usr/bin/env python
# coding: utf-8
"""Factories and team builders for ChemSpace Copilot agents.

The registry contains nine agent types. Seven are members of the production
team: ChEMBL retrieval, GTM, chemoinformatics, reporting, molecular design,
peptide design, and retrosynthesis. ``robustness_evaluation`` is a separate
evaluation agent, while ``single_agent`` is the flat baseline used by
architecture-ablation tests.

Use :func:`create_agent` to construct one registered type,
:func:`get_cs_copilot_agent_team` for the production team, or
:func:`get_cs_copilot_single_agent` for the controlled baseline.

Exports are resolved lazily so that importing a lightweight submodule such as
:mod:`cs_copilot.agents.instructions` (used verbatim by the MCP prompt
registry) does not construct the Agno team, factory, or registry modules.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .factories import AgentConfig, AgentCreationError, BaseAgentFactory
    from .registry import create_agent, get_registry, list_available_agent_types
    from .single_agent import get_cs_copilot_single_agent
    from .teams import get_cs_copilot_agent_team
    from .utils import get_last_agent_reply

__all__ = [
    # Primary API
    "create_agent",
    "list_available_agent_types",
    "get_registry",
    # Team coordination
    "get_cs_copilot_agent_team",
    # Single-agent baseline (multi-agent-vs-single-agent ablation)
    "get_cs_copilot_single_agent",
    # Utilities
    "get_last_agent_reply",
    # Configuration and exceptions
    "AgentCreationError",
    "AgentConfig",
    "BaseAgentFactory",
]

_LAZY_EXPORTS = {
    "AgentConfig": "factories",
    "AgentCreationError": "factories",
    "BaseAgentFactory": "factories",
    "create_agent": "registry",
    "get_registry": "registry",
    "list_available_agent_types": "registry",
    "get_cs_copilot_single_agent": "single_agent",
    "get_cs_copilot_agent_team": "teams",
    "get_last_agent_reply": "utils",
}
_SUBMODULES = frozenset(
    {
        "config",
        "context",
        "contracts",
        "delegation",
        "descriptions",
        "factories",
        "instructions",
        "registry",
        "single_agent",
        "teams",
        "utils",
    }
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(f"{__name__}.{_LAZY_EXPORTS[name]}"), name)
    elif name in _SUBMODULES:
        value = import_module(f"{__name__}.{name}")
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
