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
"""

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
