#!/usr/bin/env python
# coding: utf-8
"""
Cs_copilot Agents Package

This package provides a comprehensive system for creating and managing
AI agents specialized in cheminformatics tasks.

Public API:
-----------

Agent Creation (Recommended):
    create_agent(agent_type, model, **kwargs) - Create agents by type
    list_available_agent_types() - List all available agent types

Team Coordination:
    get_cs_copilot_agent_team(model, **kwargs) - Multi-agent team with intelligent coordination

Utilities:
    get_last_agent_reply(agent) - Extract last message from agent

Exceptions:
    AgentCreationError - Raised when agent creation fails

Available Agent Types (5-Agent Architecture):
----------------------------------------------
Core Agents:
- "chembl_downloader" - Download and process bioactivity data from ChEMBL database
- "gtm_agent" - Unified GTM operations (build, load, density, activity, project) with smart caching
- "chemoinformatician" - Comprehensive chemoinformatics (chemotype, clustering, SAR, similarity, QSAR)
- "report_generator" - Universal presentation layer for all analysis types
- "molecular_designer" - Small-molecule design via autoencoder and LLM engines
- "peptide_designer" - Peptide design via WAE and LLM engines plus latent-space GTM workflows

Testing/Evaluation:
- "robustness_evaluation" - Analyze robustness test results and metrics

Agent Capabilities Breakdown:
-----------------------------
**Chemoinformatician** (Most Versatile):
  - Chemotype/Scaffold Analysis: Extract and analyze molecular frameworks
  - Clustering: Group molecules by structural similarity (k-means, hierarchical, DBSCAN)
  - SAR Analysis: Structure-Activity Relationships, activity cliffs, matched molecular pairs
  - Similarity/Diversity: Molecular similarity, diversity metrics, nearest neighbors
  - QSAR Modeling: Extensible framework for predictive modeling (tools to be added)
"""

from .factories import AgentConfig, AgentCreationError, BaseAgentFactory
from .registry import create_agent, get_registry, list_available_agent_types
from .teams import get_cs_copilot_agent_team
from .utils import get_last_agent_reply

__all__ = [
    # Primary API
    "create_agent",
    "list_available_agent_types",
    "get_registry",
    # Team coordination
    "get_cs_copilot_agent_team",
    # Utilities
    "get_last_agent_reply",
    # Configuration and exceptions
    "AgentCreationError",
    "AgentConfig",
    "BaseAgentFactory",
]
