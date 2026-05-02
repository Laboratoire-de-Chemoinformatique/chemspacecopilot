#!/usr/bin/env python
# coding: utf-8
"""
Team coordination functionality for multi-agent workflows.
"""

import logging
from typing import List, Tuple

from agno.db.sqlite import SqliteDb  # ✅ v2.1.x style DB import
from agno.models.base import Model  # Agno v2 base class
from agno.team import Team

from cs_copilot.utils.resources import analyze_resources

from .config import CS_COPILOT_MEMORY_DB  # optional now; kept for compatibility
from .factories import AgentCreationError
from .prompts import AGENT_TEAM_INSTRUCTIONS
from .registry import create_agent


def get_cs_copilot_agent_team(
    model: Model,  # Agno Model instance, e.g. OpenAIChat(...) or Claude(...)
    *,
    markdown: bool = True,
    debug_mode: bool = False,
    show_members_responses: bool = True,
    enable_memory: bool = True,
    db_file: str = None,
    enable_mlflow_tracking: bool = True,
) -> Team:
    """
    Create a coordinated team of cs_copilot agents using Agno.

    Args:
        model: Agno Model instance used for team coordination and member agents
        markdown: Format output in markdown
        debug_mode: Enable debug logs
        show_members_responses: Print member responses during coordination
        enable_memory: Enable persistent session history (default: True). Cross-session
                      user/agentic memories stay disabled to prevent state leakage.
        db_file: Custom database file path. If not provided, uses CS_COPILOT_MEMORY_DB.
                Use unique paths for session isolation in testing.
        enable_mlflow_tracking: Enable MLflow tracking for agents (default: True).
                               Set to False to disable tracking.

    Returns:
        Team: Configured Cs_copilot team

    Raises:
        AgentCreationError: If one or more agents fail to initialize
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Cs_copilot Agent Team")

    # ✅ Single DB handles session storage/history in v2.1.x.
    # Cross-session memories are intentionally disabled below; only per-thread
    # history/session state should persist.
    db = None
    if enable_memory:
        db = SqliteDb(
            db_file=db_file
            or CS_COPILOT_MEMORY_DB
            # NOTE: CS_COPILOT_MEMORY_TABLE is not required by SqliteDb.
            # Agno manages its own tables for sessions/memories. Kept import for compat.
        )

    # Probe runtime environment (GPU, CPU, RAM, databases, cached models)
    resource_profile = analyze_resources()
    logger.info("Resource profile: %s", resource_profile)

    # Common agent parameters supplied by the factory
    agent_params = {
        "markdown": markdown,
        "debug_mode": debug_mode,
        "enable_mlflow_tracking": enable_mlflow_tracking,
    }

    # ============================================================================
    # 5-AGENT ARCHITECTURE
    # ============================================================================
    # Consolidation history:
    #   MERGED: GTM Optimization + Loading + Density + Activity → GTM Agent
    #   GENERALIZED: GTM Chemotype Analysis → Chemoinformatician (method-agnostic)
    #   MERGED: Autoencoder + Autoencoder GTM Sampling → Autoencoder (mode-based)
    #   ADDED: Report Generator (presentation layer)
    #   REMOVED: Robustness Evaluator (not included in main team, invoked separately)
    # ============================================================================

    # (type_key, human_name)
    agents_config: List[Tuple[str, str]] = [
        ("chembl_downloader", "ChEMBL Downloader"),
        (
            "gtm_agent",
            "GTM Agent",
        ),  # Unified GTM operations (build, load, density, activity, project)
        (
            "chemoinformatician",
            "Chemoinformatician",
        ),  # Comprehensive chemoinformatics (chemotype, clustering, SAR, similarity, QSAR)
        ("report_generator", "Report Generator"),  # Universal presentation layer
        ("autoencoder", "Autoencoder"),  # SMILES molecule generation (LSTM autoencoder)
        ("peptide_wae", "Peptide WAE"),  # Peptide sequence generation (Wasserstein autoencoder)
        ("synplanner", "SynPlanner"),
        # Note: Robustness Evaluator excluded from main team (invoked separately for testing)
    ]

    agents = []
    failures = []

    for agent_type, agent_name in agents_config:
        try:
            logger.info("Creating %s agent", agent_name)
            agent = create_agent(agent_type, model=model, **agent_params)
            agents.append(agent)
            logger.info("Successfully created %s agent", agent_name)
        except Exception as e:
            logger.exception("Failed to create %s agent", agent_name)
            failures.append(f"{agent_name}: {e!s}")

    if failures:
        msg = "Agent initialization failures:\n  - " + "\n  - ".join(failures)
        raise AgentCreationError(msg)

    team = Team(
        name="Cs_copilot Team",
        members=agents,
        model=model,
        # ✅ Attach DB directly to the team (persists sessions/history)
        # If enable_memory=False, db=None prevents any persistence
        db=db,
        # Keep session history, but never inject cross-session memories. Agno
        # defaults add_memories_to_context=True when agentic memory is enabled,
        # which caused new chats to recall prior chemical-space analyses.
        enable_agentic_memory=False,
        enable_user_memories=False,
        add_memories_to_context=False,
        add_history_to_context=enable_memory,  # include recent history in prompts
        num_history_runs=5 if enable_memory else 0,  # 🔧 LIMIT context to last 5 runs
        share_member_interactions=True,  # share member messages across team
        store_history_messages=enable_memory,  # persist message history to DB
        store_tool_messages=enable_memory,  # persist tool results
        store_media=enable_memory,  # persist any media if used
        # Session state (always enabled for within-session data passing)
        session_state={"resource_profile": resource_profile},
        add_session_state_to_context=True,
        enable_agentic_state=True,
        # Prompting
        description=(
            "You are an intelligent coordinator orchestrating a team of specialized cheminformatics agents. "
            "Your role is to understand user requests, select the appropriate agent(s) or workflows, "
            "and chain multiple agents when needed to complete complex analyses.\n\n"
            "• ChEMBL Downloader: Download bioactivity data from ChEMBL database\n"
            "• GTM Agent: All GTM operations (build/load/density/activity/project) with smart caching\n"
            "• Chemoinformatician: Downstream analysis (scaffold, SAR, similarity, clustering) - works with GTM output\n"
            "• Report Generator: Universal presentation layer for all analysis types\n"
            "• Autoencoder: Small molecule generation via LSTM autoencoders (SMILES, standalone + GTM-guided)\n"
            "• Peptide WAE: Peptide sequence generation + GTM on latent space + DBAASP antimicrobial activity landscapes\n"
            "• SynPlanner: Retrosynthetic planning for target molecules\n\n"
            "**Molecule vs Peptide Routing**:\n"
            "  - 'peptide', 'amino acid', 'AMP', 'antimicrobial peptide' → Peptide WAE agent\n"
            "  - 'SMILES', 'molecule', 'compound', 'small molecule' → Autoencoder agent\n"
            "  - DBAASP/antimicrobial landscapes → Peptide WAE agent (has GTM tools)\n"
            "  - Unqualified 'generate' → Autoencoder (small molecules)\n\n"
            "When coordinating: (1) Assess if a predefined workflow covers the request, (2) Select and chain "
            "specialized agents for multi-step tasks (GTM → Chemoinformatician → Report Generator is common), "
            "(3) For analysis requests, automatically add Report Generator unless user explicitly requests raw data only, "
            "(4) For ambiguous opening requests, apply the INITIAL CLARIFICATION FLOW (peptides vs molecules, then exploratory vs generative), (5) Synthesize insights from agent outputs into coherent analyses."
        ),
        instructions=AGENT_TEAM_INSTRUCTIONS,
        # UX & observability
        markdown=markdown,
        debug_mode=debug_mode,
        stream_member_events=True,  # stream events from members (Team API)
        show_members_responses=show_members_responses,
    )

    logger.info("Successfully created Cs_copilot Agent Team")
    return team
