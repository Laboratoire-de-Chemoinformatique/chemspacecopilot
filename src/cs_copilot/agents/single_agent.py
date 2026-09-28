#!/usr/bin/env python
# coding: utf-8
"""Single-agent baseline builder for the multi-agent ablation.

Mirrors :func:`cs_copilot.agents.teams.get_cs_copilot_agent_team` but returns ONE
flat Agno ``Agent`` holding the union of all specialist toolkits. Use the same
``Model`` instance for both this builder and ``get_cs_copilot_agent_team`` so the
only variable in the comparison is the agentic structure (single vs multi), not
the base model, tools, or harness.
"""

import logging

from agno.agent import Agent
from agno.models.base import Model  # Agno v2 base class

from cs_copilot.utils.resources import analyze_resources

from .registry import create_agent


def get_cs_copilot_single_agent(
    model: Model,
    *,
    markdown: bool = True,
    debug_mode: bool = False,
    enable_mlflow_tracking: bool = True,
) -> Agent:
    """Create the single flat agent (all toolkits, no team, no routing).

    Args:
        model: Agno Model instance — pass the SAME instance used for the team so
            the ablation holds the model constant.
        markdown: Format output in markdown (matches team members).
        debug_mode: Enable debug logs.
        enable_mlflow_tracking: Wrap ``run``/``arun`` with MLflow tracking, exactly
            as team members are wrapped, for a fair comparison.

    Returns:
        Agent: The configured single-agent baseline. Exposes ``.run`` /
        ``.get_session_state`` like the team, so the robustness harness can drive
        either arm through one interface.
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating Cs_copilot single-agent baseline")

    # Seed the same shared-state contract the team members get (teams.py:70-84):
    # resource profile + a scratch dict. The factory's union session_state is then
    # deep-merged in by BaseAgentFactory.create_agent without overwriting these.
    seed_session_state = {
        "resource_profile": analyze_resources(),
        "agent_scratch": {},
    }

    agent = create_agent(
        "single_agent",
        model=model,
        markdown=markdown,
        debug_mode=debug_mode,
        enable_mlflow_tracking=enable_mlflow_tracking,
        session_state=seed_session_state,
    )
    logger.info("Successfully created Cs_copilot single-agent baseline")
    return agent
