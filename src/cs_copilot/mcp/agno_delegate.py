"""Optional private MCP delegation into the Agno team runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from .context import MCPAgentContext


def build_agno_team_tool(ctx: MCPAgentContext):
    """Return the opt-in ``agno_team_run`` MCP tool.

    Imports are intentionally dynamic so the default MCP path never imports
    the Agno team, factories, registry, or configured model backend.
    """

    async def agno_team_run(prompt: str) -> dict[str, Any]:
        """Run the private ChemSpace Agno team for a trusted prompt."""

        return await asyncio.to_thread(_run_team, prompt, ctx)

    agno_team_run.__name__ = "agno_team_run"
    agno_team_run.__qualname__ = "agno_team_run"
    return agno_team_run


def _run_team(prompt: str, ctx: MCPAgentContext) -> dict[str, Any]:
    model_config = __import__(
        "cs_copilot.model_config",
        fromlist=["load_model_from_config"],
    )
    teams = __import__(
        "cs_copilot.agents.teams",
        fromlist=["get_cs_copilot_agent_team"],
    )

    model = model_config.load_model_from_config()
    team = teams.get_cs_copilot_agent_team(model, show_members_responses=False)
    result = team.run(prompt, stream=False)

    team_state = getattr(team, "session_state", None)
    if isinstance(team_state, dict):
        ctx.session_state.update(team_state)

    return {
        "status": "ok",
        "runtime": "agno_team",
        "content": str(result),
        "session_state_keys": sorted(str(key) for key in ctx.session_state),
    }
