#!/usr/bin/env python
# coding: utf-8
"""Unit tests for team memory/session isolation configuration."""

from agno.agent import Agent
from agno.models.base import Model

from cs_copilot.agents import teams
from cs_copilot.agents.factories import AgentConfig, BaseAgentFactory


class _ConstructionModel(Model):
    """Minimal Agno model for construction-only tests."""

    def invoke(self, *args, **kwargs):
        raise NotImplementedError

    async def ainvoke(self, *args, **kwargs):
        raise NotImplementedError

    def invoke_stream(self, *args, **kwargs):
        raise NotImplementedError
        yield

    async def ainvoke_stream(self, *args, **kwargs):
        raise NotImplementedError
        yield

    def _parse_provider_response(self, response, **kwargs):
        raise NotImplementedError

    def _parse_provider_response_delta(self, response):
        raise NotImplementedError


def _patch_lightweight_team_dependencies(monkeypatch):
    """Avoid constructing real domain toolkits while testing team wiring."""

    def fake_create_agent(agent_type, model, **kwargs):
        return Agent(
            name=f"{agent_type}_agent",
            model=model,
            session_state=kwargs.get("session_state"),
            telemetry=False,
        )

    monkeypatch.setattr(teams, "create_agent", fake_create_agent)
    monkeypatch.setattr(teams, "analyze_resources", lambda: {"cpu": "test"})


def test_team_keeps_session_history_without_cross_session_memories(monkeypatch, tmp_path):
    """Default team memory should persist thread history without recalling user memories."""
    _patch_lightweight_team_dependencies(monkeypatch)
    model = _ConstructionModel(id="test-model", provider="test")

    team = teams.get_cs_copilot_agent_team(
        model,
        db_file=str(tmp_path / "session-history.db"),
        enable_mlflow_tracking=False,
    )

    assert team.db is not None
    assert team.add_history_to_context is True
    assert team.num_history_runs == 5
    assert team.store_history_messages is True
    assert team.store_tool_messages is True
    assert team.store_media is True
    assert any(tool.__class__.__name__ == "SessionMemoryToolkit" for tool in team.tools)

    assert team.enable_agentic_memory is False
    assert team.enable_user_memories is False
    assert team.add_memories_to_context is False
    assert team.memory_manager is None
    assert all(member.session_state is team.session_state for member in team.members)
    assert team.session_state["resource_profile"] == {"cpu": "test"}
    assert team.session_state["agent_scratch"] == {}


def test_team_memory_disabled_removes_persistence(monkeypatch):
    """Explicitly disabling memory should keep tests and ad hoc runs isolated."""
    _patch_lightweight_team_dependencies(monkeypatch)
    model = _ConstructionModel(id="test-model", provider="test")

    team = teams.get_cs_copilot_agent_team(
        model,
        enable_memory=False,
        enable_mlflow_tracking=False,
    )

    assert team.db is None
    assert team.add_history_to_context is False
    assert team.num_history_runs == 0
    assert team.store_history_messages is False
    assert team.store_tool_messages is False
    assert team.store_media is False

    assert team.enable_agentic_memory is False
    assert team.enable_user_memories is False
    assert team.add_memories_to_context is False


def test_agent_factory_merges_defaults_into_shared_session_state():
    """Member factories should preserve shared state while adding their default keys."""

    class _Factory(BaseAgentFactory):
        agent_type = "test"

        def get_agent_config(self):
            return AgentConfig(
                name="test_agent",
                description="test agent",
                tools=[],
                session_state={
                    "data_file_paths": {
                        "dataset_path": None,
                        "clean_dataset_path": None,
                    },
                    "agent_only": {"enabled": True},
                },
            )

    shared_state = {
        "resource_profile": {"cpu": "test"},
        "data_file_paths": {"clean_dataset_path": "clean.csv"},
    }
    model = _ConstructionModel(id="test-model", provider="test")

    agent = _Factory().create_agent(
        model,
        session_state=shared_state,
        enable_mlflow_tracking=False,
    )

    assert agent.session_state is shared_state
    assert shared_state["resource_profile"] == {"cpu": "test"}
    assert shared_state["data_file_paths"]["clean_dataset_path"] == "clean.csv"
    assert shared_state["data_file_paths"]["dataset_path"] is None
    assert shared_state["agent_only"] == {"enabled": True}
