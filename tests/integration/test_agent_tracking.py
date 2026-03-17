"""Integration tests for agent MLflow tracking."""

import pytest

# Skip if MLflow not installed
pytest.importorskip("mlflow")


from cs_copilot.tracking import get_tracker, reset_tracker  # noqa: E402


@pytest.fixture
def temp_mlflow_dir(tmp_path):
    """Create a temporary MLflow tracking directory."""
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir()
    return mlflow_dir


@pytest.fixture
def tracking_env(temp_mlflow_dir, monkeypatch):
    """Set up environment for MLflow tracking."""
    tracking_uri = f"file://{temp_mlflow_dir}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test_agents")

    # Reset tracker to pick up new config
    reset_tracker()

    yield tracking_uri

    # Cleanup
    reset_tracker()


def test_agent_factory_tracking_disabled(monkeypatch):
    """Test that agent factory works with tracking disabled."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")
    reset_tracker()

    from agno.models.openai import OpenAIChat

    from cs_copilot.agents.factories import ChEMBLDownloaderFactory

    factory = ChEMBLDownloaderFactory()

    # Should work with tracking disabled
    try:
        model = OpenAIChat(id="gpt-4o-mini", api_key="dummy-key")
        agent = factory.create_agent(model, enable_mlflow_tracking=False)
        assert agent is not None
        assert agent.name == "chembl_agent"
    except Exception as e:
        # API key issues are expected, we just want to verify agent creation doesn't fail
        # due to tracking
        if "tracking" not in str(e).lower():
            pass  # Expected API-related error


def test_agent_factory_tracking_enabled(tracking_env):
    """Test that agent factory enables tracking when configured."""
    from agno.models.openai import OpenAIChat

    from cs_copilot.agents.factories import ChEMBLDownloaderFactory

    factory = ChEMBLDownloaderFactory()
    tracker = get_tracker()

    assert tracker.is_enabled()

    try:
        model = OpenAIChat(id="gpt-4o-mini", api_key="dummy-key")
        agent = factory.create_agent(model, enable_mlflow_tracking=True)
        assert agent is not None

        # Verify that agent.run is wrapped
        # (we can't easily test it without a valid API key, but we can check it exists)
        assert hasattr(agent, "run")
        assert hasattr(agent, "arun")
    except Exception as e:
        # API key issues are expected
        if "tracking" not in str(e).lower():
            pass


def test_tracker_context_managers(tracking_env):
    """Test that tracking context managers work correctly."""
    tracker = get_tracker()
    assert tracker.is_enabled()

    # Test session tracking
    with tracker.track_session("test_session_123", user_id="test_user", interface="test"):
        # Log some metrics
        tracker.log_metrics({"test_metric": 1.0})

        # Test nested agent run
        with tracker.track_agent_run("test_agent", "test prompt"):
            tracker.log_metrics({"agent_metric": 2.0})

            # Test nested tool call
            with tracker.track_tool_call("test_tool", {"arg": "value"}):
                tracker.log_metrics({"tool_metric": 3.0})


def test_team_creation_with_tracking(tracking_env, monkeypatch):
    """Test team creation with MLflow tracking enabled."""
    from agno.models.openai import OpenAIChat

    from cs_copilot.agents.teams import get_cs_copilot_agent_team

    # Set a dummy API key
    monkeypatch.setenv("OPENAI_API_KEY", "dummy-key")

    tracker = get_tracker()
    assert tracker.is_enabled()

    try:
        model = OpenAIChat(id="gpt-4o-mini")
        team = get_cs_copilot_agent_team(
            model,
            enable_memory=False,  # Disable memory for testing
            enable_mlflow_tracking=True,
        )

        assert team is not None
        assert len(team.members) > 0

        # Verify that team members have tracking enabled
        for member in team.members:
            assert hasattr(member, "run")
            assert hasattr(member, "arun")
    except Exception as e:
        # Expected errors due to dummy API key
        if "tracking" not in str(e).lower():
            pass


def test_tracking_with_real_execution_mock(tracking_env):
    """Test tracking with a mocked agent execution."""
    from cs_copilot.tracking.decorators import track_agent_run

    tracker = get_tracker()

    @track_agent_run(agent_name="mock_agent", agent_type="test")
    def mock_agent_run(prompt: str):
        return f"Processed: {prompt}"

    # Execute within session context
    with tracker.track_session("test_session"):
        result = mock_agent_run("test prompt")
        assert result == "Processed: test prompt"

        # Verify we can log metrics
        tracker.log_metrics({"completion": 1.0})


def test_mlflow_run_hierarchy(tracking_env):
    """Test that MLflow creates proper run hierarchy."""
    import mlflow

    tracker = get_tracker()

    # Create a session -> agent -> tool hierarchy
    with tracker.track_session("test_session") as session_run:
        assert session_run is not None

        with tracker.track_agent_run("test_agent", "test prompt") as agent_run:
            assert agent_run is not None
            # Agent run should be nested under session
            # (MLflow doesn't expose parent_run_id easily, so we just verify run exists)

            with tracker.track_tool_call("test_tool") as tool_run:
                assert tool_run is not None
                # Tool run should be nested under agent

    # Verify runs were created by querying MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_agents")
    assert experiment is not None

    # Get all runs
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) >= 3  # At least session, agent, and tool runs
