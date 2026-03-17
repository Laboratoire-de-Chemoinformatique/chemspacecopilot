"""Unit tests for MLflowTracker core functionality."""

import pytest

from cs_copilot.tracking.config import MLflowConfig
from cs_copilot.tracking.core import MLflowTracker, get_tracker, reset_tracker


@pytest.fixture
def disabled_config():
    """Create a disabled configuration for testing."""
    return MLflowConfig(enabled=False)


@pytest.fixture
def mock_config(monkeypatch):
    """Create a mock configuration for testing."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/test_mlflow")
    return MLflowConfig.from_env()


def test_tracker_disabled():
    """Test tracker with tracking disabled."""
    config = MLflowConfig(enabled=False)
    tracker = MLflowTracker(config)

    assert not tracker.is_enabled()

    # Context managers should work but do nothing
    with tracker.track_session("test_session") as run:
        assert run is None

    with tracker.track_agent_run("test_agent", "test prompt") as run:
        assert run is None

    with tracker.track_tool_call("test_tool") as run:
        assert run is None


def test_tracker_initialization_without_mlflow(monkeypatch):
    """Test tracker initialization when mlflow is not available."""
    # Mock ImportError for mlflow
    import builtins

    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("No module named 'mlflow'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    config = MLflowConfig(enabled=True)
    tracker = MLflowTracker(config)

    # Should be disabled after failed initialization
    assert not tracker.is_enabled()


def test_get_tracker_singleton():
    """Test that get_tracker returns singleton instance."""
    reset_tracker()

    tracker1 = get_tracker()
    tracker2 = get_tracker()

    assert tracker1 is tracker2


def test_reset_tracker():
    """Test tracker reset functionality."""
    reset_tracker()

    tracker1 = get_tracker()
    reset_tracker()
    tracker2 = get_tracker()

    assert tracker1 is not tracker2


def test_tracker_log_methods_disabled():
    """Test that log methods work safely when tracking is disabled."""
    config = MLflowConfig(enabled=False)
    tracker = MLflowTracker(config)

    # Should not raise errors
    tracker.log_metrics({"test": 1.0})
    tracker.log_params({"param": "value"})
    tracker.log_artifact("/tmp/test.txt")
    tracker.log_text("test", "test.txt")
    tracker.log_dict({"key": "value"}, "test.json")


def test_tracker_safe_logging():
    """Test safe logging methods handle errors gracefully."""
    config = MLflowConfig(enabled=False)
    tracker = MLflowTracker(config)

    # These should not raise errors even with invalid data
    tracker._safe_log_metrics({"metric": "not_a_number"})
    tracker._safe_log_params({"param": None})
    tracker._safe_log_tags({"tag": 123})


@pytest.mark.skipif(
    not pytest.importorskip("mlflow", reason="mlflow not installed"), reason="Requires mlflow"
)
def test_tracker_enabled_basic(tmp_path, monkeypatch):
    """Test basic tracker functionality when enabled."""
    # Use temporary directory for MLflow tracking
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")

    config = MLflowConfig.from_env()
    tracker = MLflowTracker(config)

    assert tracker.is_enabled()


@pytest.mark.skipif(
    not pytest.importorskip("mlflow", reason="mlflow not installed"), reason="Requires mlflow"
)
def test_session_tracking(tmp_path, monkeypatch):
    """Test session tracking creates proper run structure."""

    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")

    reset_tracker()
    tracker = get_tracker()

    with tracker.track_session(
        "test_session_123", user_id="user@example.com", interface="chainlit"
    ) as run:
        assert run is not None
        assert run.info.run_id is not None

        # Verify we can log metrics
        tracker.log_metrics({"test_metric": 1.0})


@pytest.mark.skipif(
    not pytest.importorskip("mlflow", reason="mlflow not installed"), reason="Requires mlflow"
)
def test_nested_tracking(tmp_path, monkeypatch):
    """Test nested run structure (session -> agent -> tool)."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")

    reset_tracker()
    tracker = get_tracker()

    with tracker.track_session("session_123") as session_run:
        assert session_run is not None

        with tracker.track_agent_run("test_agent", "test prompt") as agent_run:
            assert agent_run is not None

            with tracker.track_tool_call("test_tool", {"arg": "value"}) as tool_run:
                assert tool_run is not None

                # Verify run stack is maintained
                assert len(tracker._active_run_stack) == 3

    # Stack should be empty after exiting all contexts
    assert len(tracker._active_run_stack) == 0
