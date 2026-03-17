"""Unit tests for tracking decorators."""

import pytest

from cs_copilot.tracking.core import get_tracker, reset_tracker
from cs_copilot.tracking.decorators import track_agent_run, track_tool_call


@pytest.fixture
def disabled_tracking(monkeypatch):
    """Disable tracking for tests."""
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "false")
    reset_tracker()


def test_track_agent_run_decorator_sync(disabled_tracking):
    """Test agent run decorator on synchronous function."""

    @track_agent_run(agent_name="test_agent")
    def my_agent(prompt: str):
        return f"Processed: {prompt}"

    result = my_agent("test prompt")
    assert result == "Processed: test prompt"


def test_track_agent_run_decorator_async(disabled_tracking):
    """Test agent run decorator on async function."""
    import asyncio

    @track_agent_run(agent_name="test_agent")
    async def my_async_agent(prompt: str):
        await asyncio.sleep(0.01)
        return f"Processed: {prompt}"

    result = asyncio.run(my_async_agent("test prompt"))
    assert result == "Processed: test prompt"


def test_track_tool_call_decorator_sync(disabled_tracking):
    """Test tool call decorator on synchronous function."""

    @track_tool_call(tool_name="test_tool")
    def my_tool(arg1: str, arg2: int):
        return f"{arg1}-{arg2}"

    result = my_tool("test", 42)
    assert result == "test-42"


def test_track_tool_call_decorator_async(disabled_tracking):
    """Test tool call decorator on async function."""
    import asyncio

    @track_tool_call(tool_name="test_tool")
    async def my_async_tool(arg1: str, arg2: int):
        await asyncio.sleep(0.01)
        return f"{arg1}-{arg2}"

    result = asyncio.run(my_async_tool("test", 42))
    assert result == "test-42"


def test_track_tool_call_decorator_error_handling(disabled_tracking):
    """Test tool call decorator handles errors properly."""

    @track_tool_call(tool_name="failing_tool")
    def failing_tool():
        raise ValueError("Test error")

    with pytest.raises(ValueError, match="Test error"):
        failing_tool()


def test_decorator_without_name(disabled_tracking):
    """Test decorators without explicit name use function name."""

    @track_agent_run()
    def my_custom_agent(prompt: str):
        return prompt

    result = my_custom_agent("test")
    assert result == "test"

    @track_tool_call()
    def my_custom_tool(arg: str):
        return arg

    result = my_custom_tool("test")
    assert result == "test"


def test_decorator_with_class_method(disabled_tracking):
    """Test decorators work with class methods."""

    class MyAgent:
        @track_agent_run(agent_name="class_agent")
        def run(self, prompt: str):
            return f"Agent: {prompt}"

    agent = MyAgent()
    result = agent.run("test")
    assert result == "Agent: test"


def test_decorator_with_complex_result(disabled_tracking):
    """Test decorators handle complex return types."""

    @track_tool_call(tool_name="complex_tool")
    def complex_tool():
        return {
            "data": [1, 2, 3],
            "metadata": {"count": 3, "status": "success"},
        }

    result = complex_tool()
    assert result["data"] == [1, 2, 3]
    assert result["metadata"]["count"] == 3


@pytest.mark.skipif(
    not pytest.importorskip("mlflow", reason="mlflow not installed"), reason="Requires mlflow"
)
def test_decorator_logs_to_mlflow(tmp_path, monkeypatch):
    """Test that decorators actually log to MLflow when enabled."""
    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")

    reset_tracker()
    tracker = get_tracker()

    @track_agent_run(agent_name="test_agent")
    def test_agent(prompt: str):
        return f"Result: {prompt}"

    # Need to be in a session context for nested runs
    with tracker.track_session("test_session"):
        result = test_agent("test prompt")
        assert result == "Result: test prompt"


@pytest.mark.skipif(
    not pytest.importorskip("mlflow", reason="mlflow not installed"), reason="Requires mlflow"
)
def test_decorator_extracts_pandas_metrics(tmp_path, monkeypatch):
    """Test that decorators extract metrics from pandas DataFrames."""
    pytest.importorskip("pandas")
    import pandas as pd

    tracking_uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_TRACKING_ENABLED", "true")

    reset_tracker()
    tracker = get_tracker()

    @track_tool_call(tool_name="pandas_tool")
    def get_dataframe():
        return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    with tracker.track_session("test_session"):
        df = get_dataframe()
        assert len(df) == 3
        assert len(df.columns) == 2
