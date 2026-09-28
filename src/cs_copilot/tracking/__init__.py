"""MLflow tracking integration for Cs_copilot.

This module provides comprehensive tracking for agent executions, tool calls,
and user sessions using MLflow GenAI.

Example usage:

    from cs_copilot.tracking import get_tracker, track_agent_run, track_tool_call

    # Get global tracker
    tracker = get_tracker()

    # Track a session
    with tracker.track_session("session_123", user_id="user@example.com"):
        # Track agent execution
        with tracker.track_agent_run("ChEMBL Downloader", "Download active compounds"):
            # Track tool call
            with tracker.track_tool_call("search_chembl"):
                result = perform_search()

    # Or use decorators
    @track_agent_run(agent_name="My Agent")
    def my_agent(prompt: str):
        return process(prompt)

    @track_tool_call(tool_name="my_tool")
    def my_tool(arg1: str, arg2: int):
        return compute(arg1, arg2)
"""

from .config import MLflowConfig
from .core import MLflowTracker, get_tracker, reset_tracker
from .decorators import track_agent_run, track_tool_call
from .replay import (
    GoldenTrajectory,
    TrajectoryReport,
    evaluate_trajectory,
    golden_for_workflow,
    log_trajectory_report,
    replay_run,
)
from .streaming_buffer import StreamingBuffer
from .utils import calculate_cost, count_tokens, format_duration

__all__ = [
    # Configuration
    "MLflowConfig",
    # Core tracking
    "MLflowTracker",
    "get_tracker",
    "reset_tracker",
    # Decorators
    "track_agent_run",
    "track_tool_call",
    # Replay and golden trajectories
    "GoldenTrajectory",
    "TrajectoryReport",
    "evaluate_trajectory",
    "golden_for_workflow",
    "log_trajectory_report",
    "replay_run",
    # Utilities
    "StreamingBuffer",
    "count_tokens",
    "calculate_cost",
    "format_duration",
]
