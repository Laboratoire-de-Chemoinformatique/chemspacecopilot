"""Unit tests for StreamingBuffer."""

import time

from cs_copilot.tracking.streaming_buffer import StreamingBuffer


def test_streaming_buffer_text_chunks():
    """Test adding and retrieving text chunks."""
    buffer = StreamingBuffer()

    buffer.add_text_chunk("Hello ")
    buffer.add_text_chunk("world")
    buffer.add_text_chunk("!")

    assert buffer.get_full_text() == "Hello world!"
    assert len(buffer.text_chunks) == 3


def test_streaming_buffer_tool_calls():
    """Test adding tool calls."""
    buffer = StreamingBuffer()

    buffer.add_tool_call("search", {"query": "test"}, result="found")
    buffer.add_tool_call("filter", {"criteria": "active"})

    assert len(buffer.tool_calls) == 2
    assert buffer.tool_calls[0]["tool"] == "search"
    assert buffer.tool_calls[0]["args"] == {"query": "test"}
    assert buffer.tool_calls[0]["result"] == "found"
    assert buffer.tool_calls[1]["tool"] == "filter"


def test_streaming_buffer_metadata():
    """Test setting and retrieving metadata."""
    buffer = StreamingBuffer()

    buffer.set_metadata("agent", "test_agent")
    buffer.set_metadata("session_id", "123")

    assert buffer.metadata["agent"] == "test_agent"
    assert buffer.metadata["session_id"] == "123"


def test_streaming_buffer_metrics():
    """Test metrics generation."""
    buffer = StreamingBuffer()

    buffer.add_text_chunk("Hello")
    buffer.add_text_chunk(" world")
    buffer.add_tool_call("tool1", {})
    buffer.add_tool_call("tool2", {})

    buffer.start_time = time.time()
    time.sleep(0.1)
    buffer.end_time = time.time()

    metrics = buffer.get_metrics()

    assert metrics["text_chunks_count"] == 2.0
    assert metrics["tool_calls_count"] == 2.0
    assert metrics["total_text_length"] == 11.0  # "Hello world"
    assert "streaming_duration_seconds" in metrics
    assert metrics["streaming_duration_seconds"] > 0


def test_streaming_buffer_metrics_no_timing():
    """Test metrics without timing information."""
    buffer = StreamingBuffer()

    buffer.add_text_chunk("test")

    metrics = buffer.get_metrics()

    assert "streaming_duration_seconds" not in metrics
    assert metrics["text_chunks_count"] == 1.0


def test_streaming_buffer_params():
    """Test parameters generation."""
    buffer = StreamingBuffer()

    buffer.add_tool_call("tool_a", {})
    buffer.add_tool_call("tool_b", {})
    buffer.add_tool_call("tool_a", {})  # Duplicate

    params = buffer.get_params()

    assert "tools_used" in params
    # Should contain unique tool names
    assert "tool_a" in params["tools_used"]
    assert "tool_b" in params["tools_used"]


def test_streaming_buffer_params_empty():
    """Test parameters with no tool calls."""
    buffer = StreamingBuffer()

    params = buffer.get_params()

    assert "tools_used" not in params or params["tools_used"] == ""


def test_streaming_buffer_clear():
    """Test clearing buffer."""
    buffer = StreamingBuffer()

    buffer.add_text_chunk("Hello")
    buffer.add_tool_call("tool", {})
    buffer.set_metadata("key", "value")
    buffer.start_time = time.time()

    buffer.clear()

    assert len(buffer.text_chunks) == 0
    assert len(buffer.tool_calls) == 0
    assert len(buffer.metadata) == 0
    assert buffer.start_time is None
    assert buffer.end_time is None


def test_streaming_buffer_empty():
    """Test empty buffer behavior."""
    buffer = StreamingBuffer()

    assert buffer.get_full_text() == ""
    assert buffer.get_metrics()["text_chunks_count"] == 0.0
    assert buffer.get_metrics()["tool_calls_count"] == 0.0
    assert buffer.get_params() == {}
