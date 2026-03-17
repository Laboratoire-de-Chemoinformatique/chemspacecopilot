"""Unit tests for MLflow tracking utilities."""

from cs_copilot.tracking.utils import (
    Timer,
    build_prompt_signature,
    calculate_cost,
    count_tokens,
    format_duration,
    merge_metrics,
    safe_log_value,
    sanitize_run_name,
)


def test_count_tokens():
    """Test token counting heuristic."""
    assert count_tokens("") == 0
    assert count_tokens("test") == 1
    assert count_tokens("This is a test message") > 0
    # Approximately 4 chars per token
    text = "a" * 400
    assert count_tokens(text) == 100


def test_calculate_cost():
    """Test cost calculation."""
    cost = calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        cost_per_1k_prompt=0.001,
        cost_per_1k_completion=0.002,
    )
    expected = (1000 / 1000 * 0.001) + (500 / 1000 * 0.002)
    assert abs(cost - expected) < 0.0001

    # Zero tokens
    cost = calculate_cost(
        prompt_tokens=0, completion_tokens=0, cost_per_1k_prompt=0.001, cost_per_1k_completion=0.002
    )
    assert cost == 0.0


def test_safe_log_value():
    """Test safe value conversion."""
    assert safe_log_value(42) == 42.0
    assert safe_log_value(3.14) == 3.14
    assert safe_log_value("123") == 123.0
    assert safe_log_value(None) is None
    assert safe_log_value("not_a_number") is None
    assert safe_log_value([1, 2, 3]) is None


def test_format_duration():
    """Test duration formatting."""
    assert format_duration(30) == "30.0s"
    assert format_duration(45.7) == "45.7s"
    assert format_duration(60) == "1m 0s"
    assert format_duration(90) == "1m 30s"
    assert format_duration(125.6) == "2m 6s"
    assert format_duration(3665) == "61m 5s"


def test_timer():
    """Test Timer context manager."""
    import time

    with Timer() as timer:
        time.sleep(0.1)

    assert timer.get_duration() >= 0.1
    assert timer.get_duration() < 0.2  # Allow some margin


def test_timer_no_context():
    """Test Timer without context manager."""
    timer = Timer()
    assert timer.get_duration() == 0.0


def test_sanitize_run_name():
    """Test run name sanitization."""
    # Normal name
    assert sanitize_run_name("test_run") == "test_run"

    # Replace invalid characters
    assert sanitize_run_name("test/run") == "test_run"
    assert sanitize_run_name("test\\run") == "test_run"
    assert sanitize_run_name("test:run") == "test_run"

    # Truncate long names
    long_name = "a" * 200
    sanitized = sanitize_run_name(long_name, max_length=100)
    assert len(sanitized) == 100

    # Mixed case
    assert sanitize_run_name("test/path\\to:file", max_length=50) == "test_path_to_file"


def test_merge_metrics():
    """Test metric dictionary merging."""
    existing = {"metric_a": 10.0, "metric_b": 20.0}
    new = {"metric_b": 5.0, "metric_c": 30.0}

    merged = merge_metrics(existing, new)

    assert merged["metric_a"] == 10.0
    assert merged["metric_b"] == 25.0  # Sum of overlapping
    assert merged["metric_c"] == 30.0

    # Original should not be modified
    assert existing["metric_b"] == 20.0


def test_merge_metrics_empty():
    """Test merging with empty dictionaries."""
    existing = {"metric_a": 10.0}
    new = {}

    merged = merge_metrics(existing, new)
    assert merged == {"metric_a": 10.0}

    existing = {}
    new = {"metric_b": 20.0}

    merged = merge_metrics(existing, new)
    assert merged == {"metric_b": 20.0}


def test_build_prompt_signature_none():
    """Test prompt signature with empty input."""
    assert build_prompt_signature(None) is None
    assert build_prompt_signature([]) is None
    assert build_prompt_signature(["", "  "]) is None


def test_build_prompt_signature_list():
    """Test prompt signature from list of instructions."""
    signature = build_prompt_signature([" First line ", None, "Second line"])
    assert signature is not None
    assert signature.text == "First line\nSecond line"
    assert signature.version.startswith("sha256:")
    assert len(signature.version) == len("sha256:") + 12
    assert signature.num_lines == 2
    assert signature.num_chars == len(signature.text)


def test_build_prompt_signature_string():
    """Test prompt signature from string prompt."""
    signature = build_prompt_signature("  Just a prompt ")
    assert signature is not None
    assert signature.text == "Just a prompt"
    assert signature.num_lines == 1
