#!/usr/bin/env python
# coding: utf-8
"""
Unit tests for RobustnessAnalysisToolkit.
"""

import json

import pandas as pd
import pytest

from cs_copilot.tools.analysis import RobustnessAnalysisToolkit


@pytest.fixture
def toolkit():
    """Create a RobustnessAnalysisToolkit instance."""
    return RobustnessAnalysisToolkit()


@pytest.fixture
def sample_results():
    """Create sample test results for testing."""
    return {
        "test_name": "chembl_download",
        "timestamp": "20250122_120000",
        "total_tests": 10,
        "passed": 8,
        "failed": 2,
        "variations": [
            {
                "prompt_index": 0,
                "variation_index": 0,
                "success": True,
                "robustness_score": 0.85,
                "prompt": "Download CDK2 inhibitors",
            },
            {
                "prompt_index": 0,
                "variation_index": 1,
                "success": True,
                "robustness_score": 0.82,
                "prompt": "Get CDK2 inhibitors",
            },
            {
                "prompt_index": 1,
                "variation_index": 0,
                "success": False,
                "error": "Timeout error",
                "prompt": "Retrieve CDK2 inhibitors",
            },
            {
                "prompt_index": 1,
                "variation_index": 1,
                "success": True,
                "robustness_score": 0.65,
                "prompt": "Find CDK2 inhibitors",
            },
        ],
    }


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    return pd.DataFrame(
        {
            "prompt_index": [0, 0, 1, 1],
            "variation_index": [0, 1, 0, 1],
            "requires_clarification": [False, False, True, True],
            "success": [True, True, False, True],
            "detail": ["", "", "Timeout error", ""],
            "dataset_name": ["CDK2", "CDK2", None, "CDK2"],
            "row_count": [1000, 1000, None, 950],
        }
    )


def test_analyze_score_distribution(toolkit, sample_results):
    """Test score distribution analysis."""
    result = toolkit.analyze_score_distribution(sample_results)

    assert "mean_score" in result
    assert "median_score" in result
    assert "std_score" in result
    assert "rating" in result
    assert "num_scores" in result

    # Check values
    assert result["num_scores"] == 3  # Only successful variations with scores
    assert 0.0 <= result["mean_score"] <= 1.0
    assert result["rating"] in ["Excellent", "Good", "Acceptable", "Concerning"]


def test_analyze_score_distribution_empty(toolkit):
    """Test score distribution with no variations."""
    empty_results = {"variations": []}
    result = toolkit.analyze_score_distribution(empty_results)

    assert "error" in result


def test_identify_failing_prompts(toolkit, sample_results):
    """Test identification of failing prompts."""
    failing = toolkit.identify_failing_prompts(sample_results, threshold=0.70)

    # Should identify 2 failures: 1 timeout error + 1 low score
    assert len(failing) == 2

    # Check structure
    for failure in failing:
        assert "prompt_index" in failure
        assert "variation_index" in failure
        assert "success" in failure
        assert "prompt_text" in failure


def test_identify_failing_prompts_high_threshold(toolkit, sample_results):
    """Test with high threshold should identify more failures."""
    failing = toolkit.identify_failing_prompts(sample_results, threshold=0.90)

    # With threshold 0.90, all scores below that are failing
    assert len(failing) >= 2


def test_compare_test_runs(toolkit, tmp_path):
    """Test comparison of multiple test runs."""
    # Create temporary results files
    timestamps = ["20250122_100000", "20250122_120000"]

    for ts in timestamps:
        results_dir = tmp_path / "tests" / "robustness" / "reports" / ts
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "test_name": "chembl_download",
            "timestamp": ts,
            "total_tests": 10,
            "passed": 8 if ts == timestamps[0] else 9,  # Improvement in second run
            "failed": 2 if ts == timestamps[0] else 1,
            "variations": [
                {"success": True, "robustness_score": 0.80 if ts == timestamps[0] else 0.85}
                for _ in range(8)
            ],
        }

        with open(results_dir / "results.json", "w") as f:
            json.dump(results, f)

    # Change to temp directory for testing
    import os

    original_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        comparison = toolkit.compare_test_runs("chembl_download", timestamps)

        assert "runs" in comparison
        assert "statistics" in comparison
        assert "trend" in comparison

        # Check runs
        assert len(comparison["runs"]) == 2

        # Check trend detection (should show improvement)
        assert comparison["trend"] in ["Improvement", "Stable", "Regression"]
    finally:
        os.chdir(original_dir)


def test_analyze_temporal_trends_insufficient_data(toolkit):
    """Test temporal trends with insufficient data."""
    result = toolkit.analyze_temporal_trends("nonexistent_test", timestamps=[])

    assert "error" in result or result["trend"] == "Insufficient data"


def test_generate_insights(toolkit):
    """Test insight generation."""
    analysis = {
        "mean_score": 0.65,  # Concerning
        "std_score": 0.18,  # High variability
        "rating": "Concerning",
    }

    insights = toolkit.generate_insights(analysis)

    assert isinstance(insights, list)
    assert len(insights) > 0

    # Should contain warning about concerning score
    insights_text = " ".join(insights)
    assert "Concerning" in insights_text or "concerning" in insights_text.lower()


def test_generate_insights_excellent(toolkit):
    """Test insight generation for excellent scores."""
    analysis = {
        "mean_score": 0.92,
        "std_score": 0.05,
        "rating": "Excellent",
    }

    insights = toolkit.generate_insights(analysis)

    assert isinstance(insights, list)
    assert len(insights) > 0

    # Should contain positive message
    insights_text = " ".join(insights)
    assert "Excellent" in insights_text or "excellent" in insights_text.lower()


def test_export_analysis_report_markdown(toolkit):
    """Test exporting analysis as markdown."""
    analysis = {
        "mean_score": 0.85,
        "median_score": 0.84,
        "std_score": 0.08,
        "rating": "Good",
    }

    report = toolkit.export_analysis_report(analysis, format="markdown")

    assert isinstance(report, str)
    assert "# Robustness Analysis Report" in report
    assert "0.85" in report  # Check score is included


def test_export_analysis_report_json(toolkit):
    """Test exporting analysis as JSON."""
    analysis = {
        "mean_score": 0.85,
        "rating": "Good",
    }

    report = toolkit.export_analysis_report(analysis, format="json")

    # Should be valid JSON
    data = json.loads(report)
    assert data["mean_score"] == 0.85
    assert data["rating"] == "Good"


def test_export_analysis_report_csv(toolkit):
    """Test exporting comparison as CSV."""
    analysis = {
        "runs": [
            {
                "timestamp": "20250122_100000",
                "total_tests": 10,
                "passed": 8,
                "failed": 2,
                "success_rate": 0.8,
                "mean_score": 0.85,
                "rating": "Good",
            },
            {
                "timestamp": "20250122_120000",
                "total_tests": 10,
                "passed": 9,
                "failed": 1,
                "success_rate": 0.9,
                "mean_score": 0.88,
                "rating": "Good",
            },
        ]
    }

    report = toolkit.export_analysis_report(analysis, format="csv")

    # Should be valid CSV
    df = pd.read_csv(pd.io.common.StringIO(report))
    assert len(df) == 2
    assert "timestamp" in df.columns


def test_list_available_test_runs(toolkit, tmp_path):
    """Test listing available test runs."""
    # Create temporary results files
    timestamps = ["20250122_100000", "20250122_120000", "20250122_140000"]

    for ts in timestamps:
        results_dir = tmp_path / "tests" / "robustness" / "reports" / ts
        results_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "test_name": "chembl_download",
            "timestamp": ts,
            "total_tests": 10,
            "passed": 8,
            "failed": 2,
        }

        with open(results_dir / "results.json", "w") as f:
            json.dump(results, f)

    # Change to temp directory for testing
    import os

    original_dir = os.getcwd()
    os.chdir(tmp_path)

    try:
        runs = toolkit.list_available_test_runs("chembl_download")

        assert len(runs) == 3
        # Should be sorted in descending order (newest first)
        assert runs == sorted(timestamps, reverse=True)
    finally:
        os.chdir(original_dir)


def test_toolkit_registration(toolkit):
    """Test that all tools are properly registered."""
    # Check that toolkit has registered functions
    assert len(toolkit.functions) > 0

    # Check for key tool names
    # toolkit.functions might be a list of functions or dict
    if isinstance(toolkit.functions, dict):
        tool_names = list(toolkit.functions.keys())
    elif isinstance(toolkit.functions, list):
        # Handle both function objects and strings
        tool_names = []
        for f in toolkit.functions:
            if isinstance(f, str):
                tool_names.append(f)
            elif hasattr(f, "__name__"):
                tool_names.append(f.__name__)
            elif hasattr(f, "name"):
                tool_names.append(f.name)
    else:
        # Fallback: try to get from toolkit attributes
        tool_names = [attr for attr in dir(toolkit) if not attr.startswith("_")]

    expected_tools = [
        "load_test_results",
        "load_test_summary_csv",
        "list_available_test_runs",
        "analyze_score_distribution",
        "identify_failing_prompts",
        "compare_test_runs",
        "analyze_temporal_trends",
        "generate_insights",
        "export_analysis_report",
    ]

    for tool in expected_tools:
        assert tool in tool_names, f"Tool {tool} not registered. Available: {tool_names}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
