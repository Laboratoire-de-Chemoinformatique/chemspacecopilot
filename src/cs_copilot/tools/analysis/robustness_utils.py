#!/usr/bin/env python
# coding: utf-8
"""
Helper utilities for robustness test analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def parse_results_path(path: str) -> Tuple[str, str]:
    """
    Extract test_name and timestamp from results path.

    Args:
        path: Path like 'robustness_tests/chembl_download/20251231_103045/results.json'
              or 'tests/robustness/reports/20251231_103045/results.json'

    Returns:
        Tuple of (test_name, timestamp)
    """
    path_obj = Path(path)

    # Pattern 1: robustness_tests/{test_name}/{timestamp}/...
    if "robustness_tests" in str(path):
        parts = path_obj.parts
        idx = parts.index("robustness_tests")
        if len(parts) > idx + 2:
            test_name = parts[idx + 1]
            timestamp = parts[idx + 2]
            return test_name, timestamp

    # Pattern 2: tests/robustness/reports/{timestamp}/...
    if "reports" in str(path):
        parts = path_obj.parts
        idx = parts.index("reports")
        if len(parts) > idx + 1:
            timestamp = parts[idx + 1]
            # Try to extract test name from results.json
            try:
                import json

                results_path = path_obj.parent / "results.json"
                with open(results_path, "r") as f:
                    data = json.load(f)
                    test_name = data.get("test_name", "unknown")
                    return test_name, timestamp
            except Exception:
                return "unknown", timestamp

    raise ValueError(f"Could not parse test_name and timestamp from path: {path}")


def compute_aggregated_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate metrics across multiple test results.

    Args:
        results: List of test result dictionaries

    Returns:
        Dictionary of aggregated metrics
    """
    if not results:
        return {}

    # Extract all scores
    all_scores = []
    component_scores = {
        "data_similarity": [],
        "semantic_similarity": [],
        "process_consistency": [],
        "visual_similarity": [],
    }

    for result in results:
        # Overall score
        if "robustness_score" in result:
            all_scores.append(result["robustness_score"])

        # Component scores
        if "comparisons" in result:
            comps = result["comparisons"]
            if "data" in comps:
                component_scores["data_similarity"].append(comps["data"].get("row_jaccard", 0))
            if "text" in comps:
                component_scores["semantic_similarity"].append(
                    comps["text"].get("semantic_similarity", 0)
                )
            if "process" in comps:
                component_scores["process_consistency"].append(
                    comps["process"].get("completion_rate", 0)
                )
            if "visual" in comps:
                component_scores["visual_similarity"].append(comps["visual"].get("ssim", 0))

    aggregated = {}

    # Overall metrics
    if all_scores:
        aggregated["mean_score"] = np.mean(all_scores)
        aggregated["median_score"] = np.median(all_scores)
        aggregated["std_score"] = np.std(all_scores)
        aggregated["min_score"] = np.min(all_scores)
        aggregated["max_score"] = np.max(all_scores)

    # Component metrics
    for comp_name, scores in component_scores.items():
        if scores:
            aggregated[f"{comp_name}_mean"] = np.mean(scores)
            aggregated[f"{comp_name}_std"] = np.std(scores)

    return aggregated


def detect_outliers(scores: List[float], method: str = "iqr", threshold: float = 1.5) -> List[int]:
    """
    Detect outliers in a list of scores using IQR or Z-score method.

    Args:
        scores: List of numerical scores
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection (1.5 for IQR, 2.0 for Z-score)

    Returns:
        List of indices of outlier scores
    """
    if len(scores) < 4:
        return []

    scores_array = np.array(scores)
    outlier_indices = []

    if method == "iqr":
        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_indices = [
            i for i, score in enumerate(scores) if score < lower_bound or score > upper_bound
        ]

    elif method == "zscore":
        mean = np.mean(scores_array)
        std = np.std(scores_array)

        if std > 0:
            z_scores = np.abs((scores_array - mean) / std)
            outlier_indices = [i for i, z in enumerate(z_scores) if z > threshold]

    logger.info(f"Detected {len(outlier_indices)} outliers using {method} method")
    return outlier_indices


def categorize_failures(results: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """
    Group failures by type (timeout, validation, tool failure, etc).

    Args:
        results: List of variation results

    Returns:
        Dictionary mapping failure types to lists of failed variations
    """
    categories = {
        "timeout": [],
        "validation_error": [],
        "tool_error": [],
        "low_score": [],
        "other": [],
    }

    for result in results:
        if result.get("success"):
            # Check for low scores
            score = result.get("robustness_score", 1.0)
            if score < 0.70:
                categories["low_score"].append(result)
        else:
            # Categorize by error message
            error = result.get("error", "").lower()

            if "timeout" in error or "timed out" in error:
                categories["timeout"].append(result)
            elif "validation" in error or "invalid" in error:
                categories["validation_error"].append(result)
            elif "tool" in error or "function" in error:
                categories["tool_error"].append(result)
            else:
                categories["other"].append(result)

    # Log summary
    for cat, items in categories.items():
        if items:
            logger.info(f"  {cat}: {len(items)} failures")

    return categories


def generate_comparison_table(run1: Dict[str, Any], run2: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate side-by-side comparison table for two test runs.

    Args:
        run1: First test run results
        run2: Second test run results

    Returns:
        DataFrame with comparison
    """
    metrics = [
        "total_tests",
        "passed",
        "failed",
        "mean_score",
        "median_score",
        "std_score",
        "rating",
    ]

    data = {
        "Metric": metrics,
        "Run 1": [run1.get(m, "N/A") for m in metrics],
        "Run 2": [run2.get(m, "N/A") for m in metrics],
    }

    # Add difference column for numeric metrics
    differences = []
    for m in metrics:
        val1 = run1.get(m)
        val2 = run2.get(m)

        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            differences.append(f"{diff:+.3f}" if isinstance(diff, float) else f"{diff:+d}")
        else:
            differences.append("-")

    data["Difference"] = differences

    return pd.DataFrame(data)


def format_recommendations(analysis: Dict[str, Any]) -> List[str]:
    """
    Convert analysis to actionable recommendations.

    Args:
        analysis: Analysis dictionary

    Returns:
        List of prioritized recommendation strings
    """
    recommendations = []

    # Priority 1: Critical issues (score < 0.70)
    mean_score = analysis.get("mean_score", 1.0)
    if mean_score < 0.70:
        recommendations.append(
            "🚨 CRITICAL: Robustness score is below acceptable threshold (0.70). "
            "Immediate action required to improve prompt handling consistency."
        )

        # Identify specific issues
        if analysis.get("data_similarity_mean", 1.0) < 0.70:
            recommendations.append(
                "  → Data inconsistency detected. Review data fetching and filtering logic."
            )
        if analysis.get("semantic_similarity_mean", 1.0) < 0.70:
            recommendations.append(
                "  → Semantic inconsistency detected. Review agent instructions and LLM prompts."
            )
        if analysis.get("process_consistency_mean", 1.0) < 0.70:
            recommendations.append(
                "  → Process inconsistency detected. Review tool call logic and error handling."
            )

    # Priority 2: Regressions
    if analysis.get("trend") == "Regression":
        recommendations.append(
            "⚠️  Regression detected. Recent changes decreased robustness. "
            "Review recent commits to agent instructions or tool implementations."
        )

    # Priority 3: High variability
    std_score = analysis.get("std_score", 0)
    if std_score > 0.15:
        recommendations.append(
            f"⚠️  High score variability (std={std_score:.3f}). "
            "Results are inconsistent across prompt variations. "
            "Consider adding more explicit instructions or constraints."
        )

    # Priority 4: Improvements to maintain
    if mean_score >= 0.80:
        recommendations.append("✅ Good robustness score. Continue monitoring to maintain quality.")

    # Priority 5: Specific component recommendations
    if "component_scores" in analysis:
        comp = analysis["component_scores"]
        if comp.get("visual_similarity", 1.0) < 0.80:
            recommendations.append(
                "💡 Consider stabilizing visualization parameters to improve visual consistency."
            )

    # Default if no specific issues
    if not recommendations:
        recommendations.append(
            "✅ No critical issues detected. System is performing within expected parameters."
        )

    return recommendations


def extract_dataset_stats(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract dataset statistics from ChEMBL test results.

    Args:
        results: List of variation results with dataset info

    Returns:
        DataFrame with dataset statistics
    """
    data = []

    for result in results:
        if "dataset_name" in result:
            data.append(
                {
                    "dataset_name": result.get("dataset_name", "unknown"),
                    "row_count": result.get("row_count", 0),
                    "success": result.get("success", False),
                    "robustness_score": result.get("robustness_score", 0),
                }
            )

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # Aggregate by dataset
    summary = (
        df.groupby("dataset_name")
        .agg({"row_count": "mean", "success": "sum", "robustness_score": "mean"})
        .reset_index()
    )

    summary["success_rate"] = summary["success"] / df.groupby("dataset_name").size()

    return summary


def format_time_range(timestamps: List[str]) -> str:
    """
    Format a time range from list of timestamps.

    Args:
        timestamps: List of timestamp strings (YYYYMMDD_HHMMSS)

    Returns:
        Human-readable time range string
    """
    if not timestamps:
        return "N/A"

    sorted_ts = sorted(timestamps)
    first = sorted_ts[0]
    last = sorted_ts[-1]

    if first == last:
        return f"{first}"

    # Parse timestamps
    try:
        from datetime import datetime

        first_dt = datetime.strptime(first, "%Y%m%d_%H%M%S")
        last_dt = datetime.strptime(last, "%Y%m%d_%H%M%S")

        return f"{first_dt.strftime('%Y-%m-%d %H:%M')} to {last_dt.strftime('%Y-%m-%d %H:%M')}"
    except Exception:
        return f"{first} to {last}"
