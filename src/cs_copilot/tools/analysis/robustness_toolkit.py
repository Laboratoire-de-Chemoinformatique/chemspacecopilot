#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for analyzing robustness test results.

Provides tools for loading, analyzing, and comparing robustness test results
from S3 or local storage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.tools import Toolkit

from cs_copilot.storage import S3, is_s3_enabled

logger = logging.getLogger(__name__)

# Robustness rating thresholds
RATING_THRESHOLDS = {
    "excellent": 0.90,
    "good": 0.80,
    "acceptable": 0.70,
}


class RobustnessAnalysisToolkit(Toolkit):
    """Toolkit for analyzing robustness test results."""

    def __init__(self):
        super().__init__(name="robustness_analysis")

        # Register all tools
        self.register(self.load_test_results)
        self.register(self.load_test_summary_csv)
        self.register(self.list_available_test_runs)
        self.register(self.analyze_score_distribution)
        self.register(self.identify_failing_prompts)
        self.register(self.compare_test_runs)
        self.register(self.analyze_temporal_trends)
        self.register(self.generate_insights)
        self.register(self.export_analysis_report)

    def load_test_results(self, test_name: str, timestamp: str) -> Dict[str, Any]:
        """
        Load robustness test results from S3 or local storage.

        Args:
            test_name: Name of the test (e.g., 'chembl_download', 'chembl_interactivity')
            timestamp: Timestamp of the test run (format: YYYYMMDD_HHMMSS)

        Returns:
            Dictionary containing test results with keys:
            - test_name: Name of the test
            - timestamp: Test run timestamp
            - total_tests: Total number of tests
            - passed: Number of passed tests
            - failed: Number of failed tests
            - variations: List of variation results
        """
        # Try S3 first if enabled
        if is_s3_enabled():
            json_path = f"robustness_tests/{test_name}/{timestamp}/results.json"
            try:
                logger.info(f"Loading test results from S3: {S3.path(json_path)}")
                with S3.open(json_path, "r") as f:
                    results = json.load(f)
                logger.info(f"✅ Loaded {results.get('total_tests', 0)} test results")
                return results
            except Exception as e:
                logger.warning(f"Could not load from S3: {e}")

        # Fallback to local storage
        local_path = Path(f"tests/robustness/reports/{timestamp}/results.json")
        if local_path.exists():
            logger.info(f"Loading test results from local: {local_path}")
            with open(local_path, "r") as f:
                results = json.load(f)
            logger.info(f"✅ Loaded {results.get('total_tests', 0)} test results")
            return results

        raise FileNotFoundError(
            f"Could not find test results for {test_name} at {timestamp} " f"in S3 or local storage"
        )

    def load_test_summary_csv(self, test_name: str, timestamp: str) -> pd.DataFrame:
        """
        Load test summary CSV as a DataFrame.

        Args:
            test_name: Name of the test
            timestamp: Timestamp of the test run

        Returns:
            DataFrame with columns:
            - prompt_index: Index of the prompt template
            - variation_index: Index of the variation
            - requires_clarification: Whether prompt requires clarification
            - success: Whether the test passed
            - detail: Error detail if failed
            - dataset_name: Dataset name (for ChEMBL tests)
            - row_count: Number of rows (for data tests)
        """
        # Try S3 first if enabled
        if is_s3_enabled():
            csv_path = f"robustness_tests/{test_name}/{timestamp}/summary.csv"
            try:
                logger.info(f"Loading summary CSV from S3: {S3.path(csv_path)}")
                with S3.open(csv_path, "r") as f:
                    df = pd.read_csv(f)
                logger.info(f"✅ Loaded summary with {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Could not load from S3: {e}")

        # Fallback to local storage
        local_path = Path(f"tests/robustness/reports/{timestamp}/summary.csv")
        if local_path.exists():
            logger.info(f"Loading summary CSV from local: {local_path}")
            df = pd.read_csv(local_path)
            logger.info(f"✅ Loaded summary with {len(df)} rows")
            return df

        raise FileNotFoundError(
            f"Could not find summary CSV for {test_name} at {timestamp} " f"in S3 or local storage"
        )

    def list_available_test_runs(self, test_name: str) -> List[str]:
        """
        List all available test run timestamps for a given test type.

        Args:
            test_name: Name of the test

        Returns:
            List of timestamp strings in descending order (newest first)
        """
        timestamps = []

        # Check local storage
        local_reports_dir = Path("tests/robustness/reports")
        if local_reports_dir.exists():
            for timestamp_dir in local_reports_dir.iterdir():
                if timestamp_dir.is_dir():
                    results_file = timestamp_dir / "results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, "r") as f:
                                data = json.load(f)
                                if data.get("test_name") == test_name:
                                    timestamps.append(timestamp_dir.name)
                        except Exception as e:
                            logger.debug(f"Could not read {results_file}: {e}")

        # Sort by timestamp (newest first)
        timestamps.sort(reverse=True)

        logger.info(f"Found {len(timestamps)} test runs for {test_name}")
        return timestamps

    def analyze_score_distribution(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute statistics on robustness scores from test results.

        Args:
            results: Test results dictionary from load_test_results

        Returns:
            Dictionary containing:
            - mean_score: Average robustness score
            - median_score: Median robustness score
            - std_score: Standard deviation
            - min_score: Minimum score
            - max_score: Maximum score
            - rating: Overall rating (Excellent/Good/Acceptable/Concerning)
            - distribution: Histogram bins and counts
        """
        variations = results.get("variations", [])
        if not variations:
            return {"error": "No variations found in results"}

        # Extract scores from successful variations
        scores = []
        for var in variations:
            if var.get("success") and "robustness_score" in var:
                scores.append(var["robustness_score"])

        if not scores:
            return {"error": "No successful variations with scores found"}

        mean_score = sum(scores) / len(scores)
        sorted_scores = sorted(scores)
        median_score = sorted_scores[len(sorted_scores) // 2]

        # Calculate standard deviation
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = variance**0.5

        # Get rating
        rating = self._get_rating(mean_score)

        # Create histogram (5 bins)
        min_val, max_val = min(scores), max(scores)
        if max_val == min_val:
            bins = [min_val]
            counts = [len(scores)]
        else:
            bin_size = (max_val - min_val) / 5
            bins = [min_val + i * bin_size for i in range(6)]
            counts = [sum(1 for s in scores if bins[i] <= s < bins[i + 1]) for i in range(5)]

        return {
            "mean_score": mean_score,
            "median_score": median_score,
            "std_score": std_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "rating": rating,
            "num_scores": len(scores),
            "distribution": {"bins": bins, "counts": counts},
        }

    def identify_failing_prompts(
        self, results: Dict[str, Any], threshold: float = 0.70
    ) -> List[Dict[str, Any]]:
        """
        Identify prompts that failed or scored below threshold.

        Args:
            results: Test results dictionary
            threshold: Minimum acceptable score (default 0.70)

        Returns:
            List of dictionaries with:
            - prompt_index: Index of failing prompt
            - variation_index: Index of failing variation
            - success: Whether test passed
            - robustness_score: Score if available
            - error: Error message if failed
            - prompt_text: Text of the prompt
        """
        variations = results.get("variations", [])
        failing = []

        for var in variations:
            prompt_idx = var.get("prompt_index", 0)
            var_idx = var.get("variation_index", 0)
            success = var.get("success", False)
            score = var.get("robustness_score")

            # Identify failures
            is_failing = not success or (score is not None and score < threshold)

            if is_failing:
                failing.append(
                    {
                        "prompt_index": prompt_idx,
                        "variation_index": var_idx,
                        "success": success,
                        "robustness_score": score,
                        "error": var.get("error", ""),
                        "prompt_text": var.get("prompt", "")[:100] + "...",
                    }
                )

        logger.info(f"Identified {len(failing)} failing prompts (threshold={threshold})")
        return failing

    def compare_test_runs(self, test_name: str, timestamps: List[str]) -> Dict[str, Any]:
        """
        Compare results across multiple test runs.

        Args:
            test_name: Name of the test
            timestamps: List of timestamps to compare

        Returns:
            Dictionary with comparison data:
            - runs: List of run summaries
            - statistics: Overall statistics
            - trend: Improvement/Regression/Stable
        """
        runs = []
        for ts in timestamps:
            try:
                results = self.load_test_results(test_name, ts)
                score_dist = self.analyze_score_distribution(results)

                runs.append(
                    {
                        "timestamp": ts,
                        "total_tests": results.get("total_tests", 0),
                        "passed": results.get("passed", 0),
                        "failed": results.get("failed", 0),
                        "success_rate": (results.get("passed", 0) / results.get("total_tests", 1)),
                        "mean_score": score_dist.get("mean_score", 0),
                        "rating": score_dist.get("rating", "Unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not load results for {ts}: {e}")

        if not runs:
            return {"error": "No runs could be loaded"}

        # Calculate statistics
        success_rates = [r["success_rate"] for r in runs]
        mean_scores = [r["mean_score"] for r in runs]

        # Determine trend
        if len(runs) >= 2:
            first_score = runs[-1]["mean_score"]  # Oldest
            last_score = runs[0]["mean_score"]  # Newest
            diff = last_score - first_score

            if diff > 0.05:
                trend = "Improvement"
            elif diff < -0.05:
                trend = "Regression"
            else:
                trend = "Stable"
        else:
            trend = "Unknown"

        return {
            "runs": runs,
            "statistics": {
                "avg_success_rate": sum(success_rates) / len(success_rates),
                "min_success_rate": min(success_rates),
                "max_success_rate": max(success_rates),
                "avg_mean_score": sum(mean_scores) / len(mean_scores),
                "min_mean_score": min(mean_scores),
                "max_mean_score": max(mean_scores),
            },
            "trend": trend,
        }

    def analyze_temporal_trends(
        self, test_name: str, timestamps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze improvements/regressions over time.

        Args:
            test_name: Name of the test
            timestamps: List of timestamps (if None, use all available)

        Returns:
            Dictionary with temporal analysis:
            - timeline: List of (timestamp, score) tuples
            - trend: Overall trend direction
            - improvements: List of improvements
            - regressions: List of regressions
        """
        if timestamps is None:
            timestamps = self.list_available_test_runs(test_name)

        if not timestamps:
            return {"error": f"No test runs found for {test_name}"}

        # Load scores for all timestamps
        timeline = []
        for ts in sorted(timestamps):  # Sort chronologically
            try:
                results = self.load_test_results(test_name, ts)
                score_dist = self.analyze_score_distribution(results)
                timeline.append(
                    {
                        "timestamp": ts,
                        "mean_score": score_dist.get("mean_score", 0),
                        "rating": score_dist.get("rating", "Unknown"),
                    }
                )
            except Exception as e:
                logger.warning(f"Could not load results for {ts}: {e}")

        if len(timeline) < 2:
            return {
                "timeline": timeline,
                "trend": "Insufficient data",
                "improvements": [],
                "regressions": [],
            }

        # Identify improvements and regressions
        improvements = []
        regressions = []

        for i in range(1, len(timeline)):
            prev = timeline[i - 1]
            curr = timeline[i]
            diff = curr["mean_score"] - prev["mean_score"]

            if diff > 0.05:
                improvements.append(
                    {"from": prev["timestamp"], "to": curr["timestamp"], "improvement": diff}
                )
            elif diff < -0.05:
                regressions.append(
                    {"from": prev["timestamp"], "to": curr["timestamp"], "regression": abs(diff)}
                )

        # Overall trend
        first_score = timeline[0]["mean_score"]
        last_score = timeline[-1]["mean_score"]
        overall_diff = last_score - first_score

        if overall_diff > 0.05:
            trend = "Improving"
        elif overall_diff < -0.05:
            trend = "Declining"
        else:
            trend = "Stable"

        return {
            "timeline": timeline,
            "trend": trend,
            "improvements": improvements,
            "regressions": regressions,
            "overall_change": overall_diff,
        }

    def generate_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations from analysis results.

        Args:
            analysis: Analysis dictionary from other tools

        Returns:
            List of recommendation strings prioritized by impact
        """
        insights = []

        # Check for score distribution
        if "mean_score" in analysis:
            score = analysis["mean_score"]

            if score >= 0.90:
                insights.append(
                    f"✅ Excellent robustness (score: {score:.3f}). "
                    f"The system handles prompt variations very well."
                )
            elif score >= 0.80:
                insights.append(
                    f"✅ Good robustness (score: {score:.3f}). "
                    f"Minor inconsistencies but acceptable for production."
                )
            elif score >= 0.70:
                insights.append(
                    f"⚠️  Acceptable robustness (score: {score:.3f}). "
                    f"Room for improvement. Monitor closely."
                )
            else:
                insights.append(
                    f"❌ Concerning robustness (score: {score:.3f}). "
                    f"Significant inconsistencies detected."
                )

        # Check for high standard deviation
        if "std_score" in analysis and analysis["std_score"] > 0.15:
            insights.append(
                f"⚠️  High score variability (std: {analysis['std_score']:.3f}). "
                f"Results are inconsistent across prompt variations."
            )

        # Check for trend
        if "trend" in analysis:
            trend = analysis["trend"]
            if trend == "Regression":
                insights.append(
                    "❌ Regression detected. Recent changes decreased robustness. "
                    "Review recent agent instruction or tool modifications."
                )
            elif trend == "Improvement":
                insights.append("✅ Improvement detected. Recent changes increased robustness.")

        # Check for component-specific issues (if available)
        if "component_scores" in analysis:
            comp = analysis["component_scores"]
            if comp.get("data_similarity", 1) < 0.80:
                insights.append(
                    "⚠️  Data consistency below threshold. "
                    "Check data fetching and processing logic."
                )
            if comp.get("semantic_similarity", 1) < 0.75:
                insights.append(
                    "⚠️  Low semantic similarity in responses. "
                    "Review agent instructions for clarity."
                )
            if comp.get("process_consistency", 1) < 0.80:
                insights.append(
                    "⚠️  Process inconsistency detected. " "Tool call sequences vary across prompts."
                )

        # Generic recommendations
        if not insights:
            insights.append("No specific issues identified. Continue monitoring.")

        return insights

    def export_analysis_report(
        self, analysis: Dict[str, Any], format: str = "markdown", output_path: Optional[str] = None
    ) -> str:
        """
        Export analysis as a formatted report.

        Args:
            analysis: Analysis dictionary
            format: Output format ('markdown', 'json', 'csv')
            output_path: Optional path to save the report (S3 or local)

        Returns:
            Path to the exported report or report content as string
        """
        if format == "json":
            content = json.dumps(analysis, indent=2)
            ext = "json"
        elif format == "csv":
            # Convert to DataFrame if possible
            if "runs" in analysis:
                df = pd.DataFrame(analysis["runs"])
                content = df.to_csv(index=False)
            else:
                content = "No tabular data available"
            ext = "csv"
        else:  # markdown (default)
            content = self._format_markdown_report(analysis)
            ext = "md"

        # Save if path provided
        if output_path:
            if not output_path.endswith(f".{ext}"):
                output_path = f"{output_path}.{ext}"

            try:
                with S3.open(output_path, "w") as f:
                    f.write(content)
                logger.info(f"✅ Report exported to {S3.path(output_path)}")
                return S3.path(output_path)
            except Exception as e:
                logger.error(f"Failed to export report: {e}")
                return content

        return content

    def _get_rating(self, score: float) -> str:
        """Get human-readable rating for a score."""
        if score >= RATING_THRESHOLDS["excellent"]:
            return "Excellent"
        elif score >= RATING_THRESHOLDS["good"]:
            return "Good"
        elif score >= RATING_THRESHOLDS["acceptable"]:
            return "Acceptable"
        else:
            return "Concerning"

    def _format_markdown_report(self, analysis: Dict[str, Any]) -> str:
        """Format analysis as markdown report."""
        lines = [
            "# Robustness Analysis Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        ]

        # Score distribution
        if "mean_score" in analysis:
            lines.append("## Overall Score")
            lines.append(f"- **Mean Score:** {analysis['mean_score']:.3f}")
            lines.append(f"- **Median Score:** {analysis.get('median_score', 0):.3f}")
            lines.append(f"- **Std Dev:** {analysis.get('std_score', 0):.3f}")
            lines.append(f"- **Rating:** {analysis.get('rating', 'Unknown')}")
            lines.append("")

        # Insights
        if "insights" in analysis:
            lines.append("## Insights")
            for insight in analysis["insights"]:
                lines.append(f"- {insight}")
            lines.append("")

        # Trend
        if "trend" in analysis:
            lines.append("## Trend Analysis")
            lines.append(f"- **Overall Trend:** {analysis['trend']}")
            if "overall_change" in analysis:
                lines.append(f"- **Change:** {analysis['overall_change']:+.3f}")
            lines.append("")

        # Comparison
        if "runs" in analysis:
            lines.append("## Test Runs Comparison")
            lines.append(
                "| Timestamp | Tests | Passed | Failed | Success Rate | Mean Score | Rating |"
            )
            lines.append(
                "|-----------|-------|--------|--------|--------------|------------|--------|"
            )
            for run in analysis["runs"]:
                lines.append(
                    f"| {run['timestamp']} | {run['total_tests']} | "
                    f"{run['passed']} | {run['failed']} | "
                    f"{run['success_rate']:.1%} | {run['mean_score']:.3f} | "
                    f"{run['rating']} |"
                )
            lines.append("")

        return "\n".join(lines)
