"""MLflow-based reporting and querying utilities for robustness tests."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class MLflowRobustnessReporter:
    """Query and report on robustness test runs stored in MLflow."""

    def __init__(self, experiment_name: str = "robustness_testing"):
        """Initialize reporter.

        Args:
            experiment_name: Name of the MLflow experiment to query
        """
        self.experiment_name = experiment_name
        self._client = None
        self._experiment = None

        self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow client."""
        try:
            from mlflow.tracking import MlflowClient

            self._client = MlflowClient()

            # Get or create experiment
            try:
                self._experiment = self._client.get_experiment_by_name(self.experiment_name)
            except Exception:
                logger.warning(f"Experiment '{self.experiment_name}' not found")
                self._experiment = None

        except ImportError:
            logger.error("MLflow not installed. Reporter requires MLflow.")
            raise

    def get_recent_runs(self, days: int = 7, max_results: int = 100) -> List[Dict[str, Any]]:
        """Get recent robustness test suite runs.

        Args:
            days: Number of days to look back
            max_results: Maximum number of runs to return

        Returns:
            List of run information dictionaries
        """
        if not self._experiment:
            return []

        # Search for suite-level runs (non-nested runs)
        filter_string = "tags.suite_id != ''"
        runs = self._client.search_runs(
            experiment_ids=[self._experiment.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=["start_time DESC"],
        )

        # Filter by date
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_runs = []

        for run in runs:
            start_time = datetime.fromtimestamp(run.info.start_time / 1000)
            if start_time >= cutoff_date:
                recent_runs.append(self._format_run_info(run))

        return recent_runs

    def get_test_history(self, test_name: str, limit: int = 50) -> pd.DataFrame:
        """Get historical performance of a specific test.

        Args:
            test_name: Name of the test to query
            limit: Maximum number of runs to retrieve

        Returns:
            DataFrame with test history
        """
        if not self._experiment:
            return pd.DataFrame()

        # Search for test-level runs
        filter_string = f"tags.test_name = '{test_name}'"
        runs = self._client.search_runs(
            experiment_ids=[self._experiment.experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"],
        )

        # Extract metrics and metadata
        history_data = []
        for run in runs:
            data = {
                "run_id": run.info.run_id,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "duration_seconds": (
                    (run.info.end_time - run.info.start_time) / 1000 if run.info.end_time else None
                ),
            }

            # Add metrics
            for key, value in run.data.metrics.items():
                data[key] = value

            # Add relevant params
            for key in ["prompt_key", "description"]:
                if key in run.data.params:
                    data[key] = run.data.params[key]

            history_data.append(data)

        return pd.DataFrame(history_data)

    def compare_runs(self, run_id1: str, run_id2: str) -> Dict[str, Any]:
        """Compare two robustness test suite runs.

        Args:
            run_id1: First run ID
            run_id2: Second run ID

        Returns:
            Comparison results dictionary
        """
        run1 = self._client.get_run(run_id1)
        run2 = self._client.get_run(run_id2)

        comparison = {
            "run1": self._format_run_info(run1),
            "run2": self._format_run_info(run2),
            "metric_diffs": {},
        }

        # Compare metrics
        metrics1 = run1.data.metrics
        metrics2 = run2.data.metrics

        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        for metric in common_metrics:
            diff = metrics2[metric] - metrics1[metric]
            comparison["metric_diffs"][metric] = {
                "run1": metrics1[metric],
                "run2": metrics2[metric],
                "diff": diff,
                "percent_change": (diff / metrics1[metric] * 100) if metrics1[metric] != 0 else 0,
            }

        return comparison

    def generate_trend_report(
        self, test_name: Optional[str] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Generate trend analysis report for robustness tests.

        Args:
            test_name: Specific test to analyze (None for all tests)
            days: Number of days to analyze

        Returns:
            Trend analysis report
        """
        if test_name:
            history = self.get_test_history(test_name, limit=1000)
            history = history[history["start_time"] >= (datetime.now() - timedelta(days=days))]
        else:
            # Get all suite runs
            runs = self.get_recent_runs(days=days, max_results=1000)
            history = pd.DataFrame(runs)

        if history.empty:
            return {"error": "No data available for trend analysis"}

        report = {
            "period_days": days,
            "total_runs": len(history),
            "date_range": {
                "start": history["start_time"].min().isoformat() if not history.empty else None,
                "end": history["start_time"].max().isoformat() if not history.empty else None,
            },
        }

        # Calculate trends for key metrics
        metric_columns = [
            col
            for col in history.columns
            if col
            in [
                "robustness_score",
                "pass_rate",
                "data_similarity",
                "semantic_similarity",
            ]
        ]

        trends = {}
        for metric in metric_columns:
            if metric in history.columns:
                values = history[metric].dropna()
                if len(values) > 0:
                    trends[metric] = {
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "latest": float(values.iloc[0]) if len(values) > 0 else None,
                        "trend": self._calculate_trend(values),
                    }

        report["metric_trends"] = trends

        return report

    def get_failing_tests(self, threshold: float = 0.75) -> List[Dict[str, Any]]:
        """Get tests that are consistently failing or below threshold.

        Args:
            threshold: Robustness score threshold for passing

        Returns:
            List of failing test information
        """
        if not self._experiment:
            return []

        # Search for test runs with low robustness scores
        filter_string = f"metrics.robustness_score < {threshold}"
        runs = self._client.search_runs(
            experiment_ids=[self._experiment.experiment_id],
            filter_string=filter_string,
            max_results=100,
            order_by=["start_time DESC"],
        )

        failing_tests = []
        for run in runs:
            test_name = run.data.tags.get("test_name")
            if test_name:
                failing_tests.append(
                    {
                        "test_name": test_name,
                        "run_id": run.info.run_id,
                        "robustness_score": run.data.metrics.get("robustness_score"),
                        "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                    }
                )

        return failing_tests

    def _format_run_info(self, run) -> Dict[str, Any]:
        """Format run information into a dictionary.

        Args:
            run: MLflow Run object

        Returns:
            Formatted run information
        """
        return {
            "run_id": run.info.run_id,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
            "end_time": (
                datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None
            ),
            "status": run.info.status,
            "metrics": dict(run.data.metrics),
            "params": dict(run.data.params),
            "tags": dict(run.data.tags),
        }

    def _calculate_trend(self, values: pd.Series) -> str:
        """Calculate trend direction from time series.

        Args:
            values: Time series of values (newest first)

        Returns:
            Trend description: "improving", "declining", or "stable"
        """
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression
        recent = values.iloc[: len(values) // 2].mean()
        older = values.iloc[len(values) // 2 :].mean()

        diff = recent - older
        percent_change = abs(diff / older * 100) if older != 0 else 0

        if percent_change < 5:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"


def generate_robustness_dashboard(
    experiment_name: str = "robustness_testing", days: int = 30
) -> Dict[str, Any]:
    """Generate a comprehensive robustness dashboard.

    Args:
        experiment_name: MLflow experiment name
        days: Number of days to analyze

    Returns:
        Dashboard data dictionary
    """
    reporter = MLflowRobustnessReporter(experiment_name)

    dashboard = {
        "generated_at": datetime.now().isoformat(),
        "period_days": days,
        "recent_runs": reporter.get_recent_runs(days=days, max_results=20),
        "trend_report": reporter.generate_trend_report(days=days),
        "failing_tests": reporter.get_failing_tests(),
    }

    return dashboard
