"""MLflow-enhanced robustness test runner."""

import logging
from typing import Any, Dict, Optional

from robustness_minimal_example import RobustnessConfig, RobustnessRunner

logger = logging.getLogger(__name__)


class MLflowRobustnessRunner(RobustnessRunner):
    """Robustness test runner with MLflow tracking integration.

    This class extends RobustnessRunner to automatically log all test executions,
    metrics, and artifacts to MLflow, creating a hierarchical run structure:

    Suite Run (root)
    └── Test Run (nested)
        └── Variation Run (nested)
            └── Agent Run (nested, created by agent tracking)
                └── Tool Call Run (nested, created by tool tracking)
    """

    def __init__(
        self,
        config: RobustnessConfig,
        experiment_name: str = "robustness_testing",
        enable_mlflow: bool = True,
    ):
        """Initialize MLflow robustness runner.

        Args:
            config: Robustness test configuration
            experiment_name: MLflow experiment name
            enable_mlflow: Whether to enable MLflow tracking
        """
        super().__init__(config)

        self.enable_mlflow = enable_mlflow
        self.experiment_name = experiment_name
        self._tracker = None
        self._suite_run = None

        if self.enable_mlflow:
            self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow tracker."""
        try:
            from cs_copilot.tracking import get_tracker

            self._tracker = get_tracker()

            if not self._tracker.is_enabled():
                logger.warning(
                    "MLflow tracking is disabled. Tests will run without MLflow logging."
                )
                self.enable_mlflow = False
            else:
                logger.info(f"MLflow tracking enabled for experiment: {self.experiment_name}")

                # Set experiment
                import mlflow

                mlflow.set_experiment(self.experiment_name)

        except ImportError:
            logger.warning("MLflow tracking module not available. Running without tracking.")
            self.enable_mlflow = False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all enabled tests with MLflow suite-level tracking.

        Returns:
            Dictionary of test results
        """
        if not self.enable_mlflow:
            return super().run_all_tests()

        # Create suite-level run
        import mlflow

        suite_run_name = f"robustness_suite_{self.test_run_id}"

        with mlflow.start_run(run_name=suite_run_name) as suite_run:
            self._suite_run = suite_run

            # Log suite configuration
            mlflow.log_params(
                {
                    "n_variations": self.config.n_variations,
                    "model_provider": self.config.model_provider,
                    "model_id": self.config.model_id,
                    "debug_mode": self.config.debug_mode,
                    "s3_session_isolation": self.config.s3_session_isolation,
                    "pass_threshold": self.config.pass_threshold,
                }
            )

            # Log enabled tests
            enabled_tests = [name for name, test in self.config.tests.items() if test.enabled]
            mlflow.set_tags(
                {
                    "suite_id": self.test_run_id,
                    "test_count": len(enabled_tests),
                    "enabled_tests": ",".join(enabled_tests),
                }
            )

            # Run tests with parent tracking
            results = super().run_all_tests()

            # Log suite-level metrics
            self._log_suite_metrics(results)

            # Log suite artifacts
            if self.config.save_artifacts:
                self._log_suite_artifacts()

            return results

    def run_test(self, test_config: Any) -> Dict[str, Any]:
        """Run a single test with MLflow test-level tracking.

        Args:
            test_config: Test configuration

        Returns:
            Test results dictionary
        """
        if not self.enable_mlflow:
            return super().run_test(test_config)

        import mlflow

        test_name = test_config.name
        test_run_name = f"test_{test_name}"

        with mlflow.start_run(run_name=test_run_name, nested=True):
            # Log test configuration
            mlflow.log_params(
                {
                    "test_name": test_name,
                    "prompt_key": test_config.prompt_key,
                    "description": test_config.description[:200] if test_config.description else "",
                }
            )

            mlflow.set_tags({"test_name": test_name})

            # Run the test
            result = super().run_test(test_config)

            # Log test-level metrics
            if result:
                self._log_test_metrics(test_name, result)

            return result

    def _run_single_variation(
        self, prompt: str, test_name: str, run_id: int, s3_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a single prompt variation with MLflow variation-level tracking.

        Args:
            prompt: The prompt to test
            test_name: Name of the test
            run_id: Index of the variation (0-based)
            s3_prefix: Optional S3 prefix for session isolation

        Returns:
            Variation results dictionary
        """
        if not self.enable_mlflow:
            return super()._run_single_variation(prompt, test_name, run_id, s3_prefix)

        import mlflow

        variation_run_name = f"variation_{run_id}"

        with mlflow.start_run(run_name=variation_run_name, nested=True):
            # Log variation details
            mlflow.log_params(
                {
                    "variation_idx": run_id,
                    "prompt_preview": prompt[:200] if prompt else "",
                }
            )

            mlflow.set_tags(
                {
                    "test_name": test_name,
                    "variation_idx": run_id,
                }
            )

            # Log full prompt as artifact
            mlflow.log_text(prompt, f"prompt_variation_{run_id}.txt")

            # Run the variation
            result = super()._run_single_variation(prompt, test_name, run_id, s3_prefix)

            # Log variation-level metrics
            if result:
                self._log_variation_metrics(result)

                # Log response as artifact
                if "response" in result:
                    mlflow.log_text(str(result["response"]), f"response_variation_{run_id}.txt")

            return result

    def _log_suite_metrics(self, results: Dict[str, Any]):
        """Log suite-level aggregated metrics.

        Args:
            results: All test results
        """
        import mlflow

        total_tests = len(results)

        # Handle both successful results and error results
        passed_tests = sum(
            1 for r in results.values() if isinstance(r, dict) and r.get("passed", False)
        )
        failed_tests = total_tests - passed_tests

        # Calculate average robustness score only for successful tests
        robustness_scores = [
            r.get("robustness_score", 0)
            for r in results.values()
            if isinstance(r, dict) and "robustness_score" in r
        ]
        avg_robustness_score = (
            sum(robustness_scores) / len(robustness_scores) if robustness_scores else 0.0
        )

        mlflow.log_metrics(
            {
                "total_tests": float(total_tests),
                "passed_tests": float(passed_tests),
                "failed_tests": float(failed_tests),
                "pass_rate": float(passed_tests / total_tests if total_tests > 0 else 0),
                "avg_robustness_score": avg_robustness_score,
            }
        )

    def _log_test_metrics(self, test_name: str, result: Dict[str, Any]):
        """Log test-level metrics.

        Args:
            test_name: Name of the test
            result: Test results
        """
        import mlflow

        metrics = {}

        if "robustness_score" in result:
            metrics["robustness_score"] = float(result["robustness_score"])

        if "comparison" in result:
            comparison = result["comparison"]
            if "data_similarity" in comparison:
                metrics["data_similarity"] = float(comparison["data_similarity"])
            if "semantic_similarity" in comparison:
                metrics["semantic_similarity"] = float(comparison["semantic_similarity"])
            if "process_consistency" in comparison:
                metrics["process_consistency"] = float(comparison["process_consistency"])
            if "visual_similarity" in comparison:
                metrics["visual_similarity"] = float(comparison["visual_similarity"])

        if "passed" in result:
            metrics["passed"] = 1.0 if result["passed"] else 0.0

        if "execution_time" in result:
            metrics["execution_time_seconds"] = float(result["execution_time"])

        if metrics:
            mlflow.log_metrics(metrics)

    def _log_variation_metrics(self, result: Dict[str, Any]):
        """Log variation-level metrics.

        Args:
            result: Variation results
        """
        import mlflow

        metrics = {}

        if "success" in result:
            metrics["success"] = 1.0 if result["success"] else 0.0

        if "execution_time" in result:
            metrics["execution_time_seconds"] = float(result["execution_time"])

        if "error" in result and result["error"]:
            mlflow.log_param("error_message", str(result["error"])[:500])

        # Log output characteristics
        if "output" in result:
            output = result["output"]
            if isinstance(output, str):
                metrics["output_length"] = float(len(output))
            elif isinstance(output, dict):
                metrics["output_keys_count"] = float(len(output))

        if metrics:
            mlflow.log_metrics(metrics)

    def _log_suite_artifacts(self):
        """Log suite-level artifacts (reports, summaries, etc.)."""
        import mlflow

        # Log summary report if it exists
        summary_file = self.output_dir / "summary.json"
        if summary_file.exists():
            mlflow.log_artifact(str(summary_file), "reports")

        # Log markdown report if enabled
        if self.config.generate_markdown:
            report_file = self.output_dir / "robustness_report.md"
            if report_file.exists():
                mlflow.log_artifact(str(report_file), "reports")

        # Log configuration
        config_file = self.output_dir / "config.yaml"
        if config_file.exists():
            mlflow.log_artifact(str(config_file), "config")
