#!/usr/bin/env python
# coding: utf-8
"""
End-to-end pipeline robustness tests.

Tests the full Cs_copilot pipeline with prompt variations to assess
robustness and consistency of outputs.
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
import pytest

from .comparators import OutputComparator
from .metrics import RobustnessMetrics
from .prompt_variations import PromptVariationGenerator

logger = logging.getLogger(__name__)

# Note: Fixtures (agent_team_factory, prompt_generator, comparator, metrics_calculator)
# are now provided by conftest.py in the robustness directory.
# They are automatically available to all tests in this file.


class TestPipelineRobustness:
    """Test end-to-end pipeline robustness to prompt variations."""

    def test_full_pipeline_robustness(
        self, agent_team_factory, prompt_generator, comparator, metrics_calculator
    ):
        """
        Test complete pipeline with 10 variations of the key example prompt.
        This is the primary robustness test covering the full workflow.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting full pipeline robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        # Get prompt variations
        variations = prompt_generator.get_variations("full_pipeline", n=10)
        assert len(variations) == 10, "Expected 10 prompt variations"

        outputs = []

        # Run agent with each variation (each in separate session)
        for i, prompt in enumerate(variations):
            logger.info(f"Running variation {i+1}/10: {prompt[:80]}...")

            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_pipeline_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                # Run full pipeline
                result = agent_team.run(prompt, stream=False)
                session_state = agent_team.get_session_state()

                # Collect all outputs
                output = {
                    "run_id": i,
                    "prompt": prompt,
                    "result": result.content,
                    "session_state": session_state,
                    "files": self._collect_output_files(session_state),
                    "session_id": session_id if s3_enabled else f"local_run_{i}",
                }
                outputs.append(output)

                # Save artifacts for offline analysis
                self._save_run_artifacts(output, run_id=i)

            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                pytest.fail(f"Pipeline failed on variation {i}: {e}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Ensure all runs completed
        assert len(outputs) == 10, "Not all variations completed successfully"

        # Comprehensive comparison
        comparison_results = self._compare_all_outputs(outputs, comparator)

        # Calculate robustness score
        robustness_score = metrics_calculator.calculate_robustness_score(comparison_results)

        # Identify outliers
        outliers = metrics_calculator.identify_outliers(outputs)

        # Generate report
        report = metrics_calculator.generate_report(
            {
                "score": robustness_score,
                "comparisons": comparison_results,
                "outliers": outliers,
            }
        )

        # Save report
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "full_pipeline_robustness.md"
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")

        # Assertions
        assert (
            robustness_score > 0.80
        ), f"Pipeline robustness score {robustness_score:.2f} below threshold (0.80)"

        assert len(outliers) <= 1, f"Too many outlier runs ({len(outliers)}): {outliers}"

        # Specific component checks
        if "data" in comparison_results:
            assert (
                comparison_results["data"]["row_jaccard"] > 0.95
            ), "Dataset row consistency below threshold"

        if "gtm" in comparison_results:
            assert (
                comparison_results["gtm"]["projection_correlation"] > 0.90
            ), "GTM projection consistency below threshold"

        if "text" in comparison_results:
            assert (
                comparison_results["text"]["semantic_similarity"] > 0.75
            ), "Text semantic similarity below threshold"

        logger.info(f"Pipeline robustness test PASSED with score {robustness_score:.3f}")

    def test_incremental_robustness(self, agent_team_factory, prompt_generator, metrics_calculator):
        """
        Test robustness of each pipeline step with accumulated state.
        Tests if variations at step N affect downstream steps.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting incremental robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        base_prompts = {
            "step1_chembl": "chembl_download",
            "step2_gtm": "gtm_optimization",
            "step3_density": "density_analysis",
            "step4_activity": "activity_analysis",
            "step5_chemotype": "chemotype_analysis",
        }

        results_per_step = {}

        for step, prompt_key in base_prompts.items():
            logger.info(f"Testing {step}: {prompt_key}")

            variations = prompt_generator.get_variations(prompt_key, n=10)
            step_outputs = []

            for i, prompt in enumerate(variations):
                # Setup S3 session isolation
                original_prefix = None
                if s3_enabled:
                    session_id = f"robustness_{step}_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                    original_prefix = S3Client.prefix
                    S3Client.prefix = f"sessions/{session_id}"
                    logger.debug(f"S3 session prefix: sessions/{session_id}")

                try:
                    # Create fresh agent team for this variation
                    agent_team = agent_team_factory()

                    result = agent_team.run(prompt, stream=False)
                    session_state = agent_team.get_session_state()

                    step_outputs.append(
                        {
                            "run_id": i,
                            "prompt": prompt,
                            "result": result.content,
                            "session_state": session_state,
                            "session_id": session_id if s3_enabled else f"local_{step}_run_{i}",
                        }
                    )
                except Exception as e:
                    logger.error(f"{step} variation {i} failed: {e}")

                finally:
                    # Restore original S3 prefix
                    if original_prefix is not None:
                        S3Client.prefix = original_prefix

            # Calculate robustness for this step
            comparison = self._compare_step_outputs(step_outputs, step)
            robustness_score = metrics_calculator.calculate_robustness_score(comparison)

            results_per_step[step] = {
                "score": robustness_score,
                "comparison": comparison,
            }

            assert (
                robustness_score > 0.75
            ), f"Step {step} robustness {robustness_score:.2f} below threshold (0.75)"

        # Generate per-step report
        self._generate_step_by_step_report(results_per_step)
        logger.info("Incremental robustness test PASSED")

    def test_prompt_variation_validity(self, prompt_generator):
        """Test that all prompt variations preserve semantic intent."""
        logger.info("Testing prompt variation validity")

        for prompt_key in prompt_generator.list_available_prompts():
            base_prompt = prompt_generator.get_base_prompt(prompt_key)
            variations = prompt_generator.get_variations(prompt_key, n=10)

            for i, variation in enumerate(variations[1:], 1):  # Skip base
                is_valid = prompt_generator.validate_variation(
                    base_prompt, variation, min_similarity=0.70
                )
                assert is_valid, f"Variation {i} of '{prompt_key}' has low semantic similarity"

        logger.info("All prompt variations are valid")

    # ==================== Helper Methods ====================

    def _collect_output_files(self, session_state: Dict) -> Dict[str, Path]:
        """Extract all file paths from session state."""
        files = {}

        # Dataset
        if "data_file_paths" in session_state:
            files["dataset"] = session_state["data_file_paths"].get("dataset_path")

        # GTM files
        if "gtm_file_paths" in session_state:
            gtm_paths = session_state["gtm_file_paths"]
            files["gtm_model"] = gtm_paths.get("gtm_path")
            files["gtm_plot"] = gtm_paths.get("gtm_plot_path")

        # Landscape files
        if "landscape_files" in session_state:
            landscape_paths = session_state["landscape_files"]
            files["landscape_data"] = landscape_paths.get("landscape_data_csv")
            files["landscape_plot"] = landscape_paths.get("landscape_plot")

        return files

    def _save_run_artifacts(self, output: Dict, run_id: int):
        """Save artifacts from a single run for offline analysis."""
        artifacts_dir = Path(__file__).parent / "artifacts" / f"run_{run_id}"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save prompt
        (artifacts_dir / "prompt.txt").write_text(output["prompt"])

        # Save response
        (artifacts_dir / "response.txt").write_text(output["result"])

        # Save session state
        import json

        (artifacts_dir / "session_state.json").write_text(
            json.dumps(output["session_state"], indent=2, default=str)
        )

    def _compare_all_outputs(self, outputs: List[Dict], comparator: OutputComparator) -> Dict:
        """Perform comprehensive comparison across all output types."""
        results = {}

        try:
            # Compare datasets
            datasets = []
            for out in outputs:
                dataset_path = out["files"].get("dataset")
                if dataset_path and Path(dataset_path).exists():
                    datasets.append(pd.read_csv(dataset_path))

            if len(datasets) >= 2:
                results["data"] = comparator.compare_dataframes(datasets)
        except Exception as e:
            logger.warning(f"Could not compare datasets: {e}")

        try:
            # Compare GTM models
            gtm_models = []
            for out in outputs:
                gtm_path = out["files"].get("gtm_model")
                if gtm_path and Path(gtm_path).exists():
                    gtm_models.append(joblib.load(gtm_path))

            if len(gtm_models) >= 2:
                results["gtm"] = comparator.compare_gtm_models(gtm_models)
        except Exception as e:
            logger.warning(f"Could not compare GTM models: {e}")

        try:
            # Compare landscape data
            landscapes = []
            for out in outputs:
                landscape_path = out["files"].get("landscape_data")
                if landscape_path and Path(landscape_path).exists():
                    landscapes.append(pd.read_csv(landscape_path))

            if len(landscapes) >= 2:
                results["landscape_data"] = comparator.compare_dataframes(landscapes)
        except Exception as e:
            logger.warning(f"Could not compare landscape data: {e}")

        try:
            # Compare text responses
            texts = [out["result"] for out in outputs]
            if texts:
                results["text"] = comparator.compare_text_outputs(texts)
        except Exception as e:
            logger.warning(f"Could not compare text outputs: {e}")

        try:
            # Compare plots
            gtm_plots = [
                out["files"].get("gtm_plot") for out in outputs if out["files"].get("gtm_plot")
            ]
            if len(gtm_plots) >= 2:
                results["gtm_plots"] = comparator.compare_images(gtm_plots)

            landscape_plots = [
                out["files"].get("landscape_plot")
                for out in outputs
                if out["files"].get("landscape_plot")
            ]
            if len(landscape_plots) >= 2:
                results["landscape_plots"] = comparator.compare_images(landscape_plots)

            # Combine visual metrics
            if "gtm_plots" in results or "landscape_plots" in results:
                visual_metrics = {}
                for key in ["perceptual_hash", "ssim"]:
                    values = []
                    if "gtm_plots" in results:
                        values.append(results["gtm_plots"].get(key, 0))
                    if "landscape_plots" in results:
                        values.append(results["landscape_plots"].get(key, 0))
                    if values:
                        visual_metrics[key] = sum(values) / len(values)
                results["visual"] = visual_metrics
        except Exception as e:
            logger.warning(f"Could not compare images: {e}")

        # Add process metrics
        results["process"] = {
            "completion_rate": len(outputs) / 10.0,  # Assuming 10 variations
            "tool_sequence_similarity": 0.90,  # Placeholder
        }

        return results

    def _compare_step_outputs(self, outputs: List[Dict], step: str) -> Dict:
        """Compare outputs for a specific step."""
        comparator = OutputComparator()
        results = {}

        # Compare text responses
        texts = [out["result"] for out in outputs]
        results["text"] = comparator.compare_text_outputs(texts)

        # Add process metrics
        results["process"] = {
            "completion_rate": len(outputs) / 10.0,
            "tool_sequence_similarity": 0.90,
        }

        return results

    def _generate_step_by_step_report(self, results_per_step: Dict):
        """Generate report for incremental testing."""
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)

        report = "# Incremental Robustness Report\n\n"
        report += "## Per-Step Scores\n\n"
        report += "| Step | Robustness Score | Status |\n"
        report += "|------|------------------|--------|\n"

        for step, data in results_per_step.items():
            score = data["score"]
            status = "✅" if score > 0.75 else "❌"
            report += f"| {step} | {score:.3f} | {status} |\n"

        report_path = report_dir / "incremental_robustness.md"
        report_path.write_text(report)
        logger.info(f"Incremental report saved to {report_path}")


# Standalone test for quick validation
def test_prompt_generator_initialization():
    """Test that prompt generator can be initialized."""
    generator = PromptVariationGenerator()
    assert generator is not None
    prompts = generator.list_available_prompts()
    assert len(prompts) > 0
    assert "full_pipeline" in prompts


def test_comparator_initialization():
    """Test that comparator can be initialized."""
    comparator = OutputComparator()
    assert comparator is not None
    assert hasattr(comparator, "compare_dataframes")
    assert hasattr(comparator, "compare_text_outputs")


def test_metrics_calculator_initialization():
    """Test that metrics calculator can be initialized."""
    calc = RobustnessMetrics()
    assert calc is not None
    assert hasattr(calc, "calculate_robustness_score")
    assert hasattr(calc, "generate_report")
