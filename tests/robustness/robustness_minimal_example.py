#!/usr/bin/env python
"""
Robustness Testing Runner for Cs_copilot Agentic Operations.

This script provides a configurable robustness testing framework that:
- Runs selected tests based on a YAML configuration file
- Tests robustness of agent operations to prompt variations
- Generates detailed comparison metrics and reports
- Supports S3 session isolation for reproducible testing

Usage:
    uv run python tests/robustness/robustness_minimal_example.py
    uv run python tests/robustness/robustness_minimal_example.py --config custom_config.yaml
    uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3
"""

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from dotenv import load_dotenv  # noqa: E402

from cs_copilot.utils.logging import get_logger  # noqa: E402

# Import shared test utilities
sys.path.insert(0, str(Path(__file__).parent))
from config_schema import ConfigValidator  # noqa: E402
from test_utils import ResponseParser, S3SessionManager  # noqa: E402
from tool_tracker import ToolSequenceComparator  # noqa: E402

logger = get_logger(__name__)
load_dotenv()


@dataclass
class TestConfig:
    """Configuration for a single test."""

    name: str
    enabled: bool
    prompt_key: str
    description: str = ""
    depends_on: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    custom_prompt: Optional[str] = None


@dataclass
class RobustnessConfig:
    """Full robustness testing configuration."""

    n_variations: int = 5
    debug_mode: bool = False
    output_dir: str = "reports"
    save_artifacts: bool = True
    s3_session_isolation: bool = True

    # Model settings
    model_provider: str = "deepseek"
    model_id: str = "deepseek-chat"
    api_key_env: str = "DEEPSEEK_API_KEY"

    # Metrics settings
    weights: Dict[str, float] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    pass_threshold: float = 0.75

    # Tests to run
    tests: Dict[str, TestConfig] = field(default_factory=dict)

    # Reporting
    generate_markdown: bool = True
    generate_json: bool = True
    include_run_details: bool = True
    include_recommendations: bool = True


def load_config(config_path: Path) -> RobustnessConfig:
    """
    Load and validate configuration from YAML file.

    Performs comprehensive validation including:
    - Schema validation (all required fields present)
    - Type validation (correct data types)
    - Range validation (values within acceptable ranges)
    - Dependency validation (no circular dependencies)

    Args:
        config_path: Path to robustness_config.yaml

    Returns:
        Validated RobustnessConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Validate configuration before loading
    try:
        data = ConfigValidator.load_and_validate(config_path)
        logger.info("Configuration validation passed ✓")
    except ValueError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        raise

    general = data.get("general", {})
    model = data.get("model", {})
    metrics = data.get("metrics", {})
    reporting = data.get("reporting", {})

    # Parse tests
    tests = {}
    for test_name, test_data in data.get("tests", {}).items():
        if test_data:
            tests[test_name] = TestConfig(
                name=test_name,
                enabled=test_data.get("enabled", False),
                prompt_key=test_data.get("prompt_key", test_name),
                description=test_data.get("description", ""),
                depends_on=test_data.get("depends_on", []),
                params=test_data.get("params", {}),
            )

    # Parse custom tests
    custom_tests = data.get("custom_tests") or {}
    for test_name, test_data in custom_tests.items():
        if test_data and test_data.get("enabled", False):
            tests[test_name] = TestConfig(
                name=test_name,
                enabled=True,
                prompt_key="",
                description=test_data.get("description", ""),
                custom_prompt=test_data.get("prompt", ""),
            )

    return RobustnessConfig(
        n_variations=general.get("n_variations", 5),
        debug_mode=general.get("debug_mode", False),
        output_dir=general.get("output_dir", "reports"),
        save_artifacts=general.get("save_artifacts", True),
        s3_session_isolation=general.get("s3_session_isolation", True),
        model_provider=model.get("provider", "deepseek"),
        model_id=model.get("model_id", "deepseek-chat"),
        api_key_env=model.get("api_key_env", "DEEPSEEK_API_KEY"),
        weights=metrics.get("weights", {}),
        thresholds=metrics.get("thresholds", {}),
        pass_threshold=metrics.get("pass_threshold", 0.75),
        tests=tests,
        generate_markdown=reporting.get("generate_markdown", True),
        generate_json=reporting.get("generate_json", True),
        include_run_details=reporting.get("include_run_details", True),
        include_recommendations=reporting.get("include_recommendations", True),
    )


class RobustnessRunner:
    """Run robustness tests based on configuration."""

    def __init__(self, config: RobustnessConfig):
        """Initialize runner with configuration."""
        self.config = config
        self.test_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: Dict[str, Dict] = {}

        # Setup output directory
        self.output_dir = Path(__file__).parent / config.output_dir / self.test_run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components lazily
        self._prompt_generator = None
        self._comparator = None
        self._metrics_calculator = None
        self._model = None
        self._s3_config = None

        # Use shared S3SessionManager for safe session isolation
        self._s3_session_manager = S3SessionManager()

    @property
    def prompt_generator(self):
        """Lazy load prompt variation generator."""
        if self._prompt_generator is None:
            from prompt_variations import PromptVariationGenerator

            self._prompt_generator = PromptVariationGenerator()
        return self._prompt_generator

    @property
    def comparator(self):
        """Lazy load output comparator."""
        if self._comparator is None:
            from comparators import OutputComparator

            self._comparator = OutputComparator()
        return self._comparator

    @property
    def metrics_calculator(self):
        """Lazy load metrics calculator."""
        if self._metrics_calculator is None:
            from metrics import RobustnessMetrics

            self._metrics_calculator = RobustnessMetrics(
                weights=self.config.weights if self.config.weights else None,
                thresholds=self.config.thresholds if self.config.thresholds else None,
            )
        return self._metrics_calculator

    def _get_model(self):
        """Get LLM model based on configuration."""
        if self._model is not None:
            return self._model

        if self.config.model_provider == "ollama":
            from agno.models.ollama import Ollama

            host = os.environ.get("OLLAMA_HOST")
            self._model = Ollama(id=self.config.model_id, host=host)
        else:
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                from getpass import getpass

                api_key = getpass(f"{self.config.api_key_env}: ")

            if self.config.model_provider == "deepseek":
                from agno.models.deepseek import DeepSeek

                self._model = DeepSeek(id=self.config.model_id, api_key=api_key)
            elif self.config.model_provider == "openai":
                from agno.models.openai import OpenAI

                self._model = OpenAI(id=self.config.model_id, api_key=api_key)
            elif self.config.model_provider == "anthropic":
                from agno.models.anthropic import Anthropic

                self._model = Anthropic(id=self.config.model_id, api_key=api_key)
            else:
                raise ValueError(f"Unknown model provider: {self.config.model_provider}")

        return self._model

    def _setup_s3(self):
        """Setup S3 configuration and check availability."""
        from cs_copilot.storage import get_s3_config, is_s3_enabled

        if not is_s3_enabled():
            if self.config.s3_session_isolation:
                raise RuntimeError(
                    "S3/MinIO must be enabled for robustness testing with session isolation. "
                    "Set USE_S3=true and provide endpoint, bucket, and credentials."
                )
            logger.warning("S3 not enabled - files will be stored locally")
            return None

        self._s3_config = get_s3_config()
        logger.info(f"S3 enabled - Bucket: {self._s3_config.bucket_name}")
        return self._s3_config

    def _set_s3_prefix(self, prefix: str):
        """
        Set S3 prefix for session isolation (deprecated - use S3SessionManager context manager).

        This method is kept for backward compatibility but should not be used directly.
        The run_test method now uses S3SessionManager.create_isolated_session() context manager.
        """
        from cs_copilot.storage.client import S3 as S3Client

        S3Client.prefix = prefix

    def _restore_s3_prefix(self):
        """
        Restore original S3 prefix (deprecated - use S3SessionManager).

        This method is kept for backward compatibility. S3SessionManager now handles
        restoration automatically in finally blocks via context managers.
        """
        self._s3_session_manager.restore()

    def _get_prompts(self, test_config: TestConfig) -> List[str]:
        """Get prompt variations for a test."""
        if test_config.custom_prompt:
            # For custom prompts, just use the single prompt
            return [test_config.custom_prompt]

        # Get variations from prompt generator
        variations = self.prompt_generator.get_variations(
            test_config.prompt_key, n=self.config.n_variations
        )

        # Handle interpolation and latent_exploration with molecule parameters
        if test_config.params:
            augmented_variations = []
            for var in variations:
                augmented = var
                if "molecule_a" in test_config.params and "molecule_b" in test_config.params:
                    augmented = (
                        f"{var} Molecule A: {test_config.params['molecule_a']}, "
                        f"Molecule B: {test_config.params['molecule_b']}"
                    )
                elif "seed_molecule" in test_config.params:
                    augmented = f"{var} Seed molecule: {test_config.params['seed_molecule']}"
                augmented_variations.append(augmented)
            return augmented_variations

        return variations

    def _extract_files_from_response(self, response_text: str) -> Set[str]:
        """Extract file paths from agent response text (wrapper for ResponseParser)."""
        return ResponseParser.extract_files(response_text)

    def _extract_smiles_from_response(self, response: str) -> List[str]:
        """Extract SMILES strings from agent response."""
        smiles = []

        # Pattern 1: Backtick enclosed
        backtick_pattern = r"`([A-Za-z0-9@+\-\[\]\(\)=#$]+)`"
        smiles.extend(re.findall(backtick_pattern, response))

        # Pattern 2: Lines starting with SMILES-like strings
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith(("#", "-", "*", ">")):
                if re.match(
                    r"^[A-Za-z0-9@+\-\[\]\(\)=#$]+$", line.split()[0] if line.split() else ""
                ):
                    smiles.append(line.split()[0])

        # Remove duplicates while preserving order
        seen = set()
        unique_smiles = []
        for s in smiles:
            if s not in seen and len(s) > 2:
                seen.add(s)
                unique_smiles.append(s)

        return unique_smiles

    def _run_single_variation(
        self, prompt: str, test_name: str, run_id: int, s3_prefix: Optional[str] = None
    ) -> Dict:
        """
        Run agent with a single prompt variation.

        Note: s3_prefix parameter is deprecated. S3 session isolation is now
        handled by the context manager in run_test().
        """
        from cs_copilot.agents import get_cs_copilot_agent_team
        from cs_copilot.storage import S3

        # S3 prefix is now handled by context manager in run_test()
        # No need to set it here

        session_id = f"robustness_{self.test_run_id}_{test_name}_run{run_id}_{uuid.uuid4().hex[:8]}"

        logger.info(f"Running {test_name} variation {run_id + 1}")
        logger.debug(f"Session ID: {session_id}")
        logger.debug(f"Prompt: {prompt[:100]}...")

        try:
            # Create fresh agent with memory disabled for complete isolation
            model = self._get_model()
            agent = get_cs_copilot_agent_team(
                model=model,
                debug_mode=self.config.debug_mode,
                show_members_responses=False,
                enable_memory=False,  # Disable memory for session isolation
            )

            # Run the agent
            result = agent.run(prompt, stream=False)
            session_state = agent.get_session_state()

            # Extract response content
            response_text = result.content if result.content else ""

            # Collect generated files
            generated_files = {}
            s3_files = {}

            # From response text
            files_from_response = self._extract_files_from_response(response_text)
            for filename in files_from_response:
                s3_url = S3.path(filename) if self._s3_config else filename
                generated_files[f"response:{filename}"] = s3_url
                s3_files[f"response:{filename}"] = s3_url

            # From session state
            for key, value in session_state.items():
                if isinstance(value, str) and value:
                    if value.startswith("s3://"):
                        s3_files[f"state:{key}"] = value
                        generated_files[f"state:{key}"] = value
                    elif not value.startswith(("http://", "https://", "/")) and "." in value:
                        s3_url = S3.path(value) if self._s3_config else value
                        s3_files[f"state:{key}"] = s3_url
                        generated_files[f"state:{key}"] = s3_url
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, str) and subvalue:
                            if subvalue.startswith("s3://"):
                                s3_files[f"state:{key}.{subkey}"] = subvalue
                                generated_files[f"state:{key}.{subkey}"] = subvalue

            # Extract SMILES if applicable
            smiles_generated = self._extract_smiles_from_response(response_text)

            return {
                "run_id": run_id,
                "prompt": prompt,
                "session_id": session_id,
                "response": response_text,
                "response_object": result,  # Store for tool sequence extraction
                "response_truncated": (
                    response_text[:1000] if len(response_text) > 1000 else response_text
                ),
                "session_state_keys": list(session_state.keys()),
                "session_state": session_state,
                "generated_files": generated_files,
                "s3_files": s3_files,
                "smiles_generated": smiles_generated,
                "n_molecules": len(smiles_generated),
                "s3_prefix": s3_prefix,
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }

        except KeyboardInterrupt:
            logger.warning(f"Run {run_id + 1} interrupted")
            return {
                "run_id": run_id,
                "prompt": prompt,
                "session_id": session_id,
                "status": "interrupted",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Run {run_id + 1} failed: {e}")
            return {
                "run_id": run_id,
                "prompt": prompt,
                "session_id": session_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _compare_outputs(self, outputs: List[Dict], test_name: str) -> Dict:
        """Compare outputs from multiple runs."""
        comparison_results = {}

        # Filter successful runs
        successful_outputs = [o for o in outputs if o.get("status") == "success"]

        if len(successful_outputs) < 2:
            logger.warning(f"Not enough successful runs to compare for {test_name}")
            return {"error": "Insufficient successful runs for comparison"}

        # Compare text responses
        texts = [o["response"] for o in successful_outputs if o.get("response")]
        if len(texts) >= 2:
            comparison_results["text"] = self.comparator.compare_text_outputs(texts)

        # Compare generated molecule counts (for autoencoder tests)
        if any(o.get("smiles_generated") for o in successful_outputs):
            import numpy as np

            n_mols = [o.get("n_molecules", 0) for o in successful_outputs]
            mean_mols = np.mean(n_mols) if n_mols else 0
            comparison_results["data"] = {
                "count_cv": np.std(n_mols) / mean_mols if mean_mols > 0 else 1.0,
                "count_mean": mean_mols,
                "count_std": np.std(n_mols),
                "row_jaccard": 1.0 - (np.std(n_mols) / mean_mols if mean_mols > 0 else 0),
                "column_match": 1.0,
                "value_stability": np.std(n_mols) / mean_mols if mean_mols > 0 else 0,
            }

        # Process consistency
        completion_rate = len(successful_outputs) / len(outputs) if outputs else 0

        # Extract tool sequences and calculate similarity
        tool_sequences = []
        for output in successful_outputs:
            # Try to get agent response object if available
            response_obj = output.get("response_object") or output.get("agent_response")
            if response_obj:
                seq = ToolSequenceComparator.extract_tool_sequence(response_obj)
                tool_sequences.append(seq)
            else:
                # Fallback: try to extract from session state
                session_state = output.get("session_state", {})
                seq = ToolSequenceComparator.extract_tool_sequence(session_state)
                tool_sequences.append(seq)

        # Calculate tool sequence similarity
        tool_similarity = ToolSequenceComparator.compare_sequences(tool_sequences)

        comparison_results["process"] = {
            "completion_rate": completion_rate,
            "tool_sequence_similarity": tool_similarity,
        }

        # Log tool sequence info for debugging
        if tool_sequences:
            logger.debug(f"Tool sequences: {tool_sequences}")
            logger.debug(f"Tool sequence similarity: {tool_similarity:.3f}")

        return comparison_results

    def _save_artifacts(self, test_name: str, outputs: List[Dict], comparison: Dict, score: float):
        """Save test artifacts for later analysis."""
        if not self.config.save_artifacts:
            return

        artifacts_dir = self.output_dir / test_name
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save each run's details
        for output in outputs:
            run_dir = artifacts_dir / f"run_{output['run_id']}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # Save prompt
            (run_dir / "prompt.txt").write_text(output.get("prompt", ""))

            # Save response
            (run_dir / "response.txt").write_text(output.get("response", ""))

            # Save run metadata
            metadata = {
                k: v for k, v in output.items() if k not in ["prompt", "response", "session_state"]
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

        # Save comparison results
        (artifacts_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, default=str)
        )

        # Save score
        (artifacts_dir / "score.txt").write_text(f"{score:.4f}")

        logger.info(f"Artifacts saved to {artifacts_dir}")

    def run_test(self, test_config: TestConfig) -> Dict:
        """Run a single robustness test with S3 session isolation."""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Running test: {test_config.name}")
        logger.info(f"Description: {test_config.description}")
        logger.info(f"{'=' * 60}\n")

        # Get prompts
        prompts = self._get_prompts(test_config)
        logger.info(f"Running {len(prompts)} prompt variations")

        # Setup S3 if needed
        s3_config = self._setup_s3()

        # Run variations with guaranteed S3 cleanup via try-finally
        outputs = []
        try:
            for i, prompt in enumerate(prompts):
                # Use context manager for each run to ensure S3 isolation and cleanup
                if s3_config and self.config.s3_session_isolation:
                    with self._s3_session_manager.create_isolated_session(
                        test_run_id=self.test_run_id, prompt_idx=i, variation_idx=0
                    ) as session_id:
                        logger.debug(f"Created isolated S3 session: {session_id}")
                        output = self._run_single_variation(
                            prompt=prompt,
                            test_name=test_config.name,
                            run_id=i,
                            s3_prefix=None,  # Prefix already set by context manager
                        )
                else:
                    output = self._run_single_variation(
                        prompt=prompt,
                        test_name=test_config.name,
                        run_id=i,
                        s3_prefix=None,
                    )

                outputs.append(output)

                # Log progress
                status = "✅" if output.get("status") == "success" else "❌"
                logger.info(f"  Run {i + 1}/{len(prompts)}: {status}")

        finally:
            # Ensure S3 prefix is restored even if test fails
            logger.debug("Ensuring S3 prefix restoration...")
            self._restore_s3_prefix()

        # Compare outputs
        comparison = self._compare_outputs(outputs, test_config.name)

        # Calculate robustness score
        score = self.metrics_calculator.calculate_robustness_score(comparison)

        # Save artifacts
        self._save_artifacts(test_config.name, outputs, comparison, score)

        # Prepare result
        result = {
            "test_name": test_config.name,
            "description": test_config.description,
            "n_variations": len(prompts),
            "successful_runs": sum(1 for o in outputs if o.get("status") == "success"),
            "robustness_score": score,
            "rating": self.metrics_calculator.get_rating(score),
            "passed": score >= self.config.pass_threshold,
            "comparison": comparison,
            "outputs": outputs if self.config.include_run_details else None,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"\nTest '{test_config.name}' completed:")
        logger.info(f"  Score: {score:.3f}")
        logger.info(f"  Rating: {result['rating']}")
        logger.info(f"  Passed: {'✅' if result['passed'] else '❌'}")

        return result

    def run_all_tests(self) -> Dict:
        """Run all enabled tests."""
        logger.info(f"\n{'#' * 60}")
        logger.info("Starting Robustness Test Suite")
        logger.info(f"Test Run ID: {self.test_run_id}")
        logger.info(f"{'#' * 60}\n")

        # Get enabled tests
        enabled_tests = [tc for tc in self.config.tests.values() if tc.enabled]

        if not enabled_tests:
            logger.warning("No tests enabled in configuration!")
            return {"error": "No tests enabled"}

        logger.info(f"Enabled tests: {[t.name for t in enabled_tests]}")

        # Run each test
        results = {}
        for test_config in enabled_tests:
            try:
                result = self.run_test(test_config)
                results[test_config.name] = result
                self.results[test_config.name] = result
            except Exception as e:
                logger.error(f"Test '{test_config.name}' failed with error: {e}")
                results[test_config.name] = {
                    "test_name": test_config.name,
                    "status": "error",
                    "error": str(e),
                }

        # Generate summary
        summary = self._generate_summary(results)
        self._save_reports(summary)

        return summary

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate test suite summary."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results.values() if r.get("passed", False))
        failed_tests = total_tests - passed_tests

        scores = [r.get("robustness_score", 0) for r in results.values() if "robustness_score" in r]
        avg_score = sum(scores) / len(scores) if scores else 0

        summary = {
            "test_run_id": self.test_run_id,
            "timestamp": datetime.now().isoformat(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "average_robustness_score": avg_score,
            "overall_rating": self.metrics_calculator.get_rating(avg_score),
            "pass_threshold": self.config.pass_threshold,
            "results": results,
        }

        return summary

    def _save_reports(self, summary: Dict):
        """Save summary reports."""
        # Save JSON summary
        if self.config.generate_json:
            json_path = self.output_dir / "summary.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"JSON summary saved to {json_path}")

        # Generate and save markdown report
        if self.config.generate_markdown:
            report = self._generate_markdown_report(summary)
            md_path = self.output_dir / "report.md"
            md_path.write_text(report)
            logger.info(f"Markdown report saved to {md_path}")

    def _generate_markdown_report(self, summary: Dict) -> str:
        """Generate comprehensive markdown report."""
        report = f"""# Robustness Test Report

**Test Run ID:** {summary['test_run_id']}
**Date:** {summary['timestamp']}

## Summary

| Metric | Value |
|--------|-------|
| Total Tests | {summary['total_tests']} |
| Passed | {summary['passed']} |
| Failed | {summary['failed']} |
| Pass Rate | {summary['pass_rate']:.1%} |
| Average Score | {summary['average_robustness_score']:.3f} |
| Overall Rating | {summary['overall_rating']} |
| Pass Threshold | {summary['pass_threshold']:.2f} |

## Test Results

"""
        for test_name, result in summary.get("results", {}).items():
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            score = result.get("robustness_score", 0)
            rating = result.get("rating", "N/A")

            report += f"""### {test_name}

- **Status:** {status}
- **Score:** {score:.3f}
- **Rating:** {rating}
- **Description:** {result.get('description', 'N/A')}
- **Variations:** {result.get('n_variations', 'N/A')}
- **Successful Runs:** {result.get('successful_runs', 'N/A')}

"""
            # Include comparison details
            comparison = result.get("comparison", {})
            if comparison and "text" in comparison:
                text_metrics = comparison["text"]
                report += f"""#### Text Comparison
- Semantic Similarity: {text_metrics.get('semantic_similarity', 0):.3f}
- Entity Overlap: {text_metrics.get('entity_overlap', 0):.3f}
- Numeric Consistency: {text_metrics.get('numeric_consistency', 0):.3f}

"""

        # Recommendations
        if self.config.include_recommendations:
            report += """## Recommendations

"""
            avg_score = summary["average_robustness_score"]
            if avg_score >= 0.9:
                report += (
                    "✅ **Excellent robustness.** The system handles prompt variations very well.\n"
                )
            elif avg_score >= 0.8:
                report += (
                    "✅ **Good robustness.** Minor inconsistencies but acceptable for production.\n"
                )
            elif avg_score >= 0.7:
                report += "⚠️ **Acceptable robustness** but room for improvement. Monitor closely.\n"
            else:
                report += "❌ **Concerning robustness.** Significant inconsistencies detected. Review agent prompts and tool implementations.\n"

        report += f"""
---
*Generated by Cs_copilot Robustness Testing Framework*
*Output directory: {self.output_dir}*
"""
        return report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run robustness tests for Cs_copilot agentic operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run python tests/robustness/robustness_minimal_example.py

  # Run with custom config
  uv run python tests/robustness/robustness_minimal_example.py --config my_config.yaml

  # Run specific test with custom variations
  uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --n-variations 3

  # Run multiple tests
  uv run python tests/robustness/robustness_minimal_example.py --test chembl_download --test autoencoder_sampling
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "robustness_config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--test",
        action="append",
        dest="tests",
        help="Specific test(s) to run (can be used multiple times)",
    )

    parser.add_argument(
        "--n-variations",
        type=int,
        help="Number of prompt variations to use (overrides config)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available tests and exit",
    )

    parser.add_argument(
        "--list-prompts",
        action="store_true",
        help="List available prompt categories and exit",
    )

    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for test runs (logs all metrics, parameters, and artifacts)",
    )

    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="robustness_testing",
        help="MLflow experiment name (default: robustness_testing)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if not args.config.exists():
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)

    # Handle list commands
    if args.list_tests:
        print("\nAvailable tests:")
        for name, test in config.tests.items():
            status = "✓ enabled" if test.enabled else "✗ disabled"
            print(f"  {name}: {status}")
            print(f"    {test.description}")
        sys.exit(0)

    if args.list_prompts:
        from prompt_variations import PromptVariationGenerator

        generator = PromptVariationGenerator()
        print("\nAvailable prompt categories:")
        for name in generator.list_available_prompts():
            print(f"  - {name}")
        sys.exit(0)

    # Apply command line overrides
    if args.n_variations:
        config.n_variations = args.n_variations

    if args.debug:
        config.debug_mode = True

    if args.tests:
        # Enable only specified tests
        for test_name in config.tests:
            config.tests[test_name].enabled = test_name in args.tests

    # Create runner (MLflow-enhanced if requested)
    if args.mlflow:
        try:
            from mlflow_runner import MLflowRobustnessRunner

            logger.info(f"Creating MLflow-enhanced runner (experiment: {args.mlflow_experiment})")
            runner = MLflowRobustnessRunner(
                config, experiment_name=args.mlflow_experiment, enable_mlflow=True
            )
        except ImportError as e:
            logger.warning(f"MLflow dependencies not available: {e}. Using standard runner.")
            runner = RobustnessRunner(config)
    else:
        runner = RobustnessRunner(config)

    try:
        summary = runner.run_all_tests()

        # Print final summary
        print(f"\n{'=' * 60}")
        print("ROBUSTNESS TEST SUITE COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total Tests: {summary.get('total_tests', 0)}")
        print(f"Passed: {summary.get('passed', 0)}")
        print(f"Failed: {summary.get('failed', 0)}")
        print(f"Average Score: {summary.get('average_robustness_score', 0):.3f}")
        print(f"Overall Rating: {summary.get('overall_rating', 'N/A')}")
        print(f"\nReports saved to: {runner.output_dir}")
        print(f"{'=' * 60}")

        # Exit with appropriate code
        if summary.get("failed", 0) > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nTest suite interrupted by user")
        sys.exit(130)

    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
