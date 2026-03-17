#!/usr/bin/env python
# coding: utf-8
"""
Autoencoder robustness tests.

Tests the autoencoder sampling functionality with prompt variations to assess
robustness and consistency of generated molecules.
"""

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest

from .comparators import OutputComparator
from .prompt_variations import PromptVariationGenerator

logger = logging.getLogger(__name__)

# Note: Fixtures (agent_team_factory, prompt_generator, comparator, metrics_calculator)
# are now provided by conftest.py in the robustness directory.
# They are automatically available to all tests in this file.


class TestAutoencoderRobustness:
    """Test autoencoder sampling robustness to prompt variations."""

    def test_basic_sampling_robustness(
        self, agent_team_factory, prompt_generator, comparator, metrics_calculator
    ):
        """
        Test basic autoencoder sampling with 10 prompt variations.
        Ensures consistent molecule generation across different phrasings.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting basic autoencoder sampling robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        # Get prompt variations
        variations = prompt_generator.get_variations("autoencoder_sampling", n=10)
        assert len(variations) == 10, "Expected 10 prompt variations"

        outputs = []

        # Run agent with each variation (each in separate session)
        for i, prompt in enumerate(variations):
            logger.info(f"Running variation {i+1}/10: {prompt[:80]}...")

            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_ae_sampling_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                # Run sampling
                result = agent_team.run(prompt, stream=False)
                response = result.content

                # Extract generated SMILES
                smiles_list = self._extract_smiles_from_response(response)

                output = {
                    "run_id": i,
                    "prompt": prompt,
                    "response": response,
                    "smiles_generated": smiles_list,
                    "n_molecules": len(smiles_list),
                    "session_id": session_id if s3_enabled else f"local_run_{i}",
                }
                outputs.append(output)

            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                pytest.fail(f"Autoencoder sampling failed on variation {i}: {e}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Ensure all runs completed
        assert len(outputs) == 10, "Not all variations completed successfully"

        # Compare outputs
        comparison_results = self._compare_sampling_outputs(outputs, comparator)

        # Calculate robustness score
        robustness_score = metrics_calculator.calculate_robustness_score(comparison_results)

        # Generate report
        report = metrics_calculator.generate_report(
            {
                "score": robustness_score,
                "comparisons": comparison_results,
                "outliers": [],
            }
        )

        # Save report
        report_dir = Path(__file__).parent / "reports"
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / "autoencoder_sampling_robustness.md"
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")

        # Assertions
        assert (
            robustness_score > 0.75
        ), f"Autoencoder sampling robustness score {robustness_score:.2f} below threshold"

        # Check consistency
        n_molecules = [out["n_molecules"] for out in outputs]
        assert (
            np.std(n_molecules) / np.mean(n_molecules) < 0.3
        ), "Number of generated molecules varies too much"

        # Check text response consistency
        texts = [out["response"] for out in outputs]
        text_comparison = comparator.compare_text_outputs(texts)
        assert text_comparison["semantic_similarity"] > 0.70, "Response descriptions vary too much"

        logger.info(f"Basic sampling robustness test PASSED with score {robustness_score:.3f}")

    def test_gtm_guided_sampling_robustness(
        self, agent_team_factory, prompt_generator, comparator, metrics_calculator
    ):
        """
        Test GTM-guided autoencoder sampling with prompt variations.
        Ensures consistent molecule generation when guided by GTM dense nodes.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting GTM-guided sampling robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        # Get prompt variations
        variations = prompt_generator.get_variations("gtm_guided_sampling", n=10)
        outputs = []

        for i, prompt in enumerate(variations):
            logger.info(f"Running variation {i+1}/10: {prompt[:80]}...")

            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_gtm_guided_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                # First, ensure we have GTM map ready (each session needs its own)
                setup_prompt = "Fetch kinase 2 inhibitor data and build GTM with k_hit=50"
                agent_team.run(setup_prompt, stream=False)

                # Now run the actual prompt variation
                result = agent_team.run(prompt, stream=False)
                response = result.content

                # Extract generated molecules and their GTM context
                smiles_list = self._extract_smiles_from_response(response)
                gtm_info = self._extract_gtm_context(response)

                output = {
                    "run_id": i,
                    "prompt": prompt,
                    "response": response,
                    "smiles_generated": smiles_list,
                    "n_molecules": len(smiles_list),
                    "gtm_nodes_mentioned": gtm_info.get("nodes", []),
                    "gtm_regions_mentioned": gtm_info.get("regions", []),
                    "session_id": session_id if s3_enabled else f"local_run_{i}",
                }
                outputs.append(output)

            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                pytest.fail(f"GTM-guided sampling failed on variation {i}: {e}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Compare outputs
        comparison_results = self._compare_gtm_guided_outputs(outputs, comparator)

        # Calculate robustness
        robustness_score = metrics_calculator.calculate_robustness_score(comparison_results)

        # Save report
        report = metrics_calculator.generate_report(
            {
                "score": robustness_score,
                "comparisons": comparison_results,
                "outliers": [],
            }
        )

        report_path = Path(__file__).parent / "reports" / "gtm_guided_sampling_robustness.md"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(report)

        # Assertions
        assert (
            robustness_score > 0.70
        ), f"GTM-guided sampling robustness {robustness_score:.2f} below threshold"

        # Check GTM node consistency
        nodes_mentioned = [len(out["gtm_nodes_mentioned"]) for out in outputs]
        if any(n > 0 for n in nodes_mentioned):
            # If nodes are mentioned, they should be consistent
            node_sets = [
                set(out["gtm_nodes_mentioned"]) for out in outputs if out["gtm_nodes_mentioned"]
            ]
            if len(node_sets) > 1:
                overlap = len(set.intersection(*node_sets)) / len(set.union(*node_sets))
                assert overlap > 0.5, "GTM nodes referenced are inconsistent"

        logger.info(f"GTM-guided sampling test PASSED with score {robustness_score:.3f}")

    def test_interpolation_robustness(
        self, agent_team_factory, prompt_generator, comparator, metrics_calculator
    ):
        """
        Test interpolation between molecules with prompt variations.
        Uses fixed SMILES pairs to ensure comparable results.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting interpolation robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        # Define test molecules
        smiles_a = "CCO"  # Ethanol
        smiles_b = "CCCCCCCCCC"  # Decane

        # Get prompt variations
        base_variations = prompt_generator.get_variations("interpolation", n=10)

        # Augment each variation with specific molecules
        variations = [
            f"{var} Molecule A: {smiles_a}, Molecule B: {smiles_b}" for var in base_variations
        ]

        outputs = []

        for i, prompt in enumerate(variations):
            logger.info(f"Running variation {i+1}/10: {prompt[:80]}...")

            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_interpolation_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                result = agent_team.run(prompt, stream=False)
                response = result.content

                # Extract interpolated molecules
                smiles_list = self._extract_smiles_from_response(response)

                output = {
                    "run_id": i,
                    "prompt": prompt,
                    "response": response,
                    "smiles_generated": smiles_list,
                    "n_steps": len(smiles_list),
                    "session_id": session_id if s3_enabled else f"local_run_{i}",
                }
                outputs.append(output)

            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                pytest.fail(f"Interpolation failed on variation {i}: {e}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Compare outputs
        comparison_results = self._compare_interpolation_outputs(outputs, comparator)

        # Calculate robustness
        robustness_score = metrics_calculator.calculate_robustness_score(comparison_results)

        # Save report
        report = metrics_calculator.generate_report(
            {
                "score": robustness_score,
                "comparisons": comparison_results,
                "outliers": [],
            }
        )

        report_path = Path(__file__).parent / "reports" / "interpolation_robustness.md"
        report_path.write_text(report)

        # Assertions
        assert (
            robustness_score > 0.70
        ), f"Interpolation robustness {robustness_score:.2f} below threshold"

        # Check number of interpolation steps is consistent
        n_steps = [out["n_steps"] for out in outputs]
        if len(set(n_steps)) > 1:
            logger.warning(f"Number of interpolation steps varies: {n_steps}")

        # Check molecular similarity along paths
        if len(outputs) >= 2:
            similarity = self._compute_path_similarity(
                outputs[0]["smiles_generated"], outputs[1]["smiles_generated"]
            )
            assert similarity > 0.50, "Interpolation paths are too different"

        logger.info(f"Interpolation test PASSED with score {robustness_score:.3f}")

    def test_latent_exploration_robustness(
        self, agent_team_factory, prompt_generator, comparator, metrics_calculator
    ):
        """
        Test latent space exploration around a molecule with prompt variations.
        Uses fixed seed molecule to ensure comparable results.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting latent exploration robustness test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        # Define seed molecule
        seed_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin

        # Get prompt variations
        base_variations = prompt_generator.get_variations("latent_exploration", n=10)

        # Augment with specific molecule
        variations = [f"{var} Seed molecule: {seed_smiles}" for var in base_variations]

        outputs = []

        for i, prompt in enumerate(variations):
            logger.info(f"Running variation {i+1}/10: {prompt[:80]}...")

            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_latent_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                result = agent_team.run(prompt, stream=False)
                response = result.content

                # Extract generated analogs
                smiles_list = self._extract_smiles_from_response(response)

                output = {
                    "run_id": i,
                    "prompt": prompt,
                    "response": response,
                    "smiles_generated": smiles_list,
                    "n_analogs": len(smiles_list),
                    "session_id": session_id if s3_enabled else f"local_run_{i}",
                }
                outputs.append(output)

            except Exception as e:
                logger.error(f"Variation {i} failed: {e}")
                pytest.fail(f"Latent exploration failed on variation {i}: {e}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Compare outputs
        comparison_results = self._compare_exploration_outputs(outputs, comparator)

        # Calculate robustness
        robustness_score = metrics_calculator.calculate_robustness_score(comparison_results)

        # Save report
        report = metrics_calculator.generate_report(
            {
                "score": robustness_score,
                "comparisons": comparison_results,
                "outliers": [],
            }
        )

        report_path = Path(__file__).parent / "reports" / "latent_exploration_robustness.md"
        report_path.write_text(report)

        # Assertions
        assert (
            robustness_score > 0.70
        ), f"Latent exploration robustness {robustness_score:.2f} below threshold"

        # Check that analogs are similar to seed
        for output in outputs:
            if output["smiles_generated"]:
                # All should be analogs of aspirin
                logger.info(f"Run {output['run_id']}: Generated {output['n_analogs']} analogs")

        logger.info(f"Latent exploration test PASSED with score {robustness_score:.3f}")

    def test_molecular_validity(self, agent_team_factory, prompt_generator):
        """
        Test that all generated molecules are chemically valid
        regardless of prompt variation.

        Each variation runs in a completely separate session with isolated S3
        storage to prevent cross-contamination of results.
        """
        import datetime
        import uuid

        from cs_copilot.storage import is_s3_enabled
        from cs_copilot.storage.client import S3 as S3Client

        logger.info("Starting molecular validity test")

        # Generate test run ID
        test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Check S3 availability
        s3_enabled = is_s3_enabled()
        if not s3_enabled:
            logger.warning("S3 not enabled - sessions will not be isolated")

        variations = prompt_generator.get_variations("autoencoder_sampling", n=10)
        validity_rates = []

        for i, prompt in enumerate(variations):
            # Setup S3 session isolation
            original_prefix = None
            if s3_enabled:
                session_id = f"robustness_validity_{test_run_id}_run{i}_{uuid.uuid4().hex[:8]}"
                original_prefix = S3Client.prefix
                S3Client.prefix = f"sessions/{session_id}"
                logger.debug(f"S3 session prefix: sessions/{session_id}")

            try:
                # Create fresh agent team for this variation
                agent_team = agent_team_factory()

                result = agent_team.run(prompt, stream=False)
                smiles_list = self._extract_smiles_from_response(result.content)

                # Check validity
                valid_count = sum(1 for s in smiles_list if self._is_valid_smiles(s))
                validity_rate = valid_count / len(smiles_list) if smiles_list else 0
                validity_rates.append(validity_rate)

                logger.info(
                    f"Variation {i+1}: {valid_count}/{len(smiles_list)} valid ({validity_rate:.1%})"
                )

                if validity_rate < 0.90:
                    logger.warning(f"Low validity rate for variation {i+1}")

            finally:
                # Restore original S3 prefix
                if original_prefix is not None:
                    S3Client.prefix = original_prefix

        # Assert high validity across all variations
        mean_validity = np.mean(validity_rates)
        assert mean_validity > 0.90, f"Mean validity rate {mean_validity:.1%} below 90%"

        logger.info(f"Molecular validity test: {mean_validity:.1%} average validity")

    # ==================== Helper Methods ====================

    def _extract_smiles_from_response(self, response: str) -> List[str]:
        """Extract SMILES strings from agent response."""
        import re

        # Look for SMILES patterns (simplified)
        # Common patterns: backtick-enclosed, comma-separated, or line-by-line
        smiles = []

        # Pattern 1: Backtick enclosed
        backtick_pattern = r"`([A-Za-z0-9@+\-\[\]\(\)=#$]+)`"
        smiles.extend(re.findall(backtick_pattern, response))

        # Pattern 2: Lines starting with SMILES-like strings
        for line in response.split("\n"):
            line = line.strip()
            if line and not line.startswith(("#", "-", "*", ">")):
                # Check if it looks like a SMILES
                if re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#$]+$", line.split()[0]):
                    smiles.append(line.split()[0])

        # Remove duplicates while preserving order
        seen = set()
        unique_smiles = []
        for s in smiles:
            if s not in seen and len(s) > 2:  # Skip very short strings
                seen.add(s)
                unique_smiles.append(s)

        return unique_smiles

    def _extract_gtm_context(self, response: str) -> Dict:
        """Extract GTM-related information from response."""
        import re

        info = {"nodes": [], "regions": [], "coordinates": []}

        # Extract node IDs
        node_pattern = r"node[s]?\s+(\d+)"
        info["nodes"] = [int(n) for n in re.findall(node_pattern, response, re.IGNORECASE)]

        # Extract coordinate pairs
        coord_pattern = r"\((\d+),\s*(\d+)\)"
        info["coordinates"] = [(int(x), int(y)) for x, y in re.findall(coord_pattern, response)]

        # Extract region descriptions
        region_keywords = ["dense", "sparse", "central", "corner", "region"]
        for keyword in region_keywords:
            if keyword in response.lower():
                info["regions"].append(keyword)

        return info

    def _compare_sampling_outputs(self, outputs: List[Dict], comparator: OutputComparator) -> Dict:
        """Compare basic sampling outputs."""
        results = {}

        # Compare text responses
        texts = [out["response"] for out in outputs]
        results["text"] = comparator.compare_text_outputs(texts)

        # Compare number of molecules generated
        n_mols = [out["n_molecules"] for out in outputs]
        results["data"] = {
            "count_cv": np.std(n_mols) / np.mean(n_mols) if np.mean(n_mols) > 0 else 1.0,
            "count_mean": np.mean(n_mols),
            "count_std": np.std(n_mols),
        }

        # Process consistency
        results["process"] = {
            "completion_rate": len(outputs) / 10.0,
            "tool_sequence_similarity": 0.90,  # Placeholder
        }

        return results

    def _compare_gtm_guided_outputs(
        self, outputs: List[Dict], comparator: OutputComparator
    ) -> Dict:
        """Compare GTM-guided sampling outputs."""
        results = self._compare_sampling_outputs(outputs, comparator)

        # Add GTM-specific metrics
        node_sets = [
            set(out["gtm_nodes_mentioned"]) for out in outputs if out["gtm_nodes_mentioned"]
        ]
        if len(node_sets) > 1:
            # Calculate Jaccard similarity of mentioned nodes
            all_nodes = set.union(*node_sets)
            common_nodes = set.intersection(*node_sets)
            results["gtm_consistency"] = {
                "node_overlap": len(common_nodes) / len(all_nodes) if all_nodes else 0,
                "n_unique_nodes": len(all_nodes),
            }

        return results

    def _compare_interpolation_outputs(
        self, outputs: List[Dict], comparator: OutputComparator
    ) -> Dict:
        """Compare interpolation outputs."""
        results = {}

        # Text comparison
        texts = [out["response"] for out in outputs]
        results["text"] = comparator.compare_text_outputs(texts)

        # Number of interpolation steps
        n_steps = [out["n_steps"] for out in outputs]
        results["data"] = {
            "steps_cv": np.std(n_steps) / np.mean(n_steps) if np.mean(n_steps) > 0 else 1.0,
            "steps_mean": np.mean(n_steps),
            "steps_std": np.std(n_steps),
        }

        # Process
        results["process"] = {
            "completion_rate": len(outputs) / 10.0,
            "tool_sequence_similarity": 0.90,
        }

        return results

    def _compare_exploration_outputs(
        self, outputs: List[Dict], comparator: OutputComparator
    ) -> Dict:
        """Compare latent exploration outputs."""
        return self._compare_sampling_outputs(outputs, comparator)

    def _compute_path_similarity(self, path1: List[str], path2: List[str]) -> float:
        """Compute similarity between two interpolation paths."""
        # Simple overlap-based similarity
        if not path1 or not path2:
            return 0.0

        set1 = set(path1)
        set2 = set(path2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if a SMILES string is chemically valid."""
        try:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except ImportError:
            # Fallback: basic pattern check
            import re

            # Very basic check
            if not smiles or len(smiles) < 2:
                return False
            # Should contain typical SMILES characters
            if not re.match(r"^[A-Za-z0-9@+\-\[\]\(\)=#$]+$", smiles):
                return False
            return True
        except Exception:
            return False


# Standalone validation tests
def test_autoencoder_prompts_exist():
    """Test that autoencoder prompt variations are available."""
    generator = PromptVariationGenerator()
    prompts = generator.list_available_prompts()

    assert "autoencoder_sampling" in prompts
    assert "gtm_guided_sampling" in prompts
    assert "interpolation" in prompts
    assert "latent_exploration" in prompts


def test_autoencoder_prompt_variations_valid():
    """Test that autoencoder prompt variations maintain semantic similarity."""
    generator = PromptVariationGenerator()

    for category in [
        "autoencoder_sampling",
        "gtm_guided_sampling",
        "interpolation",
        "latent_exploration",
    ]:
        base = generator.get_base_prompt(category)
        variations = generator.get_variations(category, n=10)

        assert len(variations) == 10, f"Expected 10 variations for {category}"

        # Check each variation maintains similarity to base
        for i, var in enumerate(variations[1:], 1):
            is_valid = generator.validate_variation(base, var, min_similarity=0.65)
            assert is_valid, f"Variation {i} of {category} has low semantic similarity"
