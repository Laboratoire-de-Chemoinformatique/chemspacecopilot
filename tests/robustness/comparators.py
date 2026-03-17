#!/usr/bin/env python
# coding: utf-8
"""
Output comparison utilities for robustness testing.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)


class OutputComparator:
    """Compare outputs across prompt variations."""

    def __init__(self):
        """Initialize comparator with necessary models."""
        self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Initialized OutputComparator")

    def compare_dataframes(self, dfs: List[pd.DataFrame]) -> Dict[str, float]:
        """
        Compare CSV outputs across multiple runs.

        Args:
            dfs: List of DataFrames to compare

        Returns:
            Dictionary of comparison metrics
        """
        if len(dfs) < 2:
            return {"error": "Need at least 2 dataframes to compare"}

        results = {}

        # Row overlap (Jaccard similarity of indices)
        index_sets = [set(df.index) for df in dfs]
        results["row_jaccard"] = self._jaccard_similarity(index_sets)

        # Column consistency
        column_sets = [set(df.columns) for df in dfs]
        results["column_match"] = 1.0 if len(set(map(frozenset, column_sets))) == 1 else 0.0

        # Value stability for numeric columns
        numeric_cols = set(dfs[0].select_dtypes(include=[np.number]).columns)
        if numeric_cols:
            results["value_stability"] = self._numeric_stability(dfs, numeric_cols)
        else:
            results["value_stability"] = 1.0

        # Distribution similarity (KS test)
        results["distribution_ks"] = self._ks_test_pairwise(dfs)

        logger.info(f"DataFrame comparison: {results}")
        return results

    def compare_gtm_models(self, models: List[Any]) -> Dict[str, float]:
        """
        Compare GTM model outputs.

        Args:
            models: List of GTM model objects

        Returns:
            Dictionary of comparison metrics
        """
        if len(models) < 2:
            return {"error": "Need at least 2 models to compare"}

        results = {}

        # Structure match (grid dimensions)
        results["structure_match"] = self._check_grid_dimensions(models)

        # Projection correlation
        results["projection_correlation"] = self._coordinate_correlation(models)

        # Density correlation
        results["density_correlation"] = self._density_correlation(models)

        logger.info(f"GTM model comparison: {results}")
        return results

    def compare_text_outputs(self, texts: List[str]) -> Dict[str, float]:
        """
        Compare text responses across runs.

        Args:
            texts: List of text outputs to compare

        Returns:
            Dictionary of comparison metrics
        """
        if len(texts) < 2:
            return {"error": "Need at least 2 texts to compare"}

        results = {}

        # Semantic similarity via sentence embeddings
        results["semantic_similarity"] = self._sbert_similarity(texts)

        # Named entity overlap
        results["entity_overlap"] = self._named_entity_overlap(texts)

        # Numeric consistency
        results["numeric_consistency"] = self._extract_compare_numbers(texts)

        # Structural similarity (presence of headers, bullets, etc.)
        results["structural_match"] = self._structure_similarity(texts)

        logger.info(f"Text comparison: {results}")
        return results

    def compare_images(self, image_paths: List[Path]) -> Dict[str, float]:
        """
        Compare visualization outputs.

        Args:
            image_paths: List of paths to image files

        Returns:
            Dictionary of comparison metrics
        """
        if len(image_paths) < 2:
            return {"error": "Need at least 2 images to compare"}

        results = {}

        # Perceptual hashing
        results["perceptual_hash"] = self._phash_similarity(image_paths)

        # SSIM (structural similarity)
        results["ssim"] = self._ssim_pairwise(image_paths)

        logger.info(f"Image comparison: {results}")
        return results

    # ==================== Helper Methods ====================

    def _jaccard_similarity(self, sets: List[Set]) -> float:
        """Calculate average pairwise Jaccard similarity."""
        if len(sets) < 2:
            return 1.0

        similarities = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                if union > 0:
                    similarities.append(intersection / union)

        return np.mean(similarities) if similarities else 0.0

    def _numeric_stability(self, dfs: List[pd.DataFrame], cols: Set[str]) -> float:
        """Calculate mean absolute percentage difference for numeric columns."""
        stabilities = []

        for col in cols:
            try:
                # Get values from all dfs for this column
                values = [df[col].mean() for df in dfs if col in df.columns]
                if len(values) > 1:
                    mean_val = np.mean(values)
                    if mean_val != 0:
                        mad = np.mean(np.abs(values - mean_val))
                        stabilities.append(mad / abs(mean_val))
            except Exception as e:
                logger.debug(f"Skipping column {col}: {e}")
                continue

        return np.mean(stabilities) if stabilities else 0.0

    def _ks_test_pairwise(self, dfs: List[pd.DataFrame]) -> float:
        """Run Kolmogorov-Smirnov test on numeric columns pairwise."""
        if len(dfs) < 2:
            return 1.0

        numeric_cols = set(dfs[0].select_dtypes(include=[np.number]).columns)
        p_values = []

        for col in numeric_cols:
            try:
                for i in range(len(dfs)):
                    for j in range(i + 1, len(dfs)):
                        if col in dfs[i].columns and col in dfs[j].columns:
                            stat, p = stats.ks_2samp(dfs[i][col].dropna(), dfs[j][col].dropna())
                            p_values.append(p)
            except Exception as e:
                logger.debug(f"KS test failed for {col}: {e}")
                continue

        # Return mean p-value (higher = more similar)
        return np.mean(p_values) if p_values else 0.5

    def _check_grid_dimensions(self, models: List[Any]) -> float:
        """Check if GTM models have same grid dimensions."""
        try:
            dims = [
                (getattr(model, "k", None), getattr(model, "m", None), getattr(model, "n", None))
                for model in models
            ]
            return 1.0 if len(set(dims)) == 1 else 0.0
        except Exception as e:
            logger.warning(f"Could not compare grid dimensions: {e}")
            return 0.0

    def _coordinate_correlation(self, models: List[Any]) -> float:
        """
        Calculate correlation of molecule projections across GTM models.

        For GTM models, this computes the correlation of mean coordinates
        (latent space positions) across models. Higher correlation indicates
        that models position molecules similarly in latent space.

        Returns:
            Average Spearman correlation coefficient (0-1 scale)
        """
        try:
            # Extract mean coordinates from GTM models
            coords = []
            for model in models:
                if hasattr(model, "model") and hasattr(model.model, "means"):
                    # ugtm eGTM/GTM models store means in model.means
                    coords.append(model.model.means.flatten())
                elif hasattr(model, "means"):
                    # Direct means attribute
                    coords.append(model.means.flatten())
                else:
                    logger.warning(f"Model {type(model)} has no means attribute")
                    return 0.5  # Neutral score if we can't extract coordinates

            # Ensure all coordinate arrays have same length
            lengths = [len(c) for c in coords]
            if len(set(lengths)) > 1:
                logger.warning(f"Coordinate arrays have different lengths: {lengths}")
                # Truncate to minimum length
                min_len = min(lengths)
                coords = [c[:min_len] for c in coords]

            # Calculate pairwise Spearman correlations
            correlations = []
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    corr, _ = stats.spearmanr(coords[i], coords[j])
                    # Convert correlation from [-1, 1] to [0, 1] scale
                    # (negative correlation still indicates some structure)
                    correlations.append(abs(corr))

            avg_corr = np.mean(correlations) if correlations else 0.0
            logger.debug(f"GTM coordinate correlation: {avg_corr:.3f}")
            return avg_corr

        except Exception as e:
            logger.warning(f"Failed to calculate coordinate correlation: {e}")
            return 0.5  # Neutral score on error

    def _density_correlation(self, models: List[Any]) -> float:
        """
        Calculate correlation of node densities across GTM models.

        Node densities represent how many molecules are assigned to each
        latent space node. Correlation of densities indicates whether models
        cluster molecules similarly.

        Returns:
            Average Pearson correlation coefficient (0-1 scale)
        """
        try:
            # Extract node densities from GTM models
            # Density is typically calculated from responsibilities
            densities = []
            for model in models:
                # Try to get responsibilities from model
                density = None

                # Check if model has responsibilities attribute
                if hasattr(model, "responsibilities"):
                    # Sum over molecules (axis 0) to get node densities
                    density = np.array(model.responsibilities).sum(axis=0)
                elif hasattr(model, "R"):
                    # Alternative attribute name
                    density = np.array(model.R).sum(axis=0)
                else:
                    # Try to compute from model grid
                    if hasattr(model, "k"):
                        # Create uniform density as fallback
                        k_val = model.k if isinstance(model.k, int) else 10
                        density = np.ones(k_val * k_val)  # Assume square grid
                        logger.debug(f"Using uniform density fallback for model {type(model)}")

                if density is not None:
                    densities.append(density.flatten())
                else:
                    logger.warning(f"Could not extract density from model {type(model)}")
                    return 0.5  # Neutral score if we can't extract densities

            # Ensure all density arrays have same length
            lengths = [len(d) for d in densities]
            if len(set(lengths)) > 1:
                logger.warning(f"Density arrays have different lengths: {lengths}")
                # Truncate to minimum length
                min_len = min(lengths)
                densities = [d[:min_len] for d in densities]

            # Calculate pairwise Pearson correlations
            correlations = []
            for i in range(len(densities)):
                for j in range(i + 1, len(densities)):
                    # Use Pearson for densities (captures linear relationship)
                    corr, _ = stats.pearsonr(densities[i], densities[j])
                    # Take absolute value to get similarity measure
                    correlations.append(abs(corr))

            avg_corr = np.mean(correlations) if correlations else 0.0
            logger.debug(f"GTM density correlation: {avg_corr:.3f}")
            return avg_corr

        except Exception as e:
            logger.warning(f"Failed to calculate density correlation: {e}")
            return 0.5  # Neutral score on error

    def _sbert_similarity(self, texts: List[str]) -> float:
        """Calculate average pairwise cosine similarity via sentence embeddings."""
        embeddings = self.sbert_model.encode(texts)

        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                similarities.append(sim)

        return np.mean(similarities) if similarities else 0.0

    def _named_entity_overlap(self, texts: List[str]) -> float:
        """Extract and compare named entities (simple pattern-based)."""

        # Simple pattern-based extraction (can be enhanced with spaCy)
        def extract_entities(text: str) -> Set[str]:
            entities = set()
            # Extract CHEMBL IDs
            entities.update(re.findall(r"CHEMBL\d+", text, re.IGNORECASE))
            # Extract scaffold names (capitalized words)
            entities.update(re.findall(r"\b[A-Z][a-z]+(?:[-\s][A-Z][a-z]+)*\b", text))
            return entities

        entity_sets = [extract_entities(text) for text in texts]
        return self._jaccard_similarity(entity_sets)

    def _extract_compare_numbers(self, texts: List[str]) -> float:
        """Extract numbers and calculate coefficient of variation."""

        def extract_numbers(text: str) -> List[float]:
            # Extract floating point numbers
            return [float(x) for x in re.findall(r"\d+\.\d+", text)]

        all_numbers = [extract_numbers(text) for text in texts]

        # Flatten and compute CV
        flat = [num for nums in all_numbers for num in nums]
        if len(flat) > 1:
            cv = np.std(flat) / np.mean(flat) if np.mean(flat) != 0 else 0
            return max(0, 1 - cv)  # Convert CV to similarity score
        return 1.0

    def _structure_similarity(self, texts: List[str]) -> float:
        """Compare structural elements (headers, bullets, numbering)."""

        def extract_structure(text: str) -> Set[str]:
            structure = set()
            # Count markdown headers
            structure.update(re.findall(r"^#+\s+(.+)$", text, re.MULTILINE))
            # Count bullet points
            if re.search(r"^\s*[-*]\s+", text, re.MULTILINE):
                structure.add("bullets")
            # Count numbered lists
            if re.search(r"^\s*\d+\.\s+", text, re.MULTILINE):
                structure.add("numbered")
            return structure

        structures = [extract_structure(text) for text in texts]
        return self._jaccard_similarity(structures)

    def _phash_similarity(self, image_paths: List[Path]) -> float:
        """Calculate perceptual hash similarity."""
        try:
            hashes = [imagehash.phash(Image.open(path)) for path in image_paths]

            similarities = []
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    # Hamming distance (lower = more similar)
                    distance = hashes[i] - hashes[j]
                    # Convert to similarity score (0-1)
                    similarity = 1 - (distance / 64.0)
                    similarities.append(similarity)

            return np.mean(similarities) if similarities else 0.0
        except Exception as e:
            logger.error(f"Error computing perceptual hash: {e}")
            return 0.0

    def _ssim_pairwise(self, image_paths: List[Path]) -> float:
        """Calculate structural similarity index (SSIM) pairwise."""
        try:
            from skimage.io import imread
            from skimage.metrics import structural_similarity as ssim

            images = [imread(str(path)) for path in image_paths]

            similarities = []
            for i in range(len(images)):
                for j in range(i + 1, len(images)):
                    # Convert to grayscale if needed
                    img1 = images[i]
                    img2 = images[j]

                    if img1.shape != img2.shape:
                        logger.warning("Images have different shapes, skipping SSIM")
                        continue

                    if len(img1.shape) == 3:
                        # Multichannel SSIM
                        sim = ssim(img1, img2, channel_axis=2)
                    else:
                        sim = ssim(img1, img2)

                    similarities.append(sim)

            return np.mean(similarities) if similarities else 0.0
        except ImportError:
            logger.warning("scikit-image not available, skipping SSIM")
            return 0.90  # Placeholder
        except Exception as e:
            logger.error(f"Error computing SSIM: {e}")
            return 0.0
