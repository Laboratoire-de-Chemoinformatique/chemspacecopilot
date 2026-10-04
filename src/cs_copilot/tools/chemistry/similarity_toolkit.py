#!/usr/bin/env python
# coding: utf-8
"""
Chemical similarity toolkit providing various similarity metrics and calculations.

This module extends the base chemistry toolkit with specialized similarity
calculations including Tanimoto, Dice, Tversky, and other similarity metrics.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from numba import njit
from rdkit import DataStructs
from scipy.spatial.distance import cdist

from .base_chemistry import BaseChemistryToolkit, ChemistryError, calc_morgan_fp

logger = logging.getLogger(__name__)


class SimilarityError(ChemistryError):
    """Exception raised for similarity calculation errors."""

    pass


@njit(cache=True)
def _tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot-product Tanimoto on validated float64 rows; two empty rows score 1.

    Uses the count formula from SCOPE-DEL, but returns similarity, not 1 - T.
    Conversion before this kernel avoids integer dot/square overflow and Numba's
    unsupported integer np.dot overload. No fastmath: preserve finite checks.
    """
    scores = np.empty((len(a), len(b)))
    aa = np.sum(a * a, axis=1)
    bb = np.sum(b * b, axis=1)
    for i in range(len(a)):
        for j in range(len(b)):
            ab = np.dot(a[i], b[j])
            denominator = aa[i] + bb[j] - ab
            scores[i, j] = 1.0 if denominator == 0 else ab / denominator
    return scores


class ChemicalSimilarityToolkit(BaseChemistryToolkit):
    """
    Chemical similarity toolkit providing various similarity metrics.

    This class extends the base chemistry toolkit with specialized similarity
    calculations for molecular comparison and analysis.
    """

    def __init__(self):
        """Initialize the ChemicalSimilarityToolkit."""
        super().__init__("chemical_similarity")
        # Register all similarity tools
        self.register(self.calculate_tanimoto_similarity)
        self.register(self.calculate_dice_similarity)
        self.register(self.calculate_tversky_similarity)
        self.register(self.calculate_cosine_similarity)
        self.register(self.calculate_euclidean_distance)
        self.register(self.calculate_all_similarities)
        self.register(self.find_most_similar)
        self.register(self.calculate_similarity_matrix)

    def calculate_similarity_matrix(
        self,
        vectors_a: List[List[float]],
        vectors_b: List[List[float]],
        descriptor_kind: str,
        metric: str = "auto",
    ) -> Dict[str, Any]:
        """Compare rows of existing descriptors, preserving counts and embeddings.

        Args:
            vectors_a: Query rows in a common descriptor space.
            vectors_b: Reference rows with the same features and preprocessing.
            descriptor_kind: 'binary', 'count' (including Morgan counts), or
                'embedding' (including autoencoder latent vectors). Determine
                this from provenance, never dtype alone.
            metric: 'auto' chooses Tanimoto for binary/count fingerprints and
                Euclidean distance for embeddings; 'cosine' is also supported
                for embeddings. Tanimoto uses dot(a,b)/(a.a+b.b-a.b), which
                differs from RDKit's min/max count Tanimoto. Two zero fingerprints
                have similarity 1; zero embeddings are invalid for cosine.

        Returns:
            ``matrix[i][j]``, metric, descriptor_kind, and is_distance. Euclidean
            is a distance (lower is closer); Tanimoto and cosine are similarities
            (higher is closer). Cosine ranges from -1 to 1. Use bounded blocks
            and save large matrices as artifacts rather than inlining them.
        """
        kind = descriptor_kind.lower()
        allowed = {
            "binary": ("tanimoto",),
            "count": ("tanimoto",),
            "embedding": ("euclidean", "cosine"),
        }
        if kind not in allowed:
            raise SimilarityError("descriptor_kind must be binary, count, or embedding")
        metric = metric.lower()
        if metric == "auto":
            metric = allowed[kind][0]
        if metric not in allowed[kind]:
            raise SimilarityError(f"Use {allowed[kind]} for {kind} descriptors, not {metric}")
        try:
            a = np.ascontiguousarray(vectors_a, dtype=np.float64)
            b = np.ascontiguousarray(vectors_b, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SimilarityError("Descriptors must be rectangular numeric matrices") from exc
        if a.ndim != 2 or b.ndim != 2 or not a.size or not b.size or a.shape[1] != b.shape[1]:
            raise SimilarityError("Descriptors must be nonempty 2D matrices with equal widths")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise SimilarityError("Descriptors must contain only finite values")
        if kind != "embedding" and (np.any(a < 0) or np.any(b < 0)):
            raise SimilarityError("Fingerprint counts must be nonnegative")
        if kind == "binary" and (np.any((a != 0) & (a != 1)) or np.any((b != 0) & (b != 1))):
            raise SimilarityError("Binary fingerprints must contain only 0 and 1")
        if metric == "tanimoto":
            scores = _tanimoto_matrix(a, b)
        elif metric == "cosine":
            if np.any(~a.any(axis=1)) or np.any(~b.any(axis=1)):
                raise SimilarityError("Cosine similarity is undefined for zero vectors")
            scores = 1.0 - cdist(a, b, metric="cosine")
        else:
            scores = cdist(a, b, metric="euclidean")
        if not np.isfinite(scores).all():
            raise SimilarityError("Descriptor magnitudes caused numerical overflow")
        return {
            "matrix": scores.tolist(),
            "metric": metric,
            "descriptor_kind": kind,
            "is_distance": metric == "euclidean",
        }

    def calculate_tanimoto_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate Tanimoto similarity between two molecules' fingerprints.

        Morgan count uses dot-product Tanimoto (a.b)/(a.a+b.b-a.b), not
        RDKit's min/max count convention. For stored descriptors, use
        calculate_similarity_matrix to preserve the original fingerprint width.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: 'rdkit', 'morgan' (binary), 'maccs', or 'morgan_count'
                (radius 2, 2048 counts).

        Returns:
            Tanimoto similarity score (0.0 to 1.0)

        Raises:
            SimilarityError: If calculation fails
        """
        try:
            if fp_type.lower() == "morgan_count":
                a, b = (calc_morgan_fp(smiles, 2048) for smiles in (smiles1, smiles2))
                if a is None or b is None:
                    raise SimilarityError("Invalid SMILES for Morgan count fingerprint")
                return self.calculate_similarity_matrix([a], [b], "count")["matrix"][0][0]
            # Generate fingerprints
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            # Calculate Tanimoto similarity
            similarity = DataStructs.FingerprintSimilarity(fp1, fp2)

            logger.debug(f"Tanimoto similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Tanimoto similarity: {e}")
            raise SimilarityError(f"Failed to calculate Tanimoto similarity: {e}") from e

    def calculate_dice_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate Dice similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Dice similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.DiceSimilarity(fp1, fp2)

            logger.debug(f"Dice similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Dice similarity: {e}")
            raise SimilarityError(f"Failed to calculate Dice similarity: {e}") from e

    def calculate_tversky_similarity(
        self,
        smiles1: str,
        smiles2: str,
        alpha: float = 0.5,
        beta: float = 0.5,
        fp_type: str = "rdkit",
    ) -> float:
        """
        Calculate Tversky similarity between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            alpha: Weight for first molecule
            beta: Weight for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Tversky similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.TverskySimilarity(fp1, fp2, alpha, beta)

            logger.debug(f"Tversky similarity ({fp_type}, α={alpha}, β={beta}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating Tversky similarity: {e}")
            raise SimilarityError(f"Failed to calculate Tversky similarity: {e}") from e

    def calculate_cosine_similarity(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> float:
        """
        Calculate cosine similarity between two molecules' binary fingerprints.

        Prefer Tanimoto for fingerprints. For embeddings, pass the actual vectors
        to calculate_similarity_matrix with descriptor_kind='embedding'.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')

        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            similarity = DataStructs.CosineSimilarity(fp1, fp2)

            logger.debug(f"Cosine similarity ({fp_type}): {similarity:.4f}")
            return similarity

        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            raise SimilarityError(f"Failed to calculate cosine similarity: {e}") from e

    def calculate_euclidean_distance(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit", normalize: bool = True
    ) -> float:
        """
        Calculate Euclidean distance between two molecules' binary fingerprints.

        Prefer Tanimoto for fingerprints. For embeddings, pass the actual vectors
        to calculate_similarity_matrix with descriptor_kind='embedding'.

        Note: Euclidean distance is a dissimilarity metric - smaller values
        indicate more similar molecules. Unlike similarity metrics, the range
        is not bounded to [0, 1] unless normalized.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Type of fingerprint ('rdkit', 'morgan', 'maccs')
            normalize: If True, divide by sqrt(fingerprint length). RDKit
                fingerprints of different lengths are folded to a common length.

        Returns:
            Euclidean distance (lower values = more similar molecules)
        """
        try:
            fp1 = self.generate_fingerprint(smiles1, fp_type)
            fp2 = self.generate_fingerprint(smiles2, fp_type)

            # For binary vectors, squared L2 distance is the XOR bit count.
            # FingerprintSimilarity handles RDKit's variable-length folding.
            distance = DataStructs.FingerprintSimilarity(
                fp1,
                fp2,
                metric=lambda a, b: ((a ^ b).GetNumOnBits() / (a.GetNumBits() if normalize else 1))
                ** 0.5,
            )

            logger.debug(f"Euclidean distance ({fp_type}, normalized={normalize}): {distance:.4f}")
            return distance

        except Exception as e:
            logger.error(f"Error calculating Euclidean distance: {e}")
            raise SimilarityError(f"Failed to calculate Euclidean distance: {e}") from e

    def calculate_all_similarities(
        self, smiles1: str, smiles2: str, fp_type: str = "rdkit"
    ) -> Dict[str, float]:
        """
        Calculate all available similarity metrics between two molecules.

        Args:
            smiles1: SMILES string for first molecule
            smiles2: SMILES string for second molecule
            fp_type: Binary fingerprint type ('rdkit', 'morgan', 'maccs').
                Use calculate_tanimoto_similarity for Morgan counts.

        Returns:
            Dictionary containing all similarity scores and distance metrics
        """
        try:
            similarities = {
                "tanimoto": self.calculate_tanimoto_similarity(smiles1, smiles2, fp_type),
                "dice": self.calculate_dice_similarity(smiles1, smiles2, fp_type),
                "cosine": self.calculate_cosine_similarity(smiles1, smiles2, fp_type),
                "euclidean_distance": self.calculate_euclidean_distance(
                    smiles1, smiles2, fp_type, normalize=True
                ),
                "tversky_0.5_0.5": self.calculate_tversky_similarity(
                    smiles1, smiles2, 0.5, 0.5, fp_type
                ),
                "tversky_1.0_1.0": self.calculate_tversky_similarity(
                    smiles1, smiles2, 1.0, 1.0, fp_type
                ),
            }

            logger.debug(f"All similarities calculated for {fp_type} fingerprints")
            return similarities

        except Exception as e:
            logger.error(f"Error calculating all similarities: {e}")
            raise SimilarityError(f"Failed to calculate all similarities: {e}") from e

    def find_most_similar(
        self,
        query_smiles: str,
        reference_smiles: List[str],
        metric: str = "tanimoto",
        fp_type: str = "rdkit",
        top_k: int = 5,
    ) -> List[Tuple[str, float, int]]:
        """
        Find the most similar molecules to a query molecule.

        Args:
            query_smiles: Query molecule SMILES string
            reference_smiles: List of reference molecule SMILES strings
            metric: Similarity metric to use ('tanimoto', 'dice', 'cosine', 'euclidean')
            fp_type: 'rdkit', 'morgan', 'maccs', or 'morgan_count' with Tanimoto.
            top_k: Number of top similar molecules to return

        Returns:
            List of tuples (smiles, similarity_score, index) sorted by similarity
        """
        try:
            similarities = []
            is_distance_metric = metric.lower() == "euclidean"

            for i, ref_smiles in enumerate(reference_smiles):
                try:
                    if metric.lower() == "tanimoto":
                        sim = self.calculate_tanimoto_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "dice":
                        sim = self.calculate_dice_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "cosine":
                        sim = self.calculate_cosine_similarity(query_smiles, ref_smiles, fp_type)
                    elif metric.lower() == "euclidean":
                        sim = self.calculate_euclidean_distance(
                            query_smiles, ref_smiles, fp_type, normalize=True
                        )
                    else:
                        raise ValueError(f"Unsupported similarity metric: {metric}")

                    similarities.append((ref_smiles, sim, i))

                except Exception as e:
                    logger.warning(f"Failed to calculate similarity for molecule {i}: {e}")
                    continue

            # Sort by similarity (descending for similarity, ascending for distance)
            similarities.sort(key=lambda x: x[1], reverse=not is_distance_metric)

            logger.debug(f"Found {len(similarities)} similar molecules, returning top {top_k}")
            return similarities[:top_k]

        except Exception as e:
            logger.error(f"Error finding most similar molecules: {e}")
            raise SimilarityError(f"Failed to find most similar molecules: {e}") from e
