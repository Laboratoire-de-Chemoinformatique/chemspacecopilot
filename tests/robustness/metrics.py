#!/usr/bin/env python
# coding: utf-8
"""
Robustness metrics calculation and aggregation.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RobustnessMetrics:
    """Calculate aggregate robustness metrics and generate reports."""

    def __init__(
        self,
        weights: Dict[str, float] = None,
        thresholds: Dict[str, float] = None,
    ):
        """
        Initialize robustness metrics calculator.

        Args:
            weights: Custom weights for different metric categories
            thresholds: Custom thresholds for pass/fail criteria
        """
        # Default weights
        self.weights = weights or {
            "data_similarity": 0.4,
            "semantic_similarity": 0.3,
            "process_consistency": 0.2,
            "visual_similarity": 0.1,
        }

        # Default thresholds
        self.thresholds = thresholds or {
            "excellent": 0.90,
            "good": 0.80,
            "acceptable": 0.70,
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def calculate_robustness_score(self, comparison_results: Dict) -> float:
        """
        Aggregate all metrics into single robustness score.

        Args:
            comparison_results: Dictionary with comparison results from all categories

        Returns:
            Robustness score between 0 and 1
        """
        scores = {}

        # Data similarity
        if "data" in comparison_results:
            data_metrics = comparison_results["data"]
            scores["data_similarity"] = np.mean(
                [
                    data_metrics.get("row_jaccard", 0),
                    data_metrics.get("column_match", 0),
                    1 - data_metrics.get("value_stability", 1),  # Convert stability to similarity
                ]
            )

        # Semantic similarity
        if "text" in comparison_results:
            text_metrics = comparison_results["text"]
            scores["semantic_similarity"] = np.mean(
                [
                    text_metrics.get("semantic_similarity", 0),
                    text_metrics.get("entity_overlap", 0),
                    text_metrics.get("numeric_consistency", 0),
                ]
            )

        # Process consistency
        if "process" in comparison_results:
            process_metrics = comparison_results["process"]
            scores["process_consistency"] = np.mean(
                [
                    process_metrics.get("completion_rate", 0),
                    process_metrics.get("tool_sequence_similarity", 0),
                ]
            )

        # Visual similarity
        if "visual" in comparison_results:
            visual_metrics = comparison_results["visual"]
            scores["visual_similarity"] = np.mean(
                [
                    visual_metrics.get("perceptual_hash", 0),
                    visual_metrics.get("ssim", 0),
                ]
            )

        # Calculate weighted score
        robustness_score = sum(scores.get(key, 0) * weight for key, weight in self.weights.items())

        logger.info(f"Calculated robustness score: {robustness_score:.3f}")
        return robustness_score

    def identify_outliers(
        self, outputs: List[Any], comparison_results: Dict = None, threshold: float = 2.0
    ) -> List[int]:
        """
        Identify which prompt variations produced outlier results.

        Uses multi-metric outlier detection based on:
        1. Text semantic similarity (embedding distance from centroid)
        2. Numeric output characteristics (e.g., row counts, file counts)
        3. Process completion status (failed when others succeeded)
        4. Response length (character count deviation)

        Args:
            outputs: List of output dictionaries from each run
            comparison_results: Optional comparison results with additional metrics
            threshold: Z-score threshold for outlier detection (default: 2.0)

        Returns:
            List of indices corresponding to outlier runs
        """
        if len(outputs) < 3:
            # Need at least 3 samples for meaningful outlier detection
            return []

        outlier_indices = set()

        # 1. Text semantic similarity outliers
        try:
            texts = [o.get("response", "") for o in outputs if o.get("response")]
            if len(texts) >= 3:
                from sentence_transformers import SentenceTransformer

                # Use lightweight model for outlier detection
                model = SentenceTransformer("all-MiniLM-L6-v2")
                embeddings = model.encode(texts)

                # Calculate centroid and distances
                centroid = np.mean(embeddings, axis=0)
                distances = [np.linalg.norm(emb - centroid) for emb in embeddings]

                # Detect outliers using Z-score
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)

                if std_dist > 0:
                    z_scores = [(d - mean_dist) / std_dist for d in distances]
                    semantic_outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
                    outlier_indices.update(semantic_outliers)
                    if semantic_outliers:
                        logger.info(f"Semantic outliers detected at indices: {semantic_outliers}")

        except Exception as e:
            logger.debug(f"Semantic outlier detection failed: {e}")

        # 2. Numeric output outliers (row counts, file counts, etc.)
        try:
            # Extract numeric features from outputs
            numeric_features = []
            for output in outputs:
                features = {
                    "response_length": len(output.get("response", "")),
                    "file_count": len(output.get("generated_files", {})),
                    "s3_file_count": len(output.get("s3_files", {})),
                }

                # Try to extract row count from response
                response = output.get("response", "")
                import re

                count_match = re.search(r"(\d+)\s*(?:rows?|records?|compounds?)", response)
                if count_match:
                    features["row_count"] = int(count_match.group(1))

                numeric_features.append(features)

            # Detect outliers for each feature
            for feature_name in numeric_features[0].keys():
                values = [f.get(feature_name, 0) for f in numeric_features]

                if len(values) >= 3:
                    mean_val = np.mean(values)
                    std_val = np.std(values)

                    if std_val > 0:
                        z_scores = [(v - mean_val) / std_val for v in values]
                        feature_outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
                        if feature_outliers:
                            outlier_indices.update(feature_outliers)
                            logger.info(
                                f"Numeric outliers in {feature_name} at indices: {feature_outliers}"
                            )

        except Exception as e:
            logger.debug(f"Numeric outlier detection failed: {e}")

        # 3. Process completion outliers (failed when others succeeded)
        try:
            statuses = [o.get("status", "unknown") for o in outputs]
            success_count = sum(1 for s in statuses if s == "success")
            failure_count = len(statuses) - success_count

            # If most runs succeeded, flag failures as outliers
            if success_count > failure_count:
                failure_indices = [i for i, s in enumerate(statuses) if s != "success"]
                outlier_indices.update(failure_indices)
                if failure_indices:
                    logger.info(f"Process failure outliers at indices: {failure_indices}")

        except Exception as e:
            logger.debug(f"Process outlier detection failed: {e}")

        # 4. Response structure outliers (e.g., missing expected sections)
        try:
            # Check for presence of expected patterns (files, numbers, confirmations)
            pattern_scores = []
            for output in outputs:
                response = output.get("response", "").lower()
                score = 0
                score += (
                    1 if any(word in response for word in ["saved", "downloaded", "created"]) else 0
                )
                score += 1 if re.search(r"\d+", response) else 0
                score += 1 if any(ext in response for ext in [".csv", ".sdf", "s3://"]) else 0
                pattern_scores.append(score)

            if len(pattern_scores) >= 3:
                mean_score = np.mean(pattern_scores)
                std_score = np.std(pattern_scores)

                if std_score > 0:
                    z_scores = [(s - mean_score) / std_score for s in pattern_scores]
                    structure_outliers = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
                    if structure_outliers:
                        outlier_indices.update(structure_outliers)
                        logger.info(f"Response structure outliers at indices: {structure_outliers}")

        except Exception as e:
            logger.debug(f"Response structure outlier detection failed: {e}")

        outliers = sorted(outlier_indices)
        logger.info(f"Total outliers identified: {len(outliers)} out of {len(outputs)} runs")
        return outliers

    def generate_report(self, results: Dict) -> str:
        """
        Generate markdown report of robustness analysis.

        Args:
            results: Dictionary containing score, comparisons, and outliers

        Returns:
            Markdown-formatted report string
        """
        score = results.get("score", 0.0)
        comparisons = results.get("comparisons", {})
        outliers = results.get("outliers", [])

        # Determine status
        if score >= self.thresholds["excellent"]:
            status = "✅ EXCELLENT"
        elif score >= self.thresholds["good"]:
            status = "✅ GOOD"
        elif score >= self.thresholds["acceptable"]:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ CONCERNING"

        report = f"""# Prompt Robustness Test Report

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Overall Robustness Score:** {score:.3f} / 1.00 {status}

## Summary

- **Status:** {status}
- **Outliers Detected:** {len(outliers)}
- **Completion Rate:** {comparisons.get('process', {}).get('completion_rate', 'N/A')}

## Component Scores

"""

        # Data similarity
        if "data" in comparisons:
            data = comparisons["data"]
            report += f"""### Data Consistency
- **Row Jaccard:** {data.get('row_jaccard', 0):.3f}
- **Column Match:** {data.get('column_match', 0):.3f}
- **Value Stability:** {data.get('value_stability', 0):.3f}
- **Distribution KS:** {data.get('distribution_ks', 0):.3f}

"""

        # Text similarity
        if "text" in comparisons:
            text = comparisons["text"]
            report += f"""### Text Response Consistency
- **Semantic Similarity:** {text.get('semantic_similarity', 0):.3f}
- **Entity Overlap:** {text.get('entity_overlap', 0):.3f}
- **Numeric Consistency:** {text.get('numeric_consistency', 0):.3f}
- **Structural Match:** {text.get('structural_match', 0):.3f}

"""

        # GTM models
        if "gtm" in comparisons:
            gtm = comparisons["gtm"]
            report += f"""### GTM Model Consistency
- **Structure Match:** {gtm.get('structure_match', 0):.3f}
- **Projection Correlation:** {gtm.get('projection_correlation', 0):.3f}
- **Density Correlation:** {gtm.get('density_correlation', 0):.3f}

"""

        # Visual
        if "visual" in comparisons:
            visual = comparisons["visual"]
            report += f"""### Visualization Consistency
- **Perceptual Hash:** {visual.get('perceptual_hash', 0):.3f}
- **SSIM:** {visual.get('ssim', 0):.3f}

"""

        # Outliers
        if outliers:
            report += f"""## Outliers

{len(outliers)} outlier run(s) detected: {outliers}

These variations produced results significantly different from others.

"""

        # Recommendations
        report += self._generate_recommendations(score, comparisons, outliers)

        return report

    def _generate_recommendations(
        self, score: float, comparisons: Dict, outliers: List[int]
    ) -> str:
        """Generate recommendations based on results."""
        recommendations = ["## Recommendations\n"]

        if score >= self.thresholds["excellent"]:
            recommendations.append(
                "✅ Excellent robustness. The system handles prompt variations very well."
            )
        elif score >= self.thresholds["good"]:
            recommendations.append(
                "✅ Good robustness. Minor inconsistencies but acceptable for production."
            )
        elif score >= self.thresholds["acceptable"]:
            recommendations.append(
                "⚠️  Acceptable robustness but room for improvement. Monitor closely."
            )
        else:
            recommendations.append(
                "❌ Concerning robustness. Significant inconsistencies detected."
            )

        # Specific recommendations
        if "data" in comparisons:
            data = comparisons["data"]
            if data.get("row_jaccard", 1) < 0.95:
                recommendations.append(
                    "- ⚠️  Dataset consistency below threshold. Check data fetching logic."
                )
            if data.get("value_stability", 0) > 0.05:
                recommendations.append("- ⚠️  High value variability. Check numeric processing.")

        if "text" in comparisons:
            text = comparisons["text"]
            if text.get("semantic_similarity", 1) < 0.75:
                recommendations.append(
                    "- ⚠️  Low semantic similarity in responses. Review agent instructions."
                )

        if outliers:
            recommendations.append(
                f"- ⚠️  {len(outliers)} outlier(s) detected. Investigate these specific prompts."
            )

        return "\n".join(recommendations) + "\n"

    def get_rating(self, score: float) -> str:
        """Get human-readable rating for a score."""
        if score >= self.thresholds["excellent"]:
            return "Excellent"
        elif score >= self.thresholds["good"]:
            return "Good"
        elif score >= self.thresholds["acceptable"]:
            return "Acceptable"
        else:
            return "Concerning"
