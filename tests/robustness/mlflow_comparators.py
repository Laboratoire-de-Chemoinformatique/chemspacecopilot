"""Extended comparators with MLflow logging capabilities."""

import logging
from typing import Any, Dict

from comparators import OutputComparator

logger = logging.getLogger(__name__)


class MLflowOutputComparator(OutputComparator):
    """Output comparator with MLflow integration.

    Extends OutputComparator to automatically log comparison metrics
    and artifacts to MLflow during comparison operations.
    """

    def __init__(self, log_to_mlflow: bool = True):
        """Initialize comparator.

        Args:
            log_to_mlflow: Whether to log metrics to MLflow
        """
        super().__init__()
        self.log_to_mlflow = log_to_mlflow
        self._tracker = None

        if self.log_to_mlflow:
            self._initialize_mlflow()

    def _initialize_mlflow(self):
        """Initialize MLflow tracker."""
        try:
            from cs_copilot.tracking import get_tracker

            self._tracker = get_tracker()

            if not self._tracker.is_enabled():
                logger.debug("MLflow tracking disabled. Comparisons will not be logged.")
                self.log_to_mlflow = False

        except ImportError:
            logger.debug("MLflow not available. Comparisons will not be logged.")
            self.log_to_mlflow = False

    def compare_dataframes(self, df1: Any, df2: Any) -> Dict[str, Any]:
        """Compare two DataFrames and log metrics to MLflow.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Comparison results dictionary
        """
        result = super().compare_dataframes(df1, df2)

        if self.log_to_mlflow and self._tracker:
            self._log_dataframe_comparison(result)

        return result

    def compare_text(self, text1: str, text2: str) -> Dict[str, Any]:
        """Compare two text strings and log metrics to MLflow.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Comparison results dictionary
        """
        result = super().compare_text(text1, text2)

        if self.log_to_mlflow and self._tracker:
            self._log_text_comparison(result)

        return result

    def compare_outputs(self, output1: Any, output2: Any) -> Dict[str, Any]:
        """Compare two outputs and log comprehensive metrics to MLflow.

        Args:
            output1: First output
            output2: Second output

        Returns:
            Comparison results dictionary
        """
        result = super().compare_outputs(output1, output2)

        if self.log_to_mlflow and self._tracker:
            self._log_output_comparison(result)

        return result

    def _log_dataframe_comparison(self, result: Dict[str, Any]):
        """Log DataFrame comparison metrics to MLflow.

        Args:
            result: Comparison results
        """
        metrics = {}

        if "shape_match" in result:
            metrics["df_shape_match"] = 1.0 if result["shape_match"] else 0.0

        if "columns_match" in result:
            metrics["df_columns_match"] = 1.0 if result["columns_match"] else 0.0

        if "data_similarity" in result:
            metrics["df_data_similarity"] = float(result["data_similarity"])

        if "row_count_diff" in result:
            metrics["df_row_count_diff"] = float(result["row_count_diff"])

        if "column_count_diff" in result:
            metrics["df_column_count_diff"] = float(result["column_count_diff"])

        if metrics:
            self._tracker.log_metrics(metrics)

    def _log_text_comparison(self, result: Dict[str, Any]):
        """Log text comparison metrics to MLflow.

        Args:
            result: Comparison results
        """
        metrics = {}

        if "exact_match" in result:
            metrics["text_exact_match"] = 1.0 if result["exact_match"] else 0.0

        if "semantic_similarity" in result:
            metrics["text_semantic_similarity"] = float(result["semantic_similarity"])

        if "length_diff" in result:
            metrics["text_length_diff"] = float(result["length_diff"])

        if "word_overlap" in result:
            metrics["text_word_overlap"] = float(result["word_overlap"])

        if metrics:
            self._tracker.log_metrics(metrics)

    def _log_output_comparison(self, result: Dict[str, Any]):
        """Log comprehensive output comparison metrics to MLflow.

        Args:
            result: Comparison results
        """
        metrics = {}

        # Overall similarity
        if "overall_similarity" in result:
            metrics["overall_similarity"] = float(result["overall_similarity"])

        # Data similarity
        if "data_similarity" in result:
            metrics["data_similarity"] = float(result["data_similarity"])

        # Semantic similarity
        if "semantic_similarity" in result:
            metrics["semantic_similarity"] = float(result["semantic_similarity"])

        # Process consistency
        if "process_consistency" in result:
            metrics["process_consistency"] = float(result["process_consistency"])

        # Visual similarity
        if "visual_similarity" in result:
            metrics["visual_similarity"] = float(result["visual_similarity"])

        # Type match
        if "type_match" in result:
            metrics["type_match"] = 1.0 if result["type_match"] else 0.0

        if metrics:
            self._tracker.log_metrics(metrics)

        # Log additional details as parameters if needed
        if "comparison_details" in result:
            details = result["comparison_details"]
            params = {}

            if isinstance(details, dict):
                for key, value in list(details.items())[:10]:  # Limit to first 10 items
                    if isinstance(value, (str, int, float, bool)):
                        params[f"detail_{key}"] = str(value)[:500]

            if params:
                self._tracker.log_params(params)


def create_mlflow_comparator(enable_mlflow: bool = True) -> OutputComparator:
    """Create an output comparator with optional MLflow logging.

    Args:
        enable_mlflow: Whether to enable MLflow logging

    Returns:
        OutputComparator instance (MLflow-enabled if available and requested)
    """
    if enable_mlflow:
        try:
            return MLflowOutputComparator(log_to_mlflow=True)
        except Exception as e:
            logger.warning(f"Failed to create MLflow comparator: {e}. Using standard comparator.")
            return OutputComparator()
    else:
        return OutputComparator()
