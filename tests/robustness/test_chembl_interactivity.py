#!/usr/bin/env python
# coding: utf-8
"""
ChEMBL clarification flow robustness tests.

Verifies that ambiguous prompts trigger clarification questions and that
follow-up clarifications lead to data retrieval.

Can be run as:
1. pytest test: `uv run pytest tests/robustness/test_chembl_interactivity.py -v`
2. Standalone script: `uv run python tests/robustness/test_chembl_interactivity.py`
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from dotenv import load_dotenv

try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import numpy as np

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None

import pandas as pd

from cs_copilot.model_config import run_with_retry  # noqa: E402
from cs_copilot.storage import S3  # noqa: E402
from cs_copilot.utils.logging import get_logger  # noqa: E402

# Add robustness directory to path for clean imports (works in both pytest and standalone)
_robustness_dir = Path(__file__).parent
if str(_robustness_dir) not in sys.path:
    sys.path.insert(0, str(_robustness_dir))

from prompt_variations import PromptVariationGenerator  # noqa: E402
from test_utils import ResponseParser, create_agent_team_factory  # noqa: E402

logger = get_logger(__name__)
load_dotenv()

# Flag to determine if running in pytest mode
try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    logger.warning("pytest not available - running in standalone mode only")


def _load_model_from_config():
    """Load model using centralized .modelconf / env var configuration."""
    try:
        from cs_copilot.model_config import load_model_from_config

        return load_model_from_config()
    except (FileNotFoundError, ImportError, ValueError) as e:
        if PYTEST_AVAILABLE:
            pytest.skip(str(e))
        else:
            raise


def _setup_s3():
    """Setup S3 configuration and check availability."""
    from cs_copilot.storage import get_s3_config, is_s3_enabled

    if not is_s3_enabled():
        logger.warning("S3 not enabled - files will be stored locally")
        return None

    s3_config = get_s3_config()
    logger.info(f"S3 enabled - Bucket: {s3_config.bucket_name}")
    return s3_config


def _create_agent_team_factory():
    """Create a factory function for agent teams with memory disabled for isolation (wrapper for compatibility)."""
    model = _load_model_from_config()
    factory = create_agent_team_factory(model)

    # Wrap to add default parameters specific to this test
    def _create_team(**overrides):
        defaults = {
            "show_members_responses": False,
        }
        defaults.update(overrides)
        return factory(**defaults)

    return _create_team


def _check_contains_phrases(
    text: str, phrases: Iterable[str], require_all: bool = True
) -> tuple[bool, list[str]]:
    """Check if text contains required phrases.

    Args:
        text: Text to search in
        phrases: Phrases to search for
        require_all: If True, all phrases must be present. If False, at least one must be present.

    Returns:
        (success, missing_phrases): success is True if validation passes,
                                   missing_phrases lists what wasn't found
    """
    lower = text.lower()
    missing = [phrase for phrase in phrases if phrase.lower() not in lower]

    if require_all:
        # All phrases must be present
        return len(missing) == 0, missing
    else:
        # At least one phrase must be present (i.e., not all can be missing)
        at_least_one_found = len(missing) < len(list(phrases))
        return at_least_one_found, missing if not at_least_one_found else []


def _get_session_state_with_retry(agent_team, max_retries=3, delay=1.0):
    """Get session state with retry logic for database locks.

    When memory is disabled (db=None), session state is still available
    but stored in-memory in the agent object rather than the database.
    """
    import time

    for attempt in range(max_retries):
        try:
            # Try to get session state from the team
            return agent_team.get_session_state()
        except Exception as e:
            error_str = str(e)

            # If session not found and memory is disabled, return empty dict
            # (session state exists in-memory but may not be accessible via API)
            if "Session not found" in error_str:
                # When memory is disabled, session state is in-memory only
                # Access it directly from the agent's session_state attribute
                if hasattr(agent_team, "session_state"):
                    return agent_team.session_state or {}
                # Or check if it's stored in the run result
                return {}

            # Database lock - retry
            if "database is locked" in error_str:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Database lock issue (attempt {attempt + 1}/{max_retries}), retrying..."
                    )
                    time.sleep(delay * (attempt + 1))  # Exponential backoff
                    continue

            # Other errors - raise
            raise

    return {}


def _get_dataset_path(session_state: Optional[dict]) -> Optional[str]:
    """Extract dataset path from session state."""
    if not isinstance(session_state, dict):
        return None
    data_file_paths = session_state.get("data_file_paths")
    if isinstance(data_file_paths, dict):
        return data_file_paths.get("dataset_path")
    return None


def _check_successful_retrieval(response_text: str) -> bool:
    """
    Check if response indicates successful data retrieval (wrapper for compatibility).

    Uses ResponseParser.check_success() from shared test utilities.
    """
    return ResponseParser.check_success(response_text)


CLARIFICATION_PHRASES = ("target", "organism", "assay")


def _load_dataset(dataset_path: str) -> Optional[pd.DataFrame]:
    """Load a dataset from S3 or local storage.

    Args:
        dataset_path: Path to the dataset (S3 URL or local path)

    Returns:
        DataFrame if successfully loaded, None otherwise
    """
    if not dataset_path:
        return None

    try:
        import pandas as pd

        # Use S3 client to load (works for both S3 and local paths)
        with S3.open(dataset_path, "r") as f:
            df = pd.read_csv(f)

        logger.debug(
            f"Loaded dataset from {dataset_path}: {len(df)} rows, {len(df.columns)} columns"
        )
        return df
    except Exception as e:
        logger.warning(f"Failed to load dataset from {dataset_path}: {e}")
        return None


def _compare_datasets_for_prompt_group(
    results: list[PromptRunResult], prompt_index: int, enable_comparison: bool = True
) -> dict[int, dict]:
    """Compare datasets across all variations of a single prompt using activity_id matching
    and detailed content comparison.

    Args:
        results: All prompt results
        prompt_index: The prompt index to compare
        enable_comparison: Whether to perform comparison

    Returns:
        Dict mapping variation_index to comparison metrics
    """
    if not enable_comparison:
        return {}

    # Get all results for this prompt
    prompt_results = [r for r in results if r.prompt_index == prompt_index]

    # Filter to only immediate retrieval prompts (clarification prompts shouldn't have datasets)
    retrieval_results = [r for r in prompt_results if not r.requires_clarification]

    if len(retrieval_results) < 2:
        logger.debug(
            f"Prompt {prompt_index}: Not enough retrieval results to compare ({len(retrieval_results)})"
        )
        return {}

    # Load all datasets
    datasets = []
    valid_results = []
    for r in retrieval_results:
        if r.dataset_path:
            df = _load_dataset(r.dataset_path)
            if df is not None:
                datasets.append(df)
                valid_results.append(r)
                # Update row count in result
                r.dataset_row_count = len(df)

    if len(datasets) < 2:
        logger.debug(
            f"Prompt {prompt_index}: Not enough valid datasets to compare ({len(datasets)})"
        )
        return {}

    # Extract activity_id sets from each dataset
    activity_id_sets = []
    for df in datasets:
        if "activity_id" not in df.columns:
            logger.warning(
                f"Prompt {prompt_index}: Dataset missing 'activity_id' column. Skipping comparison."
            )
            return {}

        # Handle empty datasets
        if len(df) == 0:
            logger.warning(f"Prompt {prompt_index}: Empty dataset found. Marking as mismatch.")
            activity_id_sets.append(set())
        else:
            activity_id_sets.append(set(df["activity_id"]))

    # Check if all activity_id sets are equal
    # Convert sets to frozensets for set comparison
    frozen_sets = [frozenset(s) for s in activity_id_sets]
    all_match = len(set(frozen_sets)) == 1

    # Calculate consistency score: 1.0 if exact match, 0.0 otherwise
    consistency_score = 1.0 if all_match else 0.0

    # Store metrics
    row_counts = [len(df) for df in datasets]
    num_activities = len(activity_id_sets[0]) if all_match and activity_id_sets else 0

    # Detailed content comparison
    content_comparison = _compare_dataset_content(datasets)

    comparison_metrics = {
        "consistency_score": consistency_score,
        "activity_id_match": all_match,
        "row_counts": row_counts,
        "num_activities": num_activities,
        "columns_match": content_comparison.get("columns_match", True),
        "column_names": content_comparison.get("column_names", []),
        "dtypes_match": content_comparison.get("dtypes_match", True),
        "sample_values_match": content_comparison.get("sample_values_match", {}),
    }

    logger.info(
        f"Prompt {prompt_index} dataset comparison: "
        f"activity_id_match={all_match}, "
        f"consistency={consistency_score:.3f}, "
        f"rows={row_counts}, "
        f"num_activities={num_activities}, "
        f"columns_match={content_comparison.get('columns_match', True)}, "
        f"dtypes_match={content_comparison.get('dtypes_match', True)}"
    )

    # Assign metrics to each variation
    variation_metrics = {}
    for r in valid_results:
        variation_metrics[r.variation_index] = comparison_metrics.copy()
        r.dataset_comparison_metrics = comparison_metrics.copy()

    return variation_metrics


def _compare_dataset_content(datasets: list[pd.DataFrame]) -> dict:
    """Compare detailed content of datasets.

    Args:
        datasets: List of DataFrames to compare

    Returns:
        Dict with comparison metrics:
        - columns_match: Whether all datasets have same columns
        - column_names: List of column names from first dataset
        - dtypes_match: Whether data types match
        - sample_values_match: Dict of column -> whether sample values match
    """
    if not datasets:
        return {}

    first_df = datasets[0]

    # Compare column names
    column_sets = [set(df.columns) for df in datasets]
    columns_match = all(cols == column_sets[0] for cols in column_sets)

    # Compare data types
    dtypes_match = True
    if columns_match:
        first_dtypes = first_df.dtypes.to_dict()
        for df in datasets[1:]:
            df_dtypes = df.dtypes.to_dict()
            if df_dtypes != first_dtypes:
                dtypes_match = False
                break

    # Compare sample values for key columns
    sample_values_match = {}
    key_columns = ["activity_id", "smiles", "standard_value", "standard_units"]
    available_key_cols = [col for col in key_columns if col in first_df.columns]

    for col in available_key_cols:
        try:
            # Get sample values from each dataset (first 5 rows when sorted)
            samples = []
            for df in datasets:
                if col in df.columns and len(df) > 0:
                    sample = df[col].head(5).tolist()
                    samples.append(sample)

            # Check if samples are identical
            if samples:
                first_sample = samples[0]
                sample_values_match[col] = all(s == first_sample for s in samples)
            else:
                sample_values_match[col] = True
        except Exception as e:
            logger.debug(f"Could not compare sample values for {col}: {e}")
            sample_values_match[col] = None

    return {
        "columns_match": columns_match,
        "column_names": list(first_df.columns),
        "dtypes_match": dtypes_match,
        "sample_values_match": sample_values_match,
    }


# ============================================================================
# Test functions that can run in both pytest and standalone modes
# ============================================================================


@dataclass
class PromptRunResult:
    """Container for a single prompt execution."""

    prompt_index: int
    variation_index: int
    prompt: str
    requires_clarification: bool
    response_text: str
    dataset_path: Optional[str]
    success: bool
    detail: str
    tool_calls: Optional[list] = None  # List of tool calls made during execution
    execution_time: Optional[float] = None  # Execution time in seconds
    session_state: Optional[dict] = None  # Full session state after execution
    error_messages: Optional[list] = None  # Any errors or warnings encountered
    dataset_row_count: Optional[int] = None  # Number of rows in the dataset
    dataset_comparison_metrics: Optional[dict] = None  # Comparison metrics vs other variations


def _run_prompt(
    agent_team_factory,
    prompt: str,
    requires_clarification: bool,
    prompt_index: int,
    variation_index: int,
    s3_session_prefix: Optional[str] = None,
    verbose: bool = True,
) -> PromptRunResult:
    """Execute a prompt and validate whether clarification was requested.

    Creates a fresh agent team for each run to ensure complete session isolation.
    """
    import time

    if verbose:
        logger.info(f"[PROMPT]\n{prompt}\n")

    # Setup S3 session isolation if needed
    if s3_session_prefix:
        from cs_copilot.storage.client import S3 as S3Client

        original_prefix = S3Client.prefix
        S3Client.prefix = s3_session_prefix
        if verbose:
            logger.debug(f"S3 session prefix: {s3_session_prefix}")

    # Track execution details
    start_time = time.time()
    tool_calls = []
    error_messages = []

    try:
        # Create fresh agent team for this variation (ensures complete session isolation)
        agent_team = agent_team_factory()

        response = run_with_retry(agent_team, prompt, stream=False, max_retries=3)
        response_text = response.content or ""
        execution_time = time.time() - start_time

        if verbose:
            logger.info(f"[AGENT RESPONSE]\n{response_text}\n")
            logger.info(f"[EXECUTION TIME] {execution_time:.2f}s\n")

        # Extract tool calls from response if available
        if hasattr(response, "messages") and response.messages:
            for msg in response.messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls.append(
                            {
                                "tool_name": (
                                    getattr(tc, "function", {}).get("name", "unknown")
                                    if hasattr(tc, "function")
                                    else str(tc)
                                ),
                                "arguments": (
                                    getattr(tc, "function", {}).get("arguments", {})
                                    if hasattr(tc, "function")
                                    else {}
                                ),
                            }
                        )

        # Get session state - when memory is disabled, access from response/agent directly
        session_state = {}
        if hasattr(response, "session_state") and response.session_state:
            session_state = response.session_state
        elif hasattr(agent_team, "session_state") and agent_team.session_state:
            session_state = agent_team.session_state
        else:
            # Try the API method (works when memory is enabled)
            try:
                session_state = agent_team.get_session_state()
            except Exception as e:
                if "Session not found" not in str(e):
                    logger.warning(f"Could not get session state: {e}")
                    error_messages.append(f"Session state retrieval: {str(e)}")
                session_state = {}

        dataset_path = _get_dataset_path(session_state)
    except Exception as e:
        execution_time = time.time() - start_time
        error_messages.append(f"Execution error: {str(e)}")
        # Re-raise to let the caller handle it
        raise
    finally:
        # Restore original S3 prefix
        if s3_session_prefix:
            S3Client.prefix = original_prefix

    if requires_clarification:
        # Check if AT LEAST ONE clarification phrase is present (not all required)
        # This allows for prompts that partially specify parameters
        success, missing = _check_contains_phrases(
            response_text, CLARIFICATION_PHRASES, require_all=False
        )
        if not success:
            detail = f"No clarification phrases found. Expected at least one of: {list(CLARIFICATION_PHRASES)}"
            if verbose:
                logger.error(f"❌ FAILED: {detail}")
            return PromptRunResult(
                prompt_index,
                variation_index,
                prompt,
                requires_clarification,
                response_text,
                dataset_path,
                False,
                detail,
                tool_calls=tool_calls,
                execution_time=execution_time,
                session_state=session_state,
                error_messages=error_messages,
            )

        if dataset_path not in (None, ""):
            detail = f"Dataset path should not be set before clarification. Got: {dataset_path}"
            if verbose:
                logger.error(f"❌ FAILED: {detail}")
            return PromptRunResult(
                prompt_index,
                variation_index,
                prompt,
                requires_clarification,
                response_text,
                dataset_path,
                False,
                detail,
                tool_calls=tool_calls,
                execution_time=execution_time,
                session_state=session_state,
                error_messages=error_messages,
            )

        if verbose:
            logger.info("✓ Clarification requested and dataset path is unset (expected)")
        return PromptRunResult(
            prompt_index,
            variation_index,
            prompt,
            requires_clarification,
            response_text,
            dataset_path,
            True,
            "Clarification path validated",
            tool_calls=tool_calls,
            execution_time=execution_time,
            session_state=session_state,
            error_messages=error_messages,
        )

    # For fully specified prompts, validate successful data retrieval
    # Accept EITHER: (1) dataset_path is set, OR (2) response indicates successful retrieval
    has_dataset_path = bool(dataset_path)
    has_retrieval_indicators = _check_successful_retrieval(response_text)

    if not has_dataset_path and not has_retrieval_indicators:
        detail = (
            "Expected either dataset_path in session state OR retrieval success indicators "
            "in response for fully specified prompt"
        )
        if verbose:
            logger.error(f"❌ FAILED: {detail}")
        return PromptRunResult(
            prompt_index,
            variation_index,
            prompt,
            requires_clarification,
            response_text,
            dataset_path,
            False,
            detail,
            tool_calls=tool_calls,
            execution_time=execution_time,
            session_state=session_state,
            error_messages=error_messages,
        )

    # If we have a dataset path, validate its format
    if has_dataset_path:
        dataset_path_str = str(dataset_path)
        if not (dataset_path_str.startswith("s3://") or dataset_path_str.endswith(".csv")):
            detail = f"Unexpected dataset path format: {dataset_path_str}"
            if verbose:
                logger.error(f"❌ FAILED: {detail}")
            return PromptRunResult(
                prompt_index,
                variation_index,
                prompt,
                requires_clarification,
                response_text,
                dataset_path,
                False,
                detail,
                tool_calls=tool_calls,
                execution_time=execution_time,
                session_state=session_state,
                error_messages=error_messages,
            )
        if verbose:
            logger.info(f"✓ Dataset path set immediately: {dataset_path_str}")
    else:
        # Success based on retrieval indicators
        if verbose:
            logger.info("✓ Response indicates successful data retrieval (no dataset_path set)")

    return PromptRunResult(
        prompt_index,
        variation_index,
        prompt,
        requires_clarification,
        response_text,
        dataset_path,
        True,
        "Immediate retrieval validated",
        tool_calls=tool_calls,
        execution_time=execution_time,
        session_state=session_state,
        error_messages=error_messages,
    )


def _render_summary_table(results: list[PromptRunResult]) -> str:
    """Render a compact summary table of prompt runs."""

    def shorten(text: str, width: int) -> str:
        text = " ".join(text.split())
        if len(text) <= width:
            return text
        return text[: width - 1] + "…"

    headers = ["#", "Type", "Prompt", "Response", "Dataset / Outcome"]
    rows = []

    for result in results:
        prefix = f"{result.prompt_index}.{result.variation_index}"
        prompt_type = "clarify" if result.requires_clarification else "immediate"
        dataset_display = result.dataset_path or "<unset>"
        if not result.success:
            dataset_display = f"FAILED: {result.detail}"

        rows.append(
            [
                prefix,
                prompt_type,
                shorten(result.prompt, 60),
                shorten(result.response_text or "", 80),
                shorten(dataset_display, 50),
            ]
        )

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    def fmt_row(row: list[str]) -> str:
        return " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))

    lines = [fmt_row(headers), "-+-".join("-" * w for w in col_widths)]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def _save_results_to_s3(
    results: list[PromptRunResult], test_name: str = "chembl_interactivity"
) -> dict[str, str]:
    """
    Save test results to S3 in multiple formats.

    Creates a timestamped folder under robustness_tests/{test_name}/ and saves:
    - results.json: Complete results in JSON format (all fields)
    - summary.csv: Overview CSV (truncated prompts/responses for quick scanning)
    - detailed.csv: Full CSV with complete prompts, responses, tool calls, execution details
    - summary.txt: Human-readable text summary
    - index.html: Navigation page linking to all reports and visualizations

    Args:
        results: List of PromptRunResult objects
        test_name: Name of the test suite (used for folder organization)

    Returns:
        dict: Mapping of file types to their S3 paths
    """
    try:
        from cs_copilot.storage import is_s3_enabled

        if not is_s3_enabled():
            logger.warning("S3 not enabled - skipping result upload to S3")
            return {}

        # Create timestamped folder
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = f"robustness_tests/{test_name}/{timestamp}"

        saved_paths = {}

        # 1. Save detailed JSON results
        json_path = f"{base_path}/results.json"
        results_dict = [asdict(r) for r in results]
        with S3.open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "test_name": test_name,
                    "total_tests": len(results),
                    "passed": sum(1 for r in results if r.success),
                    "failed": sum(1 for r in results if not r.success),
                    "results": results_dict,
                },
                f,
                indent=2,
            )
        saved_paths["json"] = S3.path(json_path)
        logger.info(f"Saved JSON results to: {saved_paths['json']}")

        # 2. Save CSV summary
        import pandas as pd

        csv_path = f"{base_path}/summary.csv"
        df_data = []
        for r in results:
            df_data.append(
                {
                    "prompt_index": r.prompt_index,
                    "variation_index": r.variation_index,
                    "requires_clarification": r.requires_clarification,
                    "success": r.success,
                    "dataset_path": r.dataset_path or "",
                    "detail": r.detail,
                    "prompt": r.prompt[:200],  # Truncate for CSV
                    "response": (r.response_text or "")[:200],  # Truncate for CSV
                }
            )
        df = pd.DataFrame(df_data)
        with S3.open(csv_path, "w") as f:
            df.to_csv(f, index=False)
        saved_paths["csv"] = S3.path(csv_path)
        logger.info(f"Saved CSV summary to: {saved_paths['csv']}")

        # 3. Save text summary
        txt_path = f"{base_path}/summary.txt"
        summary_table = _render_summary_table(results)
        summary_text = f"""ChEMBL Interactivity Robustness Test Results
{'=' * 80}
Test: {test_name}
Timestamp: {timestamp}
Total Tests: {len(results)}
Passed: {sum(1 for r in results if r.success)}
Failed: {sum(1 for r in results if not r.success)}

{'=' * 80}
SUMMARY TABLE
{'=' * 80}

{summary_table}

{'=' * 80}
DETAILED RESULTS
{'=' * 80}

"""
        for r in results:
            status = "✅ PASSED" if r.success else "❌ FAILED"
            summary_text += f"""
Prompt {r.prompt_index}.{r.variation_index} - {status}
Type: {'Clarification' if r.requires_clarification else 'Immediate'}
Prompt: {r.prompt}
Response: {r.response_text or '<empty>'}
Dataset: {r.dataset_path or '<unset>'}
Detail: {r.detail}
{'-' * 80}
"""

        with S3.open(txt_path, "w") as f:
            f.write(summary_text)
        saved_paths["txt"] = S3.path(txt_path)
        logger.info(f"Saved text summary to: {saved_paths['txt']}")

        # 4. Save detailed CSV with full data (no truncation)
        logger.info("Generating detailed CSV with full execution data...")
        detailed_csv_path = f"{base_path}/detailed.csv"
        detailed_df_data = []
        for r in results:
            # Extract dataset comparison metrics if available
            comparison_metrics = r.dataset_comparison_metrics or {}
            consistency_score = comparison_metrics.get("consistency_score", None)
            activity_id_match = comparison_metrics.get("activity_id_match", None)
            num_activities = comparison_metrics.get("num_activities", None)
            columns_match = comparison_metrics.get("columns_match", None)
            dtypes_match = comparison_metrics.get("dtypes_match", None)
            sample_values_match = comparison_metrics.get("sample_values_match", {})
            row_count = r.dataset_row_count

            detailed_df_data.append(
                {
                    "prompt_index": r.prompt_index,
                    "variation_index": r.variation_index,
                    "prompt_type": "Clarification" if r.requires_clarification else "Immediate",
                    "success": r.success,
                    "execution_time_seconds": r.execution_time or 0.0,
                    "dataset_path": r.dataset_path or "",
                    "dataset_row_count": row_count if row_count is not None else "",
                    "dataset_consistency_score": (
                        consistency_score if consistency_score is not None else ""
                    ),
                    "dataset_activity_id_match": (
                        activity_id_match if activity_id_match is not None else ""
                    ),
                    "dataset_num_activities": num_activities if num_activities is not None else "",
                    "dataset_columns_match": columns_match if columns_match is not None else "",
                    "dataset_dtypes_match": dtypes_match if dtypes_match is not None else "",
                    "dataset_sample_values_match": (
                        json.dumps(sample_values_match) if sample_values_match else ""
                    ),
                    "validation_detail": r.detail,
                    "full_prompt": r.prompt,
                    "full_response": r.response_text or "",
                    "tool_calls_json": json.dumps(r.tool_calls or [], indent=None),
                    "session_state_json": json.dumps(
                        r.session_state or {}, default=str, indent=None
                    ),
                    "errors_json": json.dumps(r.error_messages or [], indent=None),
                }
            )
        detailed_df = pd.DataFrame(detailed_df_data)
        with S3.open(detailed_csv_path, "w") as f:
            detailed_df.to_csv(f, index=False)
        saved_paths["detailed_csv"] = S3.path(detailed_csv_path)
        logger.info(f"Saved detailed CSV to: {saved_paths['detailed_csv']}")

        # 5. Create HTML index for easy navigation
        index_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{test_name.replace("_", " ").title()} - Robustness Test Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f5f5f5; }}
        .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        .summary {{ background-color: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .summary-item {{ display: flex; justify-content: space-between; margin: 10px 0; }}
        .summary-label {{ font-weight: bold; }}
        .downloads {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
        .download-card {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; border-left: 4px solid #2196F3; }}
        .download-card h3 {{ margin-top: 0; color: #2196F3; }}
        .download-card p {{ color: #666; margin: 10px 0; }}
        .download-card a {{ display: inline-block; padding: 10px 20px; background-color: #2196F3; color: white; text-decoration: none; border-radius: 5px; margin-top: 10px; }}
        .download-card a:hover {{ background-color: #1976D2; }}
        .status-passed {{ color: #4CAF50; font-weight: bold; }}
        .status-failed {{ color: #f44336; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{test_name.replace("_", " ").title()} - Robustness Test Results</h1>

        <div class="summary">
            <div class="summary-item">
                <span class="summary-label">Timestamp:</span>
                <span>{timestamp}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Total Tests:</span>
                <span>{len(results)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Passed:</span>
                <span class="status-passed">{sum(1 for r in results if r.success)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Failed:</span>
                <span class="status-failed">{sum(1 for r in results if not r.success)}</span>
            </div>
            <div class="summary-item">
                <span class="summary-label">Success Rate:</span>
                <span>{(sum(1 for r in results if r.success) / len(results) * 100):.1f}%</span>
            </div>
        </div>

        <h2>📊 Download Reports</h2>
        <div class="downloads">
            <div class="download-card">
                <h3>📄 Summary CSV</h3>
                <p>Quick overview with truncated text (for spreadsheet viewing)</p>
                <a href="summary.csv">Download summary.csv</a>
            </div>

            <div class="download-card">
                <h3>📋 Detailed CSV</h3>
                <p>Complete data with full prompts, responses, tool calls, and execution details</p>
                <a href="detailed.csv">Download detailed.csv</a>
            </div>

            <div class="download-card">
                <h3>🔍 JSON Results</h3>
                <p>Machine-readable format with all fields</p>
                <a href="results.json">Download results.json</a>
            </div>

            <div class="download-card">
                <h3>📝 Text Summary</h3>
                <p>Human-readable summary report</p>
                <a href="summary.txt">Download summary.txt</a>
            </div>
        </div>

        <h2>📊 Dataset Consistency</h2>
        <p>Dataset comparison metrics show how consistent the retrieved data is across different prompt variations:</p>
        <ul>
            <li><strong>Consistency Score:</strong> Overall similarity (0.0-1.0, higher is better)</li>
            <li><strong>Column Match:</strong> Whether all datasets have the same columns</li>
            <li><strong>Row Jaccard:</strong> Overlap of data rows across variations</li>
        </ul>
        <p>See the detailed CSV for full comparison metrics per prompt variation.</p>

        <h2>📈 Visualizations</h2>
        <p>If visualizations were enabled, they will be saved in the same folder with names like:</p>
        <ul>
            <li><code>visualizations/overall_results.png</code></li>
            <li><code>visualizations/by_prompt_type.png</code></li>
            <li><code>visualizations/per_prompt.png</code></li>
            <li><code>visualizations/by_variation.png</code></li>
            <li><code>visualizations/dataset_consistency.png</code> (if dataset comparison enabled)</li>
        </ul>
    </div>
</body>
</html>
"""

        index_path = f"{base_path}/index.html"
        with S3.open(index_path, "w") as f:
            f.write(index_html)
        saved_paths["index"] = S3.path(index_path)
        logger.info(f"Saved navigation index to: {saved_paths['index']}")

        return saved_paths

    except Exception as e:
        logger.error(f"Failed to save results to S3: {e}")
        import traceback

        traceback.print_exc()
        return {}


def _generate_robustness_analysis(
    test_name: str,
    timestamp: str,
    model=None,
    save_to_s3: bool = True,
    verbose: bool = True,
) -> Optional[dict]:
    """
    Invoke robustness evaluator agent to analyze test results.

    Args:
        test_name: Name of the test (e.g., 'chembl_interactivity')
        timestamp: Timestamp of the test run
        model: LLM model to use (if None, uses model from config)
        save_to_s3: Whether to save report to S3
        verbose: Whether to log details

    Returns:
        Dict with analysis results or None if failed
    """
    try:
        from cs_copilot.agents.factories import RobustnessEvaluationFactory

        if verbose:
            logger.info("=" * 80)
            logger.info("Generating robustness analysis report...")
            logger.info("=" * 80)

        # Create robustness evaluator agent
        if model is None:
            model = _load_model_from_config()

        factory = RobustnessEvaluationFactory()
        agent = factory.create_agent(model=model)

        # Create analysis prompt
        analysis_prompt = f"""Analyze the robustness test results for test '{test_name}' at timestamp '{timestamp}'.

Please:
1. Load the test results from S3 or local storage
2. Calculate overall performance metrics and score distribution
3. Identify any failing prompts or low-scoring variations
4. Compare performance between clarification and immediate prompts
5. Analyze dataset consistency metrics (if applicable)
6. Generate actionable recommendations for improving robustness
7. Export a comprehensive analysis report in markdown format

Be thorough and specific in your analysis."""

        # Run the agent
        if verbose:
            logger.info(f"Running robustness analysis for {test_name}/{timestamp}...")

        response = agent.run(analysis_prompt, stream=False)
        response_text = response.content if response.content else ""

        if verbose:
            logger.info("✅ Robustness analysis completed")
            logger.info(f"\n{response_text}\n")

        # Extract report path from session state if available
        session_state = {}
        if hasattr(response, "session_state") and response.session_state:
            session_state = response.session_state
        elif hasattr(agent, "session_state") and agent.session_state:
            session_state = agent.session_state

        analysis_outputs = session_state.get("analysis_outputs", {})
        report_path = analysis_outputs.get("summary_report")

        return {
            "response": response_text,
            "report_path": report_path,
            "session_state": session_state,
            "timestamp": datetime.now(datetime.timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to generate robustness analysis: {e}")
        import traceback

        traceback.print_exc()
        return None


def _create_visualizations(
    results: list[PromptRunResult],
    test_name: str = "chembl_interactivity",
    output_dir: Optional[str] = None,
) -> dict[str, str]:
    """
    Create visual plots of test results.

    Creates multiple bar plots showing:
    1. Overall pass/fail rates
    2. Pass/fail by prompt type (clarification vs immediate)
    3. Pass/fail per prompt number
    4. Success rate by variation number

    Args:
        results: List of PromptRunResult objects
        test_name: Name of the test suite
        output_dir: Optional directory for saving plots (defaults to robustness_tests/{test_name}/visualizations)

    Returns:
        dict: Mapping of plot types to their file paths
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - skipping visualizations")
        return {}

    if not results:
        logger.warning("No results to visualize")
        return {}

    try:

        # Setup output directory
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = f"robustness_tests/{test_name}/{timestamp}/visualizations"

        saved_paths = {}

        # Prepare data
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests

        clarification_results = [r for r in results if r.requires_clarification]
        immediate_results = [r for r in results if not r.requires_clarification]

        # Group by prompt index
        prompt_groups = {}
        for r in results:
            if r.prompt_index not in prompt_groups:
                prompt_groups[r.prompt_index] = []
            prompt_groups[r.prompt_index].append(r)

        # Group by variation index
        variation_groups = {}
        for r in results:
            if r.variation_index not in variation_groups:
                variation_groups[r.variation_index] = []
            variation_groups[r.variation_index].append(r)

        # Set style
        plt.style.use(
            "seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"
        )
        colors = {"pass": "#2ecc71", "fail": "#e74c3c"}

        # ============================================================
        # Plot 1: Overall Pass/Fail
        # ============================================================
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ["Total Tests"]
        pass_counts = [passed_tests]
        fail_counts = [failed_tests]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pass_counts, width, label="Passed", color=colors["pass"])
        bars2 = ax.bar(x + width / 2, fail_counts, width, label="Failed", color=colors["fail"])

        ax.set_ylabel("Number of Tests", fontsize=12)
        ax.set_title(
            f'{test_name.replace("_", " ").title()}\nOverall Test Results',
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        for bar in bars2:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Add success rate text
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        ax.text(
            0.5,
            0.95,
            f"Success Rate: {success_rate:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
            fontsize=12,
            fontweight="bold",
        )

        plt.tight_layout()
        plot_path = f"{output_dir}/overall_results.png"
        with S3.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["overall"] = S3.path(plot_path)
        logger.info(f"Saved overall results plot to: {saved_paths['overall']}")

        # ============================================================
        # Plot 2: Pass/Fail by Prompt Type
        # ============================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ["Clarification\nPrompts", "Immediate\nRetrieval Prompts"]

        clarification_passed = sum(1 for r in clarification_results if r.success)
        clarification_failed = len(clarification_results) - clarification_passed
        immediate_passed = sum(1 for r in immediate_results if r.success)
        immediate_failed = len(immediate_results) - immediate_passed

        pass_counts = [clarification_passed, immediate_passed]
        fail_counts = [clarification_failed, immediate_failed]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width / 2, pass_counts, width, label="Passed", color=colors["pass"])
        bars2 = ax.bar(x + width / 2, fail_counts, width, label="Failed", color=colors["fail"])

        ax.set_ylabel("Number of Tests", fontsize=12)
        ax.set_title(
            f'{test_name.replace("_", " ").title()}\nResults by Prompt Type',
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        plot_path = f"{output_dir}/by_prompt_type.png"
        with S3.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["by_type"] = S3.path(plot_path)
        logger.info(f"Saved prompt type plot to: {saved_paths['by_type']}")

        # ============================================================
        # Plot 3: Success Rate per Prompt
        # ============================================================
        fig, ax = plt.subplots(figsize=(12, 6))
        prompt_numbers = sorted(prompt_groups.keys())
        success_rates = []
        colors_list = []

        for pnum in prompt_numbers:
            prompt_results = prompt_groups[pnum]
            passed = sum(1 for r in prompt_results if r.success)
            total = len(prompt_results)
            rate = (passed / total * 100) if total > 0 else 0
            success_rates.append(rate)
            # Color by prompt type
            is_clarification = prompt_results[0].requires_clarification
            colors_list.append("#3498db" if is_clarification else "#e67e22")

        x = np.arange(len(prompt_numbers))
        bars = ax.bar(x, success_rates, color=colors_list, alpha=0.7, edgecolor="black")

        ax.set_xlabel("Prompt Number", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            f'{test_name.replace("_", " ").title()}\nSuccess Rate per Prompt',
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i}" for i in prompt_numbers])
        ax.set_ylim(0, 105)
        ax.axhline(
            y=100, color="green", linestyle="--", alpha=0.5, linewidth=1, label="100% Success"
        )
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for _i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Add legend for colors
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#3498db", alpha=0.7, edgecolor="black", label="Clarification"),
            Patch(facecolor="#e67e22", alpha=0.7, edgecolor="black", label="Immediate Retrieval"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        plot_path = f"{output_dir}/per_prompt.png"
        with S3.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["per_prompt"] = S3.path(plot_path)
        logger.info(f"Saved per-prompt plot to: {saved_paths['per_prompt']}")

        # ============================================================
        # Plot 4: Success Rate by Variation Number
        # ============================================================
        fig, ax = plt.subplots(figsize=(10, 6))
        variation_numbers = sorted(variation_groups.keys())
        success_rates_var = []

        for vnum in variation_numbers:
            var_results = variation_groups[vnum]
            passed = sum(1 for r in var_results if r.success)
            total = len(var_results)
            rate = (passed / total * 100) if total > 0 else 0
            success_rates_var.append(rate)

        x = np.arange(len(variation_numbers))
        bars = ax.bar(x, success_rates_var, color="#9b59b6", alpha=0.7, edgecolor="black")

        ax.set_xlabel("Variation Number", fontsize=12)
        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_title(
            f'{test_name.replace("_", " ").title()}\nSuccess Rate by Variation',
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"V{i}" for i in variation_numbers])
        ax.set_ylim(0, 105)
        ax.axhline(
            y=100, color="green", linestyle="--", alpha=0.5, linewidth=1, label="100% Success"
        )
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()
        plot_path = f"{output_dir}/by_variation.png"
        with S3.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=150, bbox_inches="tight")
        plt.close()
        saved_paths["by_variation"] = S3.path(plot_path)
        logger.info(f"Saved variation plot to: {saved_paths['by_variation']}")

        # ============================================================
        # Plot 5: Dataset Activity ID Match (Binary)
        # ============================================================
        # Extract dataset consistency scores per prompt
        consistency_data = {}
        for r in results:
            if r.dataset_comparison_metrics and not r.requires_clarification:
                consistency_score = r.dataset_comparison_metrics.get("consistency_score")
                if consistency_score is not None:
                    if r.prompt_index not in consistency_data:
                        consistency_data[r.prompt_index] = consistency_score

        if consistency_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            prompt_numbers = sorted(consistency_data.keys())
            consistency_scores = [consistency_data[pnum] for pnum in prompt_numbers]

            # Binary colors: green for 1.0 (match), red for 0.0 (no match)
            colors = ["#2ecc71" if score == 1.0 else "#e74c3c" for score in consistency_scores]

            x = np.arange(len(prompt_numbers))
            bars = ax.bar(x, consistency_scores, color=colors, alpha=0.7, edgecolor="black")

            ax.set_xlabel("Prompt Number", fontsize=12)
            ax.set_ylabel("Activity ID Match", fontsize=12)
            ax.set_title(
                f'{test_name.replace("_", " ").title()}\nDataset Activity ID Match Across Variations',
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels([f"P{i}" for i in prompt_numbers])
            ax.set_ylim(0, 1.05)
            ax.axhline(
                y=1.0,
                color="green",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Required: Exact Match (1.0)",
            )
            ax.grid(axis="y", alpha=0.3)
            ax.legend()

            # Add value labels on bars
            for _i, bar in enumerate(bars):
                height = bar.get_height()
                label = "✓ Match" if height == 1.0 else "✗ Mismatch"
                color = "#2ecc71" if height == 1.0 else "#e74c3c"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color=color,
                    fontweight="bold",
                )

            plt.tight_layout()
            plot_path = f"{output_dir}/dataset_consistency.png"
            with S3.open(plot_path, "wb") as f:
                plt.savefig(f, dpi=150, bbox_inches="tight")
            plt.close()
            saved_paths["dataset_consistency"] = S3.path(plot_path)
            logger.info(f"Saved dataset consistency plot to: {saved_paths['dataset_consistency']}")

        return saved_paths

    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
        import traceback

        traceback.print_exc()
        return {}


def run_chembl_prompt_matrix(
    agent_team_factory,
    prompt_generator,
    n_variations=5,
    verbose=True,
    show_summary_table: bool = False,
    save_to_s3: bool = True,
    s3_session_isolation: bool = True,
    create_visualizations: bool = False,
    compare_datasets: bool = True,
    dataset_consistency_threshold: float = 1.0,
    enable_mlflow: bool = False,
    mlflow_experiment_name: str = "chembl_interactivity_tests",
):
    """
    Validate both clarification-seeking and fully specified ChEMBL prompts.

    This function runs ALL prompts even if some fail, allowing you to see complete
    test results. The overall success status is determined at the end based on
    whether all prompts passed or any failed.

    Each prompt variation runs in a completely separate session with isolated S3
    storage to prevent cross-contamination of results.

    Args:
        agent_team_factory: Factory function to create agent teams
        prompt_generator: PromptVariationGenerator instance
        n_variations: Number of variations per prompt
        verbose: Whether to log detailed output
        show_summary_table: Whether to display summary table in logs
        save_to_s3: Whether to save results to S3 (default: True)
        s3_session_isolation: Whether to isolate S3 storage per variation (default: True)
        create_visualizations: Whether to create visual plots of results (default: False)
        compare_datasets: Whether to compare datasets across variations (default: True)
        dataset_consistency_threshold: Minimum consistency score for datasets (default: 1.0, exact match)
        enable_mlflow: Whether to enable MLflow tracking (default: False)
        mlflow_experiment_name: MLflow experiment name (default: "chembl_interactivity_tests")

    Returns:
        tuple: (success: bool, message: str, results: list[PromptRunResult], s3_paths: dict[str, str], viz_paths: dict[str, str], analysis_result: dict)
            - success: True if all prompts passed, False if any failed
            - message: Summary message (includes failure details if any)
            - results: Complete list of all prompt results
            - s3_paths: Dict mapping file types to S3 paths (empty if save_to_s3=False)
            - viz_paths: Dict mapping visualization types to paths (empty if create_visualizations=False)
            - analysis_result: Robustness analysis from evaluator agent (None if S3 disabled)
    """
    import datetime
    import uuid

    # Generate base test run ID for this execution
    test_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize MLflow tracking if enabled
    if enable_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(mlflow_experiment_name)
            if verbose:
                logger.info(f"MLflow tracking enabled for experiment: {mlflow_experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
            enable_mlflow = False
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Tracking disabled.")
            enable_mlflow = False

    # Check if S3 is enabled for session isolation
    from cs_copilot.storage import is_s3_enabled

    if s3_session_isolation and not is_s3_enabled():
        logger.warning(
            "S3 session isolation requested but S3 is not enabled. "
            "Sessions will not be isolated. Set USE_S3=true to enable isolation."
        )
        s3_session_isolation = False

    prompt_cases = prompt_generator.get_prompt_cases(
        "chembl_interactivity", n_variations=n_variations
    )
    results: list[PromptRunResult] = []
    failed_results: list[tuple[int, int, str]] = []  # Track (prompt_idx, variation_idx, detail)

    # Wrap entire test suite in MLflow run if enabled
    def _run_test_suite():
        """Inner function to wrap with MLflow tracking."""
        nonlocal enable_mlflow  # Allow modification of outer variable
        for idx, prompt_case in enumerate(prompt_cases, 1):
            prompts = [prompt_case["base"]] + list(prompt_case["variations"])

            # Start MLflow run for each prompt case if enabled
            prompt_case_run_id = None
            if enable_mlflow:
                try:
                    import mlflow

                    prompt_case_name = f"prompt_{idx}_{'clarification' if prompt_case['requires_clarification'] else 'immediate'}"
                    prompt_case_run = mlflow.start_run(run_name=prompt_case_name, nested=True)
                    prompt_case_run_id = prompt_case_run.info.run_id
                    mlflow.log_params(
                        {
                            "prompt_index": idx,
                            "requires_clarification": prompt_case["requires_clarification"],
                            "n_variations": len(prompts),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to start MLflow prompt case run: {e}")
                    enable_mlflow = False  # Disable for this case

            try:
                for variation_idx, prompt in enumerate(prompts, 1):
                    # Start MLflow run for each variation if enabled
                    variation_run_id = None
                    if enable_mlflow and prompt_case_run_id:
                        try:
                            import mlflow

                            variation_run_name = f"variation_{variation_idx}"
                            variation_run = mlflow.start_run(
                                run_name=variation_run_name, nested=True
                            )
                            variation_run_id = variation_run.info.run_id
                            mlflow.log_params(
                                {
                                    "variation_index": variation_idx,
                                    "prompt_preview": prompt[:200] if prompt else "",
                                }
                            )
                            mlflow.log_text(prompt, f"prompt_p{idx}_v{variation_idx}.txt")
                        except Exception as e:
                            logger.warning(f"Failed to start MLflow variation run: {e}")
                            variation_run_id = None

                    try:
                        if verbose:
                            logger.info("=" * 80)
                            logger.info(
                                f"TEST: Prompt {idx} variation {variation_idx} "
                                f"({'clarification' if prompt_case['requires_clarification'] else 'immediate'})"
                            )
                            logger.info("=" * 80)

                        # Generate unique S3 prefix for this variation
                        s3_prefix = None
                        if s3_session_isolation:
                            session_id = f"robustness_{test_run_id}_p{idx}_v{variation_idx}_{uuid.uuid4().hex[:8]}"
                            s3_prefix = f"sessions/{session_id}"
                            if verbose:
                                logger.debug(f"Session ID: {session_id}")

                        result = _run_prompt(
                            agent_team_factory,
                            prompt,
                            requires_clarification=prompt_case["requires_clarification"],
                            prompt_index=idx,
                            variation_index=variation_idx,
                            s3_session_prefix=s3_prefix,
                            verbose=verbose,
                        )
                        results.append(result)

                        # Log variation metrics to MLflow if enabled
                        if enable_mlflow and variation_run_id:
                            try:
                                import mlflow

                                # Determine if clarification was provided (no dataset = asked for clarification)
                                clarification_provided = (
                                    result.requires_clarification and not result.dataset_path
                                )
                                mlflow.log_metrics(
                                    {
                                        "success": 1.0 if result.success else 0.0,
                                        "requires_clarification": (
                                            1.0 if result.requires_clarification else 0.0
                                        ),
                                        "clarification_provided": (
                                            1.0 if clarification_provided else 0.0
                                        ),
                                    }
                                )
                                if result.dataset_path:
                                    mlflow.log_param("dataset_path", result.dataset_path)
                            except Exception as e:
                                logger.warning(f"Failed to log variation metrics: {e}")

                        if not result.success:
                            failed_results.append((idx, variation_idx, result.detail))
                            if verbose:
                                logger.warning(
                                    f"⚠️  Prompt {idx} variation {variation_idx} failed: {result.detail}"
                                )
                                logger.info("Continuing with remaining prompts...\n")

                    finally:
                        # Close variation run
                        if variation_run_id:
                            try:
                                import mlflow

                                mlflow.end_run()
                            except Exception as e:
                                logger.debug(f"Failed to end variation run: {e}")

            finally:
                # Log prompt case metrics and close run
                if prompt_case_run_id:
                    try:
                        import mlflow

                        case_results = [r for r in results if r.prompt_index == idx]
                        case_success_count = sum(1 for r in case_results if r.success)
                        mlflow.log_metrics(
                            {
                                "total_variations": len(case_results),
                                "passed_variations": float(case_success_count),
                                "pass_rate": (
                                    float(case_success_count / len(case_results))
                                    if case_results
                                    else 0.0
                                ),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log prompt case metrics: {e}")
                    finally:
                        try:
                            mlflow.end_run()
                        except Exception as e:
                            logger.debug(f"Failed to end prompt case run: {e}")

    # Execute test suite with MLflow tracking if enabled
    if enable_mlflow:
        try:
            import mlflow

            suite_run_name = f"chembl_interactivity_suite_{test_run_id}"
            with mlflow.start_run(run_name=suite_run_name):
                # Log suite configuration
                try:
                    mlflow.log_params(
                        {
                            "test_run_id": test_run_id,
                            "n_variations": n_variations,
                            "n_prompt_cases": len(prompt_cases),
                            "s3_session_isolation": s3_session_isolation,
                            "compare_datasets": compare_datasets,
                            "dataset_consistency_threshold": dataset_consistency_threshold,
                        }
                    )
                    mlflow.set_tags(
                        {
                            "test_type": "chembl_interactivity",
                            "test_run_id": test_run_id,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log suite parameters: {e}")

                # Run the test suite
                _run_test_suite()

                # Log suite-level metrics after test execution
                try:
                    total_tests = len(results)
                    passed_tests = sum(1 for r in results if r.success)
                    mlflow.log_metrics(
                        {
                            "total_tests": float(total_tests),
                            "passed_tests": float(passed_tests),
                            "failed_tests": float(total_tests - passed_tests),
                            "pass_rate": (
                                float(passed_tests / total_tests) if total_tests > 0 else 0.0
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to log suite metrics: {e}")
        except Exception as e:
            logger.error(f"MLflow suite tracking failed: {e}")
            # Continue without MLflow
            _run_test_suite()
    else:
        # Run without MLflow tracking
        _run_test_suite()

    # Compare datasets across variations if requested
    dataset_comparison_failures = []
    if compare_datasets:
        if verbose:
            logger.info("=" * 80)
            logger.info("Comparing datasets across prompt variations...")
            logger.info("=" * 80)

        # Get unique prompt indices
        prompt_indices = sorted({r.prompt_index for r in results})

        for prompt_idx in prompt_indices:
            try:
                comparison_metrics = _compare_datasets_for_prompt_group(
                    results, prompt_idx, enable_comparison=compare_datasets
                )

                if comparison_metrics:
                    # Check if consistency meets threshold
                    consistency_score = next(iter(comparison_metrics.values()), {}).get(
                        "consistency_score", 1.0
                    )

                    if consistency_score < dataset_consistency_threshold:
                        msg = (
                            f"Prompt {prompt_idx} dataset consistency below threshold: "
                            f"{consistency_score:.3f} < {dataset_consistency_threshold}"
                        )
                        dataset_comparison_failures.append((prompt_idx, msg))
                        if verbose:
                            logger.warning(f"⚠️  {msg}")
                    else:
                        if verbose:
                            logger.info(
                                f"✓ Prompt {prompt_idx} dataset consistency: {consistency_score:.3f}"
                            )

            except Exception as e:
                logger.warning(f"Failed to compare datasets for prompt {prompt_idx}: {e}")
                if verbose:
                    import traceback

                    traceback.print_exc()

    # Calculate overall status
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.success)
    failed_tests = len(failed_results)
    dataset_failures = len(dataset_comparison_failures)
    all_passed = failed_tests == 0 and dataset_failures == 0

    # Log summary
    if verbose:
        logger.info("=" * 80)
        if all_passed:
            logger.info("\n✅ TEST PASSED: All ChEMBL interactivity prompts behaved as expected")
            if compare_datasets:
                logger.info("   ✅ All dataset comparisons passed consistency threshold\n")
        else:
            logger.warning(
                f"\n⚠️  TEST COMPLETED WITH FAILURES: {passed_tests}/{total_tests} prompts passed\n"
            )
            if failed_results:
                logger.warning("Failed prompts:")
                for prompt_idx, variation_idx, detail in failed_results:
                    logger.warning(f"  - Prompt {prompt_idx}.{variation_idx}: {detail}")
            if dataset_comparison_failures:
                logger.warning("Dataset consistency failures:")
                for prompt_idx, msg in dataset_comparison_failures:
                    logger.warning(f"  - Prompt {prompt_idx}: {msg}")
            logger.info("")

    if show_summary_table:
        logger.info("\nPrompt summary table:\n" + _render_summary_table(results))

    # Save results to S3 (always save, even if some failed)
    s3_paths = {}
    if save_to_s3:
        s3_paths = _save_results_to_s3(results, test_name="chembl_interactivity")

    # Create visualizations if requested
    viz_paths = {}
    if create_visualizations:
        if verbose:
            logger.info("Creating visualizations...")
        viz_paths = _create_visualizations(results, test_name="chembl_interactivity")
        if verbose and viz_paths:
            logger.info(f"Created {len(viz_paths)} visualization(s)")

    # Generate robustness analysis report
    analysis_result = None
    if s3_paths:
        # Extract timestamp from S3 paths (format: robustness_tests/{test_name}/{timestamp}/...)
        if s3_paths.get("json"):
            import re

            match = re.search(r"/(\d{8}_\d{6})/", s3_paths["json"])
            if match:
                timestamp = match.group(1)
                analysis_result = _generate_robustness_analysis(
                    test_name="chembl_interactivity",
                    timestamp=timestamp,
                    model=None,  # Will use model from config
                    save_to_s3=True,
                    verbose=verbose,
                )
                if analysis_result and verbose:
                    logger.info(
                        f"Analysis report: {analysis_result.get('report_path', 'generated in console')}"
                    )

    # Return appropriate status
    if all_passed:
        msg = "All prompt cases validated"
        if compare_datasets and not dataset_comparison_failures:
            msg += " and datasets are consistent"
        return True, msg, results, s3_paths, viz_paths, analysis_result
    else:
        failure_parts = []
        if failed_tests > 0:
            failure_parts.append(f"{failed_tests}/{total_tests} prompts failed")
        if dataset_failures > 0:
            failure_parts.append(f"{dataset_failures} dataset consistency checks failed")

        failure_summary = "; ".join(failure_parts)

        if failed_results:
            first_failure = failed_results[0]
            failure_summary += f" (first prompt failure: {first_failure[0]}.{first_failure[1]}: {first_failure[2]})"
        elif dataset_comparison_failures:
            first_failure = dataset_comparison_failures[0]
            failure_summary += f" (first dataset failure: {first_failure[1]})"

        return False, failure_summary, results, s3_paths, viz_paths, analysis_result


# ============================================================================
# Pytest fixtures and test wrappers (only active when pytest is available)
# ============================================================================

if PYTEST_AVAILABLE:

    @pytest.fixture(scope="session", autouse=True)
    def setup_s3_for_tests():
        """Setup S3 configuration before running tests."""
        s3_config = _setup_s3()
        if s3_config is None:
            logger.warning(
                "S3 not configured. Tests may fail if S3 is required. "
                "Set USE_S3=true and provide endpoint, bucket, and credentials."
            )
        yield s3_config

    @pytest.fixture
    def agent_team_factory() -> Callable:
        return _create_agent_team_factory()

    @pytest.fixture
    def agent_team(agent_team_factory):
        return agent_team_factory()

    @pytest.fixture
    def prompt_generator():
        """
        Generate prompts without loading embedding model for speed.

        Respects CS_COPILOT_PROMPT_TEMPLATES environment variable for custom templates.
        """
        # Allow environment variable to override (e.g., for short templates)
        templates_path = None
        if "CS_COPILOT_PROMPT_TEMPLATES" in os.environ:
            templates_path = Path(os.environ["CS_COPILOT_PROMPT_TEMPLATES"])

        return PromptVariationGenerator(templates_path=templates_path, validate_similarity=False)

    def test_chembl_prompt_matrix(agent_team_factory, prompt_generator):
        """
        Validate clarification-seeking vs immediate ChEMBL prompts.

        Each prompt variation runs in a completely separate session with isolated
        S3 storage to prevent cross-contamination of results.

        Note: All prompts are executed even if some fail, allowing you to see
        the complete test results. The test will fail if any prompt fails.

        This test also validates that datasets retrieved by different prompt
        variations are consistent (same data).
        """
        (
            success,
            message,
            results,
            s3_paths,
            viz_paths,
            analysis_result,
        ) = run_chembl_prompt_matrix(
            agent_team_factory,
            prompt_generator,
            n_variations=5,
            verbose=False,
            show_summary_table=False,
            save_to_s3=True,
            s3_session_isolation=True,
            create_visualizations=False,  # Disabled by default in pytest
            compare_datasets=True,  # Enable dataset comparison
            dataset_consistency_threshold=1.0,  # Require exact activity_id match
        )

        # Log S3 paths if results were saved
        if s3_paths:
            logger.info("Test results saved to S3:")
            for file_type, path in s3_paths.items():
                logger.info(f"  {file_type}: {path}")

        # Log visualization paths if created
        if viz_paths:
            logger.info("Visualizations created:")
            for viz_type, path in viz_paths.items():
                logger.info(f"  {viz_type}: {path}")

        # Log analysis result if available
        if analysis_result:
            logger.info("Robustness analysis generated:")
            if analysis_result.get("report_path"):
                logger.info(f"  Report: {analysis_result['report_path']}")

        # Log detailed failure information if test failed
        if not success:
            failed_results = [r for r in results if not r.success]
            if failed_results:
                logger.error(f"\nFailed prompts ({len(failed_results)}/{len(results)}):")
                for r in failed_results:
                    logger.error(f"  Prompt {r.prompt_index}.{r.variation_index}: {r.detail}")

        assert success, message


# ============================================================================
# Standalone script execution
# ============================================================================


def main():
    """Run tests as a standalone script with detailed output."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ChEMBL Interactivity Robustness Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run tests with all reports and visualizations (default)
  python test_chembl_interactivity.py

  # Use short prompt templates (faster testing)
  python test_chembl_interactivity.py --prompt-templates short

  # Run with fewer variations for faster testing
  python test_chembl_interactivity.py --n-variations 3 --prompt-templates short

  # Run without visualizations
  python test_chembl_interactivity.py --no-visualize

  # Run without S3 saving (local output only)
  python test_chembl_interactivity.py --no-s3

  # Run without dataset comparison (faster but less thorough)
  python test_chembl_interactivity.py --no-compare-datasets

  # Disable dataset consistency check (allow any differences)
  python test_chembl_interactivity.py --dataset-consistency-threshold 0.0

  # Use custom prompt templates file
  python test_chembl_interactivity.py --prompt-templates /path/to/templates.yaml

  # Enable MLflow tracking
  python test_chembl_interactivity.py --mlflow

  # MLflow with custom experiment name
  python test_chembl_interactivity.py --mlflow --mlflow-experiment custom_experiment
        """,
    )
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Enable MLflow tracking for test execution",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="chembl_interactivity_tests",
        help="MLflow experiment name (default: chembl_interactivity_tests)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable visual plots generation (enabled by default)",
    )
    parser.add_argument(
        "--n-variations",
        type=int,
        default=5,
        help="Number of prompt variations to test per prompt (default: 5)",
    )
    parser.add_argument("--no-s3", action="store_true", help="Disable S3 result saving")
    parser.add_argument(
        "--no-compare-datasets",
        action="store_true",
        help="Disable dataset comparison across variations (enabled by default)",
    )
    parser.add_argument(
        "--dataset-consistency-threshold",
        type=float,
        default=1.0,
        help="Minimum consistency score for datasets (default: 1.0 for exact activity_id match, set to 0.0 to disable)",
    )
    parser.add_argument(
        "--prompt-templates",
        type=str,
        default=None,
        help="Path to prompt templates YAML file (default: fixtures/prompt_templates.yaml, use 'short' for prompt_templates_short.yaml)",
    )

    args = parser.parse_args()

    # Visualizations enabled by default (unless --no-visualize is passed)
    create_visualizations = not args.no_visualize
    compare_datasets = not args.no_compare_datasets

    logger.info("ChEMBL Interactivity Robustness Tests")
    logger.info("=" * 80)
    if args.mlflow:
        logger.info(f"MLflow tracking enabled (experiment: {args.mlflow_experiment})")
    if compare_datasets:
        logger.info(
            f"Dataset comparison enabled (consistency threshold: {args.dataset_consistency_threshold:.2f})"
        )
    else:
        logger.info("Dataset comparison disabled")

    if create_visualizations and not MATPLOTLIB_AVAILABLE:
        logger.warning("Matplotlib not available - visualizations will be skipped")
        logger.warning("Install matplotlib with: pip install matplotlib")
        create_visualizations = False

    # Setup
    logger.info("Setting up test environment...")
    _setup_s3()

    # Determine prompt templates path
    templates_path = None
    if args.prompt_templates:
        if args.prompt_templates == "short":
            # Shortcut for short templates
            templates_path = Path(__file__).parent / "fixtures" / "prompt_templates_short.yaml"
            logger.info("Using short prompt templates")
        else:
            templates_path = Path(args.prompt_templates)
            logger.info(f"Using custom prompt templates: {templates_path}")
    else:
        logger.info("Using default prompt templates")

    try:
        agent_team_factory = _create_agent_team_factory()
        prompt_generator = PromptVariationGenerator(
            templates_path=templates_path, validate_similarity=False
        )
        logger.info("✓ Test environment ready\n")
    except Exception as e:
        logger.error(f"Failed to setup test environment: {e}")
        return 1

    # Run tests
    all_passed = True
    s3_paths = {}
    viz_paths = {}
    analysis_result = None

    try:
        (
            success,
            message,
            results,
            s3_paths,
            viz_paths,
            analysis_result,
        ) = run_chembl_prompt_matrix(
            agent_team_factory,
            prompt_generator,
            n_variations=args.n_variations,
            verbose=True,
            show_summary_table=True,
            save_to_s3=not args.no_s3,
            s3_session_isolation=True,
            create_visualizations=create_visualizations,
            compare_datasets=not args.no_compare_datasets,
            dataset_consistency_threshold=args.dataset_consistency_threshold,
            enable_mlflow=args.mlflow,
            mlflow_experiment_name=args.mlflow_experiment,
        )
        if not success:
            all_passed = False
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Summary
    logger.info("=" * 80)
    if s3_paths:
        logger.info("Test results saved to S3:")
        for file_type, path in s3_paths.items():
            logger.info(f"  {file_type}: {path}")
        logger.info("=" * 80)

    if viz_paths:
        logger.info("Visualizations created:")
        for viz_type, path in viz_paths.items():
            logger.info(f"  {viz_type}: {path}")
        logger.info("=" * 80)

    if analysis_result:
        logger.info("Robustness Analysis Report:")
        if analysis_result.get("report_path"):
            logger.info(f"  Report: {analysis_result['report_path']}")
        logger.info("=" * 80)

    if all_passed:
        logger.info("✅ ALL TESTS PASSED")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
