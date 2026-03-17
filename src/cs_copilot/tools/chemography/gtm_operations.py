#!/usr/bin/env python
# coding: utf-8
"""
Canonical GTM Operations Module

This module provides the single source of truth for all GTM operations.
All GTM toolkit classes should delegate to these functions.

Functions are organized into categories:
- Core Operations: GTM optimization, ruggedness, save/load
- Data Operations: Loading and preparing GTM data
- Analysis Operations: Scaffold analysis, source checking, coordinate mapping
- Utility Functions: Helper functions for data manipulation
"""

import base64
import gzip
import math
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import altair as alt
import dill
import numpy as np

# GTM optimization imports
import optuna
import pandas as pd
import torch
from agno.agent import Agent
from agno.tools.pandas import PandasTools
from chemographykit.gtm import GTM
from chemographykit.plots.altair_landscapes import (
    altair_discrete_class_landscape,
    altair_discrete_density_landscape,
    altair_discrete_regression_landscape,
    altair_points_chart,
)
from chemographykit.utils.classification import class_density_to_table, get_class_density_matrix
from chemographykit.utils.density import density_to_table, get_density_matrix
from chemographykit.utils.molecules import calculate_latent_coords
from chemographykit.utils.regression import get_reg_density_matrix, reg_density_to_table
from optuna.samplers import TPESampler
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.ndimage import convolve
from sklearn.neighbors import NearestNeighbors

from cs_copilot.storage import S3
from cs_copilot.utils.logging import setup_logging

from ..chemistry.base_chemistry import _smiles_to_mol_or_none
from ..chemistry.descriptors import (
    DEFAULT_DESCRIPTOR_TYPE,
    MolecularDescriptorEncoder,
)
from ..chemistry.standardize import standardize_smiles, standardize_smiles_column
from ..constants import (
    CSV_EXTENSION,
    DEFAULT_CHART_HEIGHT,
    DEFAULT_CHART_WIDTH,
    DEFAULT_DBAASP_DATA_PATH,
    DEFAULT_GRADIENT_MAX_LENGTH,
    DEFAULT_GRADIENT_THICKNESS,
    DEFAULT_GTM_MODEL_PATH,
    DEFAULT_LEGEND_FONT_SIZE,
    DEFAULT_NODE_THRESHOLD,
    DEFAULT_POINTS_OPACITY,
    DEFAULT_POINTS_SIZE,
    DEFAULT_TICK_COUNT,
    GTM_MODEL_SUFFIXES,
    HTML_EXTENSION,
    HUGGINGFACE_GTM_REPO,
    MIN_ORGANISM_DATA_POINTS,
    PKL_GZ_EXTENSION,
    PNG_EXTENSION,
    SEQUENCE_COLUMN,
    SMILES_COLUMN,
)
from ..io.formatting import df_as_str, has_integer_sqrt, smiles_to_png_bytes, value_counts_df
from ..io.utils import validate_positive_int

# Set up logging with warning suppression
logger = setup_logging(suppress_warnings=True, suppress_tqdm=True)

# Session state key for storing the current GTM model
SESSION_GTM_MODEL_KEY = "_current_gtm_model"
SESSION_GTM_MODEL_PATH_KEY = "_current_gtm_model_path"

# Standard SMILES column name variations to check (in priority order)
_SMILES_COLUMN_VARIANTS = [SMILES_COLUMN, "SMILES", "smiles", "Smiles"]


def find_smiles_column(df: pd.DataFrame) -> str:
    """
    Find and return the SMILES column name in a DataFrame.

    Checks for common SMILES column name variations and returns the first match.

    Args:
        df: DataFrame to search for SMILES column

    Returns:
        Name of the SMILES column found

    Raises:
        ValueError: If no SMILES column is found
    """
    for col_name in _SMILES_COLUMN_VARIANTS:
        if col_name in df.columns:
            return col_name

    raise ValueError(
        f"No SMILES column found in DataFrame. "
        f"Available columns: {list(df.columns)}. "
        f"Expected one of: {_SMILES_COLUMN_VARIANTS}"
    )


def normalize_smiles_column(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Normalize SMILES column name to the standard name.

    Finds the SMILES column and renames it to the standard name if needed.

    Args:
        df: DataFrame to normalize
        inplace: If True, modify the DataFrame in place; otherwise return a copy

    Returns:
        DataFrame with normalized SMILES column name

    Raises:
        ValueError: If no SMILES column is found
    """
    if not inplace:
        df = df.copy()

    smiles_col = find_smiles_column(df)

    # Rename to standard name if needed
    if smiles_col != SMILES_COLUMN:
        df.rename(columns={smiles_col: SMILES_COLUMN}, inplace=True)

    return df


def calculate_nn_preservation(
    X_high_dim: np.ndarray,
    X_low_dim: np.ndarray,
    k_neighbors: Union[int, List[int]],
    high_dim_indexes: np.ndarray = None,
    high_dim_metric: str = "euclidean",
) -> Union[float, List[float]]:
    """
    Calculate the nearest neighbor preservation scores for different k values.

    Args:
        X_high_dim (np.ndarray): High-dimensional data of shape (n_samples, n_features_high).
        X_low_dim (np.ndarray): Low-dimensional data of shape (n_samples, n_features_low).
        k_neighbors (int or List[int]): Single k value or list of k values.
        high_dim_indexes (np.ndarray, optional): Precomputed high-dimensional neighbor indices.
        high_dim_metric (str, optional): Metric to use in low-dimensional space

    Returns:
        float or List[float]: Preservation score(s) as a percentage.
    """
    # Ensure k_neighbors is a list
    if isinstance(k_neighbors, int):
        k_list = [k_neighbors]
        single_k = True
    else:
        k_list = k_neighbors
        single_k = False

    nn_preservation_scores = []

    # Precompute high-dimensional nearest neighbors if not provided
    if high_dim_indexes is None:
        max_k = max(k_list)
        nbrs_high = NearestNeighbors(n_neighbors=max_k + 1, metric=high_dim_metric).fit(X_high_dim)
        _, indices_high = nbrs_high.kneighbors(X_high_dim)
        indices_high = indices_high[:, 1:]  # Exclude self
    else:
        indices_high = high_dim_indexes
        max_k = indices_high.shape[1]

    # Precompute nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=max_k + 1).fit(X_low_dim)
    _, indices_low = nbrs_low.kneighbors(X_low_dim)
    indices_low = indices_low[:, 1:]  # Exclude self

    for k in k_list:
        indices_high_k = indices_high[:, :k]  # shape (n_samples, k)
        indices_low_k = indices_low[:, :k]  # shape (n_samples, k)

        # Vectorized computation
        combined_indices = np.concatenate((indices_high_k, indices_low_k), axis=1)
        sorted_indices = np.sort(combined_indices, axis=1)
        diffs = np.diff(sorted_indices, axis=1)
        overlaps_per_sample = np.sum(diffs == 0, axis=1)
        overlap_counts = overlaps_per_sample / k
        avg_preservation = np.mean(overlap_counts) * 100
        nn_preservation_scores.append(avg_preservation)

    if single_k:
        return nn_preservation_scores[0]
    else:
        return nn_preservation_scores


# =============================================================================
# GTM Model Resolution and Loading Functions
# =============================================================================


def get_session_gtm_model(agent: Optional[Agent]) -> Optional[Any]:
    """
    Get the GTM model from agent session state if available.

    Args:
        agent: Agent instance with session_state

    Returns:
        GTM model object if found in session state, None otherwise
    """
    if agent is None or agent.session_state is None:
        return None

    return agent.session_state.get(SESSION_GTM_MODEL_KEY)


def set_session_gtm_model(agent: Optional[Agent], gtm_model: Any, model_path: str) -> None:
    """
    Store the GTM model in agent session state.

    Args:
        agent: Agent instance with session_state
        gtm_model: GTM model object to store
        model_path: Path to the model file
    """
    if agent is None:
        return

    if agent.session_state is None:
        agent.session_state = {}

    agent.session_state[SESSION_GTM_MODEL_KEY] = gtm_model
    agent.session_state[SESSION_GTM_MODEL_PATH_KEY] = model_path


def resolve_gtm_model_path(
    gtm_file: Optional[str] = None,
    *,
    agent: Optional[Agent] = None,
    use_default: bool = False,
    generate_framesets: bool = False,
) -> str:
    """
    Resolve GTM model path with priority:
    1. Explicit file path if provided and use_default=False
    2. Session state model path if available and use_default=False
    3. Default model path (S3 assets, default directory, Hugging Face)

    Args:
        gtm_file: Optional explicit path to a GTM model file
        agent: Optional Agent instance to check session state
        use_default: If True, force use of default model even if session model exists
        generate_framesets: When True, generate cached frameset CSVs when downloading

    Returns:
        Resolved path to the GTM model file
    """
    # Priority 1: Explicit file path (unless use_default is True)
    if gtm_file and not use_default:
        logger.debug(f"Using explicit GTM model path: {gtm_file}")
        return gtm_file

    # Priority 2: Session state model (unless use_default is True)
    if agent is not None and agent.session_state is not None:
        session_model_path = agent.session_state.get(SESSION_GTM_MODEL_PATH_KEY)
        if session_model_path and not use_default:
            logger.debug(f"Using session state GTM model path: {session_model_path}")
            return session_model_path

    # Priority 3: Default model resolution
    logger.debug("Resolving default GTM model path")

    # Try S3 assets first (check for any .pkl.gz file)
    # This is a placeholder - actual S3 discovery would need to be implemented
    # based on your S3 storage structure

    # Try default directory
    default_path = Path(DEFAULT_GTM_MODEL_PATH).expanduser()
    if default_path.exists():
        # Look for model files in the default directory
        for suffix in GTM_MODEL_SUFFIXES:
            pattern = f"*{suffix}"
            matches = list(default_path.glob(pattern))
            if matches:
                model_path = str(matches[0])
                logger.info(f"Found default GTM model at: {model_path}")
                return model_path

    # Try Hugging Face download
    try:
        from huggingface_hub import snapshot_download

        default_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Downloading GTM model from HuggingFace: {HUGGINGFACE_GTM_REPO}")
        downloaded_path = snapshot_download(
            repo_id=HUGGINGFACE_GTM_REPO,
            local_dir=str(default_path),
            local_dir_use_symlinks=False,
        )

        # Find the model file in the downloaded directory
        for suffix in GTM_MODEL_SUFFIXES:
            pattern = f"*{suffix}"
            matches = list(Path(downloaded_path).glob(pattern))
            if matches:
                model_path = str(matches[0])
                logger.info(f"Downloaded GTM model to: {model_path}")
                return model_path

    except ImportError:
        logger.warning("huggingface_hub not available, cannot download from HuggingFace")
    except Exception as e:
        logger.warning(f"Failed to download from HuggingFace: {e}")

    # If nothing worked, return the default path (may not exist)
    logger.warning(f"Could not resolve GTM model, using default path: {default_path}")
    return str(default_path)


def load_gtm_model(gtm_model_path: str) -> Any:
    """
    Load a GTM model from a file path.

    Supports both gzipped (.pkl.gz) and non-gzipped (.pkl) pickle files.
    If a file with .pkl.gz extension is not actually gzipped, it will
    automatically fall back to loading it as a regular pickle file.

    Args:
        gtm_model_path: Path to the GTM model file

    Returns:
        Loaded GTM model object

    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: If loading fails
    """
    gtm_model_path = _ensure_suffix(gtm_model_path, ".pkl.gz")

    logger.info(f"Loading GTM model from: {gtm_model_path}")

    try:
        # First, try to load as a gzipped file (expected format)
        try:
            with S3.open(gtm_model_path, "rb") as f:
                with gzip.open(f, "rb") as gz:
                    gtm = dill.load(gz)
            logger.info("Successfully loaded GTM model (gzipped)")
            return gtm
        except gzip.BadGzipFile:
            # File has .gz extension but is not actually gzipped
            # Fall back to loading as a regular pickle file
            logger.warning(
                f"File {gtm_model_path} has .gz extension but is not gzipped. "
                "Loading as regular pickle file."
            )
            # Reopen the file for non-gzipped loading
            with S3.open(gtm_model_path, "rb") as f:
                gtm = dill.load(f)
            logger.info("Successfully loaded GTM model (non-gzipped)")
            return gtm
    except FileNotFoundError:
        logger.error(f"GTM model file not found: {gtm_model_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading GTM model: {e}")
        raise


# =============================================================================
# GTM Data Loading and Landscape Functions
# =============================================================================


def calculate_nn_preservation_per_sample(
    X_high_dim: np.ndarray,
    X_low_dim: np.ndarray,
    k_neighbors: int,
    high_dim_indexes: np.ndarray,
) -> np.ndarray:
    """
    Calculate the nearest neighbor preservation score for each sample.

    Args:
        X_high_dim (np.ndarray): High-dimensional data of shape (n_samples, n_features_high).
        X_low_dim (np.ndarray): Low-dimensional data of shape (n_samples, n_features_low).
        k_neighbors (int): Number of nearest neighbors to consider.
        high_dim_indexes (np.ndarray): Precomputed high-dimensional neighbor indices of shape (n_samples, k_max),
            where k_max >= k_neighbors. Should exclude self (first column removed).

    Returns:
        np.ndarray: Per-sample preservation scores as a percentage, shape (n_samples,).
    """
    # Ensure k_neighbors doesn't exceed available neighbors
    max_k_available = high_dim_indexes.shape[1]
    k = min(k_neighbors, max_k_available)

    # Precompute nearest neighbors in low-dimensional space
    nbrs_low = NearestNeighbors(n_neighbors=k + 1).fit(X_low_dim)
    _, indices_low = nbrs_low.kneighbors(X_low_dim)
    indices_low = indices_low[:, 1:]  # Exclude self

    # Get k nearest neighbors from both spaces
    indices_high_k = high_dim_indexes[:, :k]  # shape (n_samples, k)
    indices_low_k = indices_low[:, :k]  # shape (n_samples, k)

    # Vectorized computation: count overlaps per sample
    combined_indices = np.concatenate((indices_high_k, indices_low_k), axis=1)
    sorted_indices = np.sort(combined_indices, axis=1)
    diffs = np.diff(sorted_indices, axis=1)
    overlaps_per_sample = np.sum(diffs == 0, axis=1)

    # Convert to percentage preservation scores per sample
    preservation_score = (overlaps_per_sample / k) * 100

    return preservation_score


def configure_chart(
    chart: alt.Chart, chart_width: int = 600, chart_height: int = 600, label_font_size: int = 20
) -> alt.Chart:
    """Configure Altair chart with standard styling."""
    return chart.properties(
        width=chart_width,
        height=chart_height,
    ).configure_legend(
        labelFontSize=label_font_size,
        gradientVerticalMaxLength=600,
        gradientThickness=30,
        tickCount=6,
    )


def _ensure_suffix(path: str, suffix: str) -> str:
    """
    Ensure path ends with suffix; if it already has any suffix, leave it.
    Only append when the given suffix is missing and there is no existing suffix.
    """
    p = Path(path)
    if p.suffix:
        return path
    return f"{path}{suffix}"


def _read_csv_flexible(file_handle) -> pd.DataFrame:
    """
    Read a CSV file with flexible delimiter detection and index handling.

    Handles common cases:
    - Tab-separated files (with or without index column)
    - Comma-separated files (with or without index column)
    - Single-column files (just SMILES)

    Args:
        file_handle: File handle or path to read from

    Returns:
        DataFrame with data columns preserved (not consumed as index)
    """
    from io import StringIO

    # Read raw content to analyze
    if hasattr(file_handle, "read"):
        content = file_handle.read()
        if hasattr(file_handle, "seek"):
            file_handle.seek(0)
    else:
        content = file_handle

    # Detect delimiter by sampling first few lines
    sample = content[:4096] if len(content) > 4096 else content

    # Count occurrences of common delimiters in first line
    first_line = sample.split("\n")[0]
    tab_count = first_line.count("\t")
    comma_count = first_line.count(",")

    # Choose delimiter based on counts
    if tab_count > comma_count:
        sep = "\t"
    elif comma_count > 0:
        sep = ","
    else:
        sep = "\t"  # Default to tab

    # Try to read with detected delimiter
    try:
        # First, try reading without index_col to see column structure
        df_test = pd.read_csv(StringIO(content), sep=sep, nrows=5)

        # Check if first column looks like an index (integers or unnamed)
        first_col = df_test.columns[0]

        # Heuristics to detect if first column is an index:
        # 1. Column name is 'Unnamed: 0' or empty
        # 2. Column name is numeric string
        # 3. All values are sequential integers
        is_likely_index = (
            str(first_col).startswith("Unnamed")
            or first_col == ""
            or (isinstance(first_col, str) and first_col.isdigit())
        )

        # Also check if we'd lose all data columns by using index_col=0
        if len(df_test.columns) == 1:
            # Single column - don't use as index!
            is_likely_index = False

        # Read the full file with appropriate settings
        if is_likely_index and len(df_test.columns) > 1:
            df = pd.read_csv(StringIO(content), sep=sep, index_col=0)
        else:
            df = pd.read_csv(StringIO(content), sep=sep)

        return df

    except Exception as e:
        logger.warning(f"Flexible CSV parsing failed: {e}, falling back to simple read")
        return pd.read_csv(StringIO(content))


def _compute_descriptors(
    df: pd.DataFrame,
    smiles_column: str = "smi",
    descriptor_type: Optional[str] = None,
    descriptor_column: Optional[str] = None,
) -> tuple[pd.DataFrame, np.ndarray, str]:
    """
    Compute molecular descriptors for a DataFrame column of SMILES strings.

    This helper function centralizes descriptor computation to avoid code duplication.
    Uses autoencoder encoding by default.

    Args:
        df: DataFrame containing SMILES strings
        smiles_column: Name of the column containing SMILES strings
        descriptor_type: Type of descriptor to compute (None = autoencoder by default)
        descriptor_column: Name for the descriptor column (None = determined by encoder)

    Returns:
        tuple: (df_with_descriptors, descriptor_matrix, column_name) where:
            - df_with_descriptors: DataFrame with descriptor column added
            - descriptor_matrix: NumPy array of descriptor vectors
            - column_name: Name of the descriptor column that was used/created
    """
    encoder = MolecularDescriptorEncoder(
        default_descriptor=descriptor_type or DEFAULT_DESCRIPTOR_TYPE
    )
    column_name = descriptor_column or encoder.column_name()

    # Compute descriptors
    smiles_list = df[smiles_column].tolist()
    descriptor_matrix = encoder.encode(smiles_list)

    # Store descriptors as lists in DataFrame (for compatibility with existing code)
    # descriptor_matrix is shape (n_molecules, n_features), convert each row to list
    df[column_name] = [vec.tolist() for vec in descriptor_matrix]

    # Use the matrix directly for projection (already in correct format)
    X = descriptor_matrix

    return df, X, column_name


def data_load_and_prep(dataset: str, gtm_model: str):
    """
    Load GTM model and dataset from S3 storage, prepare descriptors and projections.

    This function loads both the CSV dataset and the pickled GTM model from S3 storage
    (or falls back to local filesystem if S3 is not available), then prepares molecular
    descriptors and computes GTM projections.

    Args:
        dataset: Path to dataset CSV file (can be S3 URL, local path, or relative path)
        gtm_model: Path to GTM model file (can be S3 URL, local path, or relative path)

    Returns:
        tuple: (gtm, df, X, resps) where:
            - gtm: Loaded GTM model object
            - df: DataFrame with molecular data and descriptors
            - X: Molecular descriptor matrix
            - resps: GTM response matrix with shape (n_molecules, n_nodes)

    Raises:
        FileNotFoundError: If either file doesn't exist in S3 or locally
        ValueError: If files are invalid or empty
        Exception: If GTM loading or processing fails
    """
    # Normalize filenames if the caller passed bare stems (no extensions)
    gtm_saved_file = _ensure_suffix(gtm_model, ".pkl.gz")
    data_file = _ensure_suffix(dataset, ".csv")

    gtm = None
    try:
        with S3.open(gtm_saved_file, "rb") as f:
            with gzip.open(f, "rb") as gz:
                gtm = dill.load(gz)
    except ModuleNotFoundError as e:
        logger.error(f"Error loading GTM model: {e}")
        raise

    with S3.open(data_file, "r") as f:
        df = _read_csv_flexible(f)

    df = df.reset_index(drop=True)  # Reset index to be 0-based consecutive

    # Normalize SMILES column name to standard 'smi'
    df = normalize_smiles_column(df)
    df = standardize_smiles_column(df, SMILES_COLUMN)
    df = df.dropna(subset=[SMILES_COLUMN]).reset_index(drop=True)

    # Compute descriptors using autoencoder by default
    df, X, _ = _compute_descriptors(df, smiles_column=SMILES_COLUMN)

    df = df.rename(columns={"assay_chembl_id": "source"})
    resps, _ = gtm.project(torch.from_numpy(X).to(torch.double))
    # Keep resps in the correct shape for ChemographyKit: (n_molecules, n_nodes)
    resps = resps.cpu().numpy()

    logger.info(f"Loaded GTM model: {resps.shape[0]} molecules, {resps.shape[1]} nodes")
    return gtm, df, X, resps


def project_data_on_gtm(dataset_file: str, gtm_model_file: str) -> str:
    """
    Preprocess a dataset for projection onto an existing GTM map.

    This function loads a dataset containing SMILES strings, validates and filters
    molecules, normalizes the SMILES column, and verifies compatibility with the
    specified GTM model. The preprocessed dataset is saved to a new CSV file that
    can be directly used with save_gtm_plot().

    The function addresses common issues when preparing new data for GTM projection:
    - Validates descriptor dimensionality matches GTM model expectations
    - Handles missing or invalid SMILES gracefully with tracking
    - Normalizes SMILES column name to the standard format
    - Adds a 'source' column if missing (using the dataset filename)
    - Provides informative error messages for compatibility issues

    Args:
        dataset_file: Path to the dataset CSV file. Must contain a SMILES column
                     (named 'smi', 'SMILES', 'smiles', or 'Smiles'). Can be S3 URL,
                     local path, or relative path.
        gtm_model_file: Path to the GTM model file (.pkl.gz format). Can be S3 URL,
                       local path, or relative path.

    Returns:
        Success message with the path to the preprocessed dataset CSV file,
        which can be used with save_gtm_plot().

    Raises:
        ValueError: If inputs are invalid or incompatible
        FileNotFoundError: If dataset or model files don't exist
        RuntimeError: If descriptor dimensionality doesn't match GTM model
    """
    # Validate inputs
    if not dataset_file:
        raise ValueError("dataset_file cannot be empty")
    if not gtm_model_file:
        raise ValueError("gtm_model_file cannot be empty")

    logger.info(f"Preprocessing dataset {dataset_file} for GTM model {gtm_model_file}")

    # Normalize file extensions
    gtm_saved_file = _ensure_suffix(gtm_model_file, ".pkl.gz")
    data_file = _ensure_suffix(dataset_file, ".csv")

    # -------------------------------------------------------------------------
    # Step 1: Load the GTM model to validate compatibility
    # -------------------------------------------------------------------------
    logger.debug("Loading GTM model for compatibility check...")
    try:
        with S3.open(gtm_saved_file, "rb") as f:
            with gzip.open(f, "rb") as gz:
                gtm = dill.load(gz)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"GTM model file not found: {gtm_saved_file}") from e
    except ModuleNotFoundError as e:
        logger.error(f"Error loading GTM model (missing module): {e}")
        raise
    except Exception as e:
        raise ValueError(f"Failed to load GTM model: {e}") from e

    # Extract GTM model properties
    try:
        num_nodes = gtm.num_nodes
    except AttributeError as e:
        raise ValueError(
            f"GTM model appears to be corrupted or incompatible. "
            f"Cannot extract model properties: {e}"
        ) from e

    logger.info(f"GTM model loaded: {num_nodes} grid nodes")

    # -------------------------------------------------------------------------
    # Step 2: Load and preprocess the dataset
    # -------------------------------------------------------------------------
    logger.debug("Loading dataset...")
    try:
        with S3.open(data_file, "r") as f:
            df = _read_csv_flexible(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {data_file}") from e
    except Exception as e:
        raise ValueError(f"Failed to read dataset file: {e}") from e

    if df.empty:
        raise ValueError("Dataset file is empty")

    # Reset index to ensure 0-based consecutive indices
    df = df.reset_index(drop=True)
    original_count = len(df)

    # Normalize SMILES column name to standard 'smi'
    try:
        df = normalize_smiles_column(df)
    except ValueError as e:
        raise ValueError(
            f"Dataset must contain a SMILES column. {e}. "
            f"Expected one of: {_SMILES_COLUMN_VARIANTS}"
        ) from e

    logger.info(f"Loaded dataset with {original_count} molecules")

    # -------------------------------------------------------------------------
    # Step 3: Validate and filter SMILES
    # -------------------------------------------------------------------------
    # Track invalid SMILES for reporting
    invalid_smiles_indices = []
    valid_smiles_mask = []

    for idx, smiles in enumerate(df[SMILES_COLUMN]):
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            invalid_smiles_indices.append(idx)
            valid_smiles_mask.append(False)
        else:
            smiles_std = standardize_smiles(smiles)
            if smiles_std is None:
                invalid_smiles_indices.append(idx)
                valid_smiles_mask.append(False)
            else:
                df.at[idx, SMILES_COLUMN] = smiles_std
                valid_smiles_mask.append(True)

    # Filter to valid molecules only
    df_valid = df[valid_smiles_mask].reset_index(drop=True)
    valid_count = len(df_valid)

    if valid_count == 0:
        raise ValueError(
            f"No valid SMILES found in dataset. "
            f"All {original_count} molecules had invalid or empty SMILES."
        )

    if invalid_smiles_indices:
        logger.warning(
            f"Filtered out {len(invalid_smiles_indices)} invalid/empty SMILES "
            f"(indices: {invalid_smiles_indices[:10]}{'...' if len(invalid_smiles_indices) > 10 else ''})"
        )

    # -------------------------------------------------------------------------
    # Step 4: Compute descriptors and validate compatibility via trial projection
    # -------------------------------------------------------------------------
    logger.debug("Computing molecular descriptors...")
    df_valid, X, descriptor_col = _compute_descriptors(df_valid, smiles_column=SMILES_COLUMN)

    # Validate descriptor compatibility by doing a trial projection
    logger.debug("Validating descriptor compatibility with GTM model...")
    try:
        # Try projecting a single sample to verify dimensions match
        test_sample = torch.from_numpy(X[:1]).to(torch.double)
        test_resps, _ = gtm.project(test_sample)
        # Verify the response has expected number of nodes
        if test_resps.shape[1] != num_nodes:
            raise RuntimeError(
                f"GTM projection validation failed: expected {num_nodes} nodes, "
                f"got {test_resps.shape[1]}. The model may be corrupted."
            )
    except RuntimeError as e:
        if "size mismatch" in str(e).lower() or "dimension" in str(e).lower():
            raise RuntimeError(
                f"Descriptor dimensionality mismatch: computed descriptors have {X.shape[1]} dimensions, "
                f"but the GTM model was trained with a different descriptor size. "
                f"Ensure the same descriptor type is used. Error: {e}"
            ) from e
        raise

    logger.info(
        f"Descriptor compatibility verified: {X.shape[1]} dimensions, {valid_count} molecules"
    )

    # -------------------------------------------------------------------------
    # Step 5: Prepare the output dataset
    # -------------------------------------------------------------------------
    # Handle missing 'source' column gracefully
    if "source" not in df_valid.columns and "assay_chembl_id" in df_valid.columns:
        df_valid = df_valid.rename(columns={"assay_chembl_id": "source"})

    if "source" not in df_valid.columns:
        df_valid["source"] = Path(dataset_file).stem  # Use filename as source

    # Remove the descriptor column from output (will be recomputed by save_gtm_plot)
    # Keep only essential columns: smi, source, and any other original columns
    if descriptor_col in df_valid.columns:
        df_valid = df_valid.drop(columns=[descriptor_col])

    # -------------------------------------------------------------------------
    # Step 6: Save preprocessed dataset
    # -------------------------------------------------------------------------
    dataset_stem = Path(dataset_file).with_suffix("").stem
    model_stem = Path(gtm_model_file).with_suffix("").stem
    if model_stem.endswith(".pkl"):
        model_stem = model_stem[:-4]  # Remove .pkl if present (from .pkl.gz)

    output_filename = f"{dataset_stem}_preprocessed_for_{model_stem}{CSV_EXTENSION}"

    logger.debug(f"Saving preprocessed dataset to {output_filename}")

    with S3.open(output_filename, "w") as f:
        df_valid.to_csv(f, sep="\t", index=True)

    # -------------------------------------------------------------------------
    # Step 7: Build summary message
    # -------------------------------------------------------------------------
    summary_parts = [
        f"Successfully preprocessed {valid_count} molecules for GTM projection.",
        f"Preprocessed dataset saved to: `{S3.path(output_filename)}`.",
        f"Use save_gtm_plot('{S3.path(output_filename)}', '{gtm_model_file}') to generate the GTM plot.",
    ]

    if invalid_smiles_indices:
        summary_parts.append(
            f"Note: {len(invalid_smiles_indices)} molecules with invalid SMILES were excluded."
        )

    logger.info(f"Preprocessed dataset saved: {output_filename}")
    return " ".join(summary_parts)


def create_activity_landscapes(
    source_activity: pd.DataFrame,
    node_threshold: float = 0.1,
    chart_width: int = 600,
    chart_height: int = 600,
) -> alt.Chart:
    """
    Create density and neighborhood preservation landscapes from GTM responses.

    Args:
        source_activity (pd.DataFrame): Activity data.
        node_threshold (float): Threshold value for node density filtering.
                               Nodes with density below this value will be excluded.
        chart_width (int): Width of the output chart in pixels.
        chart_height (int): Height of the output chart in pixels.

    Returns:
        alt.Chart: Altair chart showing the activity landscape.
    """
    # Creation of both density, magnification and neighborhood preservation landscapes

    # Configure and save the chart
    if "prob" in source_activity.columns:
        chart = altair_discrete_class_landscape(
            source_activity, title="Classification Activity landscape"
        )
        chart = configure_chart(chart, chart_width, chart_height)
    else:
        chart = altair_discrete_regression_landscape(
            source_activity, title="Regression Activity landscape"
        )
        chart = configure_chart(chart, chart_width, chart_height)

    return chart


def _convert_to_nm(value: float, units: str) -> float | None:
    """
    Convert activity value to nanomolar (nM) units.

    Parameters
    ----------
    value : float
        Activity value
    units : str
        Units string (e.g., 'nM', 'uM', 'mM', 'M')

    Returns
    -------
    float | None
        Value in nM, or None if conversion is not possible
    """
    if pd.isna(value):
        return None

    if not units or not isinstance(units, str):
        return None

    units_lower = units.lower().strip()
    if not units_lower:
        return None

    # Conversion factors to nM
    conversion_factors = {
        "nm": 1.0,
        "nanomolar": 1.0,
        "um": 1000.0,
        "μm": 1000.0,
        "micromolar": 1000.0,
        "mm": 1_000_000.0,
        "millimolar": 1_000_000.0,
        "m": 1_000_000_000.0,
        "molar": 1_000_000_000.0,
        "pm": 0.001,
        "picomolar": 0.001,
    }

    # Try exact match first
    if units_lower in conversion_factors:
        return value * conversion_factors[units_lower]

    # Try partial matches (e.g., "nM" -> "nm")
    for unit_key, factor in conversion_factors.items():
        if unit_key in units_lower or units_lower in unit_key:
            return value * factor

    logger.warning(f"Unknown units '{units}' for value {value}, cannot convert to nM")
    return None


def _select_activity_threshold(
    df: pd.DataFrame,
    thresholds_nm: list[float] | None = None,
) -> float | None:
    """
    Select the best activity threshold for a target based on classification rules.

    A threshold is valid if:
    - More than 100 compounds can be successfully classified
    - At least 20 compounds are active
    - Inactives represent more than 50% of the set

    If multiple thresholds satisfy these conditions, the one with active proportion
    closest to 25% is selected.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'value_nm' column containing activity values in nM
    thresholds_nm : list[float]
        List of thresholds to test (in nM)

    Returns
    -------
    float | None
        Selected threshold in nM, or None if no valid threshold found
    """
    if df.empty or "value_nm" not in df.columns:
        return None

    # Filter to only valid values
    valid_df = df[df["value_nm"].notna()].copy()
    if len(valid_df) == 0:
        return None

    valid_thresholds = []

    thresholds = thresholds_nm or [1000.0, 500.0, 100.0, 50.0]
    for threshold_nm in thresholds:
        # Classify compounds based on this threshold
        labels = []
        for _, row in valid_df.iterrows():
            value_nm = row["value_nm"]
            if pd.isna(value_nm):
                labels.append(None)
                continue

            if value_nm < threshold_nm:
                labels.append("active")
            elif value_nm >= 10 * threshold_nm:
                labels.append("inactive")
            else:
                labels.append(None)  # Rejected (between threshold and 10×threshold)

        # Count valid classifications
        valid_labels = [label for label in labels if label is not None]
        if len(valid_labels) <= 100:
            continue

        n_active = sum(1 for label in valid_labels if label == "active")
        n_inactive = sum(1 for label in valid_labels if label == "inactive")

        # Check conditions
        if n_active < 20:
            continue
        if n_inactive / len(valid_labels) <= 0.5:
            continue

        # All conditions met
        active_proportion = n_active / len(valid_labels)
        valid_thresholds.append((threshold_nm, active_proportion, len(valid_labels)))

    if not valid_thresholds:
        return None

    # Select threshold with active proportion closest to 25%
    best_threshold = min(valid_thresholds, key=lambda x: abs(x[1] - 0.25))
    logger.info(
        f"Selected threshold {best_threshold[0]} nM with {best_threshold[1]:.1%} active "
        f"proportion ({best_threshold[2]} compounds)"
    )
    return best_threshold[0]


def _parse_activity_comment(activity_comment: str) -> str | None:
    """
    Parse activity_comment string into 'active' or 'inactive' label.

    Rules:
    - If contains "no", "inconclusive", or "inactive" (case-insensitive) → "inactive"
    - Everything else → "active"

    Parameters
    ----------
    activity_comment : str
        Activity comment string

    Returns
    -------
    str | None
        'active' or 'inactive', or None if input is invalid
    """
    if pd.isna(activity_comment) or not isinstance(activity_comment, str):
        return None

    comment_lower = activity_comment.lower().strip()

    # Check for inactive indicators
    inactive_patterns = ["no", "inconclusive", "inactive"]
    for pattern in inactive_patterns:
        if pattern in comment_lower:
            return "inactive"

    # Everything else is active
    return "active"


def classify_activity_data(
    df: pd.DataFrame,
    target_column: str | None = None,
) -> pd.Series | None:
    """
    Classify ChEMBL activity data into 'active'/'inactive' labels based on activity rules.

    Classification rules:
    1. Inhibition % <= 50% → inactive
    2. Dose-dependent activity < threshold → active
    3. Dose-dependent activity >= 10×threshold → inactive
    4. Compounds with contradictory assignments or unclassifiable → ignored (NaN)

    Threshold selection (for dose-dependent activities):
    - Test thresholds: 1000, 500, 100, 50 nM
    - Threshold is valid if:
      * >100 compounds can be classified
      * At least 20 are active
      * Inactives > 50% of set
    - If multiple thresholds valid, select one with active proportion closest to 25%

    Parameters
    ----------
    df : pd.DataFrame
        ChEMBL activity DataFrame with columns: standard_type, standard_value, standard_units
        Optionally: target_chembl_id or assay_chembl_id for grouping by target
    target_column : str | None
        Column name to group by target (e.g., 'target_chembl_id' or 'assay_chembl_id').
        If None, processes all data together.

    Returns
    -------
    pd.Series | None
        Series with 'active'/'inactive' labels (or NaN for unclassifiable).
        Returns None if insufficient data for classification.
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for classification")
        return None

    required_cols = ["standard_type", "standard_value", "standard_units"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(
            f"Cannot classify activity data: missing columns {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
        return None

    # Create a copy to avoid modifying original
    df_work = df.copy()

    # Convert standard_value to numeric
    df_work["standard_value"] = pd.to_numeric(df_work["standard_value"], errors="coerce")

    # Convert units to nM for dose-dependent activities
    df_work["value_nm"] = df_work.apply(
        lambda row: _convert_to_nm(row["standard_value"], row.get("standard_units", "")),
        axis=1,
    )

    # Initialize labels
    labels = pd.Series([None] * len(df_work), index=df_work.index, dtype=object)

    # Process inhibition percentage data
    inh_mask = df_work["standard_type"].str.contains("inhibition", case=False, na=False)
    if inh_mask.any():
        inh_values = pd.to_numeric(df_work.loc[inh_mask, "standard_value"], errors="coerce")
        # Inhibition % <= 50% → inactive
        labels.loc[inh_mask & (inh_values <= 50)] = "inactive"
        # Inhibition % > 50% → active
        labels.loc[inh_mask & (inh_values > 50)] = "active"

    # Process dose-dependent activity data (Ki, IC50, EC50, Potency)
    dose_types = ["ki", "ic50", "ec50", "potency"]
    dose_mask = df_work["standard_type"].str.lower().isin(dose_types) | df_work[
        "standard_type"
    ].str.contains("|".join(dose_types), case=False, na=False)

    if dose_mask.any():
        dose_df = df_work[dose_mask].copy()

        # Group by target if target_column is provided
        if target_column and target_column in df_work.columns:
            grouped = dose_df.groupby(target_column, dropna=False)
        else:
            # Process all together
            grouped = [(None, dose_df)]

        all_dose_labels = pd.Series([None] * len(dose_df), index=dose_df.index, dtype=object)

        for target_id, target_df in grouped:
            if target_id:
                logger.debug(
                    f"Processing target {target_id} with {len(target_df)} dose-dependent activities"
                )

            # Select threshold for this target
            threshold_nm = _select_activity_threshold(target_df)
            if threshold_nm is None:
                logger.warning(
                    f"{f'Target {target_id}: ' if target_id else ''}"
                    f"No valid threshold found, skipping dose-dependent classification"
                )
                continue

            # Classify based on selected threshold
            for idx in target_df.index:
                value_nm = target_df.loc[idx, "value_nm"]
                if pd.isna(value_nm):
                    continue

                if value_nm < threshold_nm:
                    all_dose_labels.loc[idx] = "active"
                elif value_nm >= 10 * threshold_nm:
                    all_dose_labels.loc[idx] = "inactive"
                # else: remains None (rejected - between threshold and 10×threshold)

        # Update labels for dose-dependent activities
        labels.loc[dose_df.index] = all_dose_labels

    # Handle contradictory assignments
    # Group by molecule (using molecule_chembl_id if available, otherwise use index)
    if "molecule_chembl_id" in df_work.columns:
        mol_groups = df_work.groupby("molecule_chembl_id")
    else:
        # If no molecule ID, treat each row independently
        mol_groups = [(idx, df_work.loc[[idx]]) for idx in df_work.index]

    final_labels = pd.Series([None] * len(df_work), index=df_work.index, dtype=object)

    for mol_id, mol_df in mol_groups:
        mol_indices = mol_df.index
        mol_label_set = set(labels.loc[mol_indices].dropna())

        if len(mol_label_set) == 0:
            # No labels assigned
            final_labels.loc[mol_indices] = None
        elif len(mol_label_set) == 1:
            # Single consistent label
            final_labels.loc[mol_indices] = list(mol_label_set)[0]
        else:
            # Contradictory assignments - set to None (ignored)
            logger.debug(f"Molecule {mol_id} has contradictory labels {mol_label_set}, ignoring")
            final_labels.loc[mol_indices] = None

    # Convert None to NaN for pandas consistency
    final_labels = final_labels.replace([None], pd.NA)

    n_classified = final_labels.notna().sum()
    n_active = (final_labels == "active").sum()
    n_inactive = (final_labels == "inactive").sum()

    if n_classified == 0:
        logger.warning("No compounds could be classified")
        return None

    logger.info(
        f"Classified {n_classified}/{len(df_work)} compounds: "
        f"{n_active} active, {n_inactive} inactive"
    )

    return final_labels


def get_activity_column(
    df: pd.DataFrame,
) -> tuple[pd.Series, Literal["regression", "classification"]]:
    """
    Extract a single 'activity' series from a ChEMBL activity DataFrame.

    Priority order
    --------------
    1. 'pchembl_value'  – negative-log potency values (float)
    2. 'activity_comment' – qualitative labels such as 'Active'/'Inactive' (str)

    Parameters
    ----------
    df : pd.DataFrame
        ChEMBL activity table.

    Returns
    -------
    tuple: (activity_column, activity_type)
        The chosen activity column and its type ('regression' or 'classification')

    Raises
    ------
    ValueError
        When neither column is present in *df* or both columns have only null values.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")

    available_cols = list(df.columns)

    # Check for pchembl_value first (regression)
    if "pchembl_value" in df.columns and df["pchembl_value"].notna().any():
        activity_column = df["pchembl_value"]
        activity_type: Literal["regression", "classification"] = "regression"
        logger.debug("Selected 'pchembl_value' column for regression activity")
        return activity_column, activity_type

    # Check for activity_comment (classification)
    if "activity_comment" in df.columns and df["activity_comment"].notna().any():
        activity_column = df["activity_comment"]
        activity_type = "classification"
        logger.debug("Selected 'activity_comment' column for classification activity")
        return activity_column, activity_type

    # Neither column found or both are all null
    has_pchembl = "pchembl_value" in df.columns
    has_comment = "activity_comment" in df.columns

    if has_pchembl and has_comment:
        raise ValueError(
            f"Input DataFrame contains both 'pchembl_value' and 'activity_comment' columns, "
            f"but both have only null values. Available columns: {available_cols}"
        )
    elif has_pchembl:
        raise ValueError(
            f"Input DataFrame contains 'pchembl_value' column but it has only null values. "
            f"Available columns: {available_cols}"
        )
    elif has_comment:
        raise ValueError(
            f"Input DataFrame contains 'activity_comment' column but it has only null values. "
            f"Available columns: {available_cols}"
        )
    else:
        raise ValueError(
            f"Input DataFrame must contain either 'pchembl_value' or 'activity_comment' columns. "
            f"Available columns: {available_cols}"
        )


def preprocess_gtm_activity_data(
    dataset: str, gtm_model: str, node_threshold: float = DEFAULT_NODE_THRESHOLD
) -> pd.DataFrame:
    """
    Preprocess GTM activity data and generate activity landscapes.

    This function loads a dataset and GTM model, identifies the activity column,
    filters out invalid rows, computes activity landscapes, and saves the results.

    Creates both regression and classification landscapes if sufficient data is available:
    - Regression: Uses pchembl_value when available
    - Classification: First attempts to classify raw activity data (standard_type,
      standard_value, standard_units) based on inhibition percentage and dose-dependent
      activity rules. Falls back to activity_comment if raw data classification fails
      or produces insufficient results.

    Args:
        dataset: Path to dataset file (can be S3 URL, local path, or relative path)
        gtm_model: Path to GTM model file (can be S3 URL, local path, or relative path)
        node_threshold: Threshold below which nodes are excluded (default: DEFAULT_NODE_THRESHOLD)

    Returns:
        pd.DataFrame: Activity landscape data (regression if available, otherwise classification)

    Raises:
        ValueError: If dataset or gtm_model paths are empty or invalid
        FileNotFoundError: If dataset or model files don't exist
        ValueError: If no valid activity column is found in the dataset
    """
    # Input validation
    if not dataset:
        raise ValueError("dataset path cannot be empty")
    if not gtm_model:
        raise ValueError("gtm_model path cannot be empty")
    if not (0 < node_threshold <= 1):
        raise ValueError(f"node_threshold must be between 0 and 1, got {node_threshold}")

    logger.info(f"Preprocessing GTM activity data: dataset={dataset}, model={gtm_model}")

    # Load GTM model and prepare data (includes descriptors and projections)
    gtm, df, X, resps = data_load_and_prep(dataset, gtm_model)
    logger.debug(f"Loaded dataset with {len(df)} molecules and {resps.shape[1]} GTM nodes")

    # Try to get regression activity column first
    regression_column = None
    if "pchembl_value" in df.columns and df["pchembl_value"].notna().any():
        regression_column = df["pchembl_value"]
        logger.info("Found 'pchembl_value' column for regression landscape")

    # Try to get classification activity column
    classification_column = None

    # First, try to classify from raw activity data (preferred method)
    if all(col in df.columns for col in ["standard_type", "standard_value", "standard_units"]):
        # Determine target column for grouping (prefer target_chembl_id, fallback to assay_chembl_id)
        target_col = None
        if "target_chembl_id" in df.columns:
            target_col = "target_chembl_id"
        elif "assay_chembl_id" in df.columns:
            target_col = "assay_chembl_id"

        classification_column = classify_activity_data(df, target_column=target_col)
        if classification_column is not None and classification_column.notna().any():
            n_classified_raw = classification_column.notna().sum()
            logger.info(
                f"Classified activity data from raw values: "
                f"{n_classified_raw} compounds classified"
            )

            # Fill in gaps using activity_comment for rows that weren't classified
            if "activity_comment" in df.columns and df["activity_comment"].notna().any():
                unclassified_mask = classification_column.isna()
                if unclassified_mask.any():
                    # Parse activity_comment for unclassified rows that have activity_comment
                    unclassified_with_comment = unclassified_mask & df["activity_comment"].notna()
                    if unclassified_with_comment.any():
                        activity_comments = df.loc[unclassified_with_comment, "activity_comment"]
                        parsed_labels = activity_comments.apply(_parse_activity_comment)

                        # Fill in the gaps (only where we got valid parsed labels)
                        valid_parsed = parsed_labels.notna()
                        if valid_parsed.any():
                            n_filled = valid_parsed.sum()
                            # Use the indices from parsed_labels (which match unclassified_with_comment)
                            classification_column.loc[parsed_labels[valid_parsed].index] = (
                                parsed_labels[valid_parsed]
                            )
                            logger.info(
                                f"Filled {n_filled} unclassified rows using activity_comment "
                                f"(total classified: {classification_column.notna().sum()})"
                            )
        else:
            logger.debug("Insufficient data to classify from raw activity values")

    # Fallback to activity_comment if raw data classification failed or wasn't possible
    if (
        classification_column is None
        and "activity_comment" in df.columns
        and df["activity_comment"].notna().any()
    ):
        # Parse all activity_comment values
        classification_column = df["activity_comment"].apply(_parse_activity_comment)
        if classification_column.notna().any():
            logger.info(
                f"Using 'activity_comment' column for classification landscape (fallback): "
                f"{classification_column.notna().sum()} compounds classified"
            )

    # Determine which landscapes to create
    create_regression = regression_column is not None
    create_classification = classification_column is not None

    if not create_regression and not create_classification:
        raise ValueError(
            "No valid activity data found. Need either 'pchembl_value' for regression, "
            "or sufficient raw activity data (standard_type, standard_value, standard_units) "
            "for classification (with fallback to 'activity_comment' if available). "
            f"Available columns: {list(df.columns)}"
        )

    results = []

    # Create regression landscape if available
    if create_regression:
        logger.debug("Computing regression activity landscape")
        valid_mask = regression_column.notna()
        df_reg = df[valid_mask].copy()
        resps_reg = resps[valid_mask]
        activity_col_reg = df_reg["pchembl_value"]

        density, density_activity = get_reg_density_matrix(resps_reg, activity_col_reg)
        source_activity_reg = reg_density_to_table(
            density, density_activity, node_threshold=node_threshold
        )

        # Save regression landscape
        path_reg = _ensure_suffix(gtm_model, ".pkl.gz").replace(".pkl.gz", "_regression.csv")
        logger.info(f"Saving regression activity landscape to {path_reg}")
        with S3.open(path_reg, "w") as f:
            source_activity_reg.to_csv(f)
        logger.debug(
            f"Successfully saved regression landscape with shape {source_activity_reg.shape}"
        )
        results.append(source_activity_reg)

    # Create classification landscape if available
    if create_classification:
        logger.debug("Computing classification activity landscape")
        valid_mask = classification_column.notna()
        resps_class = resps[valid_mask]
        activity_col_class = classification_column[valid_mask]

        # Normalize and validate class labels before passing to get_class_density_matrix
        # This ensures consistency and prevents errors from mismatched class labels
        activity_col_class_normalized = activity_col_class.copy()

        # Normalize to lowercase and strip whitespace to handle case/whitespace variations
        # Convert to string first to handle categorical and other non-object dtypes
        if not isinstance(activity_col_class_normalized.dtype, pd.StringDtype):
            activity_col_class_normalized = activity_col_class_normalized.astype(str)
            # Replace 'nan' strings (from NaN conversion) back to actual NaN
            activity_col_class_normalized = activity_col_class_normalized.replace("nan", pd.NA)

        # Apply normalization only to non-NA values
        mask_notna = activity_col_class_normalized.notna()
        if mask_notna.any():
            activity_col_class_normalized.loc[mask_notna] = (
                activity_col_class_normalized.loc[mask_notna].str.lower().str.strip()
            )

        # Final filter to ensure no NaN values remain (defensive)
        final_valid_mask = activity_col_class_normalized.notna()
        if not final_valid_mask.all():
            logger.warning(
                f"Filtering out {(~final_valid_mask).sum()} NaN values after normalization"
            )
            activity_col_class_normalized = activity_col_class_normalized[final_valid_mask]
            resps_class = resps_class[final_valid_mask]

        # Validate that we have at least 2 unique classes
        unique_classes = activity_col_class_normalized.unique()
        n_unique = len(unique_classes)

        logger.debug(
            f"Classification data: {len(activity_col_class_normalized)} samples, "
            f"{n_unique} unique classes: {unique_classes}"
        )

        if n_unique < 2:
            logger.warning(
                f"Insufficient class diversity for classification landscape: "
                f"found {n_unique} unique class(es) {list(unique_classes)}. "
                f"Need at least 2 classes. Value counts: {activity_col_class_normalized.value_counts().to_dict()}"
            )
            # Skip classification landscape creation but don't fail
            # This allows regression landscapes to still be created if available
        else:
            # Validate that all labels are expected values
            expected_classes = {"active", "inactive"}
            unexpected_classes = set(unique_classes) - expected_classes
            if unexpected_classes:
                logger.warning(
                    f"Found unexpected class labels: {unexpected_classes}. "
                    f"Expected only 'active' and 'inactive'. "
                    f"This may cause issues with get_class_density_matrix."
                )

            try:
                # get_class_density_matrix requires class_name parameter to specify the class labels
                # It defaults to ['1', '2'] but we need to pass the actual class names
                density, class_density, class_prob = get_class_density_matrix(
                    resps_class,
                    activity_col_class_normalized,
                    class_name=sorted(unique_classes.tolist()),  # Sort for consistent ordering
                )
            except ValueError as e:
                # Provide more context if get_class_density_matrix fails
                if "'inactive' is not in list" in str(e) or "'active' is not in list" in str(e):
                    logger.error(
                        f"Class label mismatch error in get_class_density_matrix: {e}. "
                        f"Unique classes in data: {unique_classes}, "
                        f"Value counts: {activity_col_class_normalized.value_counts().to_dict()}, "
                        f"Sample values: {activity_col_class_normalized.head(10).tolist()}"
                    )
                    raise ValueError(
                        f"Failed to create classification landscape due to class label mismatch. "
                        f"Found classes: {list(unique_classes)}, "
                        f"but get_class_density_matrix expected different classes. "
                        f"Original error: {e}"
                    ) from e
                raise

            # Create and save classification landscape
            source_activity_class = class_density_to_table(
                density, class_density, class_prob, node_threshold=node_threshold
            )

            # Save classification landscape
            path_class = _ensure_suffix(gtm_model, ".pkl.gz").replace(
                ".pkl.gz", "_classification.csv"
            )
            logger.info(f"Saving classification activity landscape to {path_class}")
            with S3.open(path_class, "w") as f:
                source_activity_class.to_csv(f)
            logger.debug(
                f"Successfully saved classification landscape with shape {source_activity_class.shape}"
            )
            results.append(source_activity_class)

    # Return the first available result (prefer regression if both available)
    # This maintains backward compatibility
    return results[0] if results else None


def load_gtm(dataset: str, gtm_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load GTM model and compute density and neighborhood preservation landscapes.

    Args:
        dataset: Path to dataset file
        gtm_model: Path to GTM model file

    Returns:
        tuple: (source density DataFrame, source neighborhood preservation DataFrame)
    """
    gtm, df, X, resps = data_load_and_prep(dataset, gtm_model)

    # Validate that the dataset is compatible with the GTM model
    # The responses matrix should have shape (n_molecules, n_nodes)
    n_molecules = resps.shape[0]  # Number of molecules
    n_nodes = resps.shape[1]  # Number of GTM grid points

    # Check if the GTM model grid size matches the responses
    expected_nodes = gtm.num_nodes

    if n_nodes != expected_nodes:
        raise ValueError(
            f"GTM model incompatibility: Model grid has {expected_nodes} points, "
            f"but responses matrix has {n_nodes} points. The model may be corrupted."
        )

    logger.info(f"GTM model loaded successfully: {n_nodes} grid points, {n_molecules} molecules")

    # K-nearest neighbors for score calculations
    k_hit = 10
    nbrs_high = NearestNeighbors(n_neighbors=k_hit + 1).fit(X)
    _, high_dim_indexes = nbrs_high.kneighbors(X)
    high_dim_indexes = high_dim_indexes[:, 1:]

    # Calculate latent coordinates using ChemographyKit's function
    # resps now has the correct shape (n_molecules, n_nodes)
    coords_df = calculate_latent_coords(resps, correction=True, return_node=True)

    NB_node = calculate_nn_preservation_per_sample(
        X_high_dim=X,
        X_low_dim=coords_df[["x", "y"]].values,
        k_neighbors=k_hit,
        high_dim_indexes=high_dim_indexes,
    )

    # Creation of both density, magnification and neighborhood preservation landscapes
    density, density_NB = get_reg_density_matrix(resps, NB_node)

    source = density_to_table(
        density,
        node_threshold=0.1,
    )

    # Ensure source coordinates are integers (defensive: handle any float conversion)
    if "x" in source.columns:
        source["x"] = source["x"].astype(int)
    if "y" in source.columns:
        source["y"] = source["y"].astype(int)

    source_NB = reg_density_to_table(
        density,
        density_NB,
        node_threshold=0.1,
    )

    NB_score = sum(NB_node) / len(NB_node)
    logger.info(f"Neighborhood preservation score: {NB_score:5.3f}")

    source = source.fillna(0)
    source_NB = source_NB.fillna(0)

    return source, source_NB


def inspect_nodes(source_mols: pd.DataFrame, node_ids: list) -> pd.DataFrame:
    """
    Inspect and extract compounds that belong to specific nodes on the density map.

    Args:
        source_mols (pd.DataFrame): DataFrame containing molecules with node information.
        node_ids (list): List of node IDs to inspect.

    Returns:
        pd.DataFrame: DataFrame of molecules corresponding to the specified nodes.
    """
    df_specific = view_molecules_by_nodes(source_mols, node_ids)
    return df_specific


def analyze_scaffolds(df_specific: pd.DataFrame, max_scaffolds: int = 50) -> alt.Chart:
    """
    Analyze scaffold representations in a given set of molecules and produce a bar chart.

    Args:
        df_specific (pd.DataFrame): DataFrame containing molecules from a specific node region.
        max_scaffolds (int): Maximum number of scaffolds to display (default is 50).

    Returns:
        alt.Chart: Altair bar chart representing the frequency of each scaffold.
    """
    # Compute scaffold SMILES using MurckoScaffold
    df_specific["scaffold_smi"] = df_specific[SMILES_COLUMN].apply(
        MurckoScaffold.MurckoScaffoldSmiles
    )

    # Count occurrences of each scaffold
    scaffold_df = value_counts_df(df_specific, "scaffold_smi")
    scaffold_df["mol"] = scaffold_df.scaffold_smi.apply(_smiles_to_mol_or_none)

    # Create a column for scaffold IDs (required for Altair)
    scaffold_df["Scaffold ID"] = scaffold_df.index

    # Limit to the top max_scaffolds scaffolds
    # NOTE: encode_molecules call removed since we only use Scaffold ID and count (image would be dropped anyway)
    alt_df = scaffold_df[:max_scaffolds].filter(["Scaffold ID", "count"])

    # Create an interactive selection (bound to the legend)
    selection = alt.selection_point(fields=["Scaffold ID"], bind="legend")

    # Create the bar chart for scaffold frequency
    bar_chart = (
        alt.Chart(alt_df)
        .mark_bar(color="steelblue")
        .encode(
            x=alt.X("Scaffold ID:N", title="Scaffold ID"),
            y=alt.Y("count:Q", title="Frequency of appearance"),
            tooltip=[
                alt.Tooltip("Scaffold ID:N", title="Scaffold ID"),
                alt.Tooltip("count:Q", title="Number of examples"),
            ],
        )
        .add_params(selection)
        .properties(width=1000, height=500)
    )

    return bar_chart


def tri(grid: np.array) -> int:
    """
    Vectorized Topographic Ruggedness Index (TRI) calculation.
    Uses convolution to sum neighbor heights and squared heights.
    Handles borders correctly by counting only existing neighbors.
    Returns the mean TRI over the entire grid.

    Definition taken from:
    Riley, S. J., DeGloria, S. D., & Elliot, R. (1999).
    Index that quantifies topographic heterogeneity.
    intermountain Journal of sciences, 5(1-4), 23-27.
    """

    elev = np.asarray(grid, dtype=float)
    # 3×3 kernel of ones
    kernel = np.ones((3, 3), dtype=int)

    # m: number of cells (including center) in each 3×3 window
    m = convolve(np.ones_like(elev), kernel, mode="constant", cval=0)

    # Sum of heights and sum of squared heights in each 3×3 window
    sum_h = convolve(elev, kernel, mode="constant", cval=0)
    sum_h2 = convolve(elev**2, kernel, mode="constant", cval=0)

    # Following formula: TRI^2 = m*h0^2 - 2*h0*sum_h + sum_h2
    tri_sq = m * elev**2 - 2 * elev * sum_h + sum_h2
    tri_val = np.sqrt(tri_sq)

    # Exclude cells with no neighbors (m == 1) to avoid dividing by zero issues if desired
    valid = m > 1

    return float(tri_val[valid].mean()) * 100


def calculate_map_ruggedness(df, gtm):
    """
    Calculate the Topographic Ruggedness Index (TRI) for a GTM density map.

    Args:
        df: DataFrame with descriptor column (autoencoder by default)
        gtm: Fitted GTM model

    Returns:
        str: TRI value as formatted string
    """
    # Find descriptor column - prefer autoencoder, fall back to any descriptor column
    encoder = MolecularDescriptorEncoder()
    descriptor_column = encoder.column_name()

    if descriptor_column not in df.columns:
        # Try to find any descriptor column
        possible_columns = [
            col
            for col in df.columns
            if "descriptor" in col.lower()
            or "embedding" in col.lower()
            or "fingerprint" in col.lower()
        ]
        if possible_columns:
            descriptor_column = possible_columns[0]
            X = np.vstack(
                df[descriptor_column]
                .apply(lambda x: np.array(x) if isinstance(x, list) else x)
                .tolist()
            )
            X = X.astype(np.float64)
        else:
            # Compute descriptors if none found
            try:
                smiles_col = find_smiles_column(df)
            except ValueError as e:
                raise ValueError(
                    "Cannot compute descriptors: DataFrame has no SMILES column and no descriptor columns"
                ) from e
            df, X, descriptor_column = _compute_descriptors(df, smiles_column=smiles_col)
            X = X.astype(np.float64)
    else:
        X = np.vstack(
            df[descriptor_column]
            .apply(lambda x: np.array(x) if isinstance(x, list) else x)
            .tolist()
        )
        X = X.astype(np.float64)
    resps, _ = gtm.project(torch.tensor(X.astype(np.float64)))
    resps = resps.cpu().numpy()
    resps = resps.T
    density = get_density_matrix(resps)

    size = gtm.num_nodes
    assert has_integer_sqrt(
        size
    ), f"The resps array (len {size}) doesn't have an integer square root"
    side_size = int(math.sqrt(size))

    density_grid = density.reshape(side_size, side_size)
    density_grid = np.rot90(density_grid)

    return f"{tri(density_grid)}"


# =============================================================================
# Core GTM Operations
# =============================================================================


def optimize_gtm(df: pd.DataFrame, smiles_column: str = "smi"):
    """
    Optimize GTM hyperparameters and fit the final model.

    Args:
        df: DataFrame containing SMILES column
        smiles_column: Name of the column containing SMILES (default: 'smi')

    Returns:
        tuple: (df with descriptors, fitted GTM model, best score)
    """

    # Compute descriptors using autoencoder by default
    df, X, descriptor_column = _compute_descriptors(df, smiles_column=smiles_column)
    df = df[~df[descriptor_column].isna()]

    n_trials_max = 10
    logger.info(f"Starting GTM optimization with {len(df)} molecules")
    logger.debug(
        f"Sample descriptors from column '{descriptor_column}': {df[descriptor_column].head()}"
    )
    X = X.astype(np.float64)

    # Check CUDA availability and suppress NVML warning
    device = "cpu"  # Default to CPU to avoid CUDA issues
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            torch.tensor([1.0]).cuda()
            device = "cuda"
            logger.info("Using CUDA device for GTM optimization")
        except Exception as e:
            logger.warning(f"CUDA available but not functional: {e}. Falling back to CPU.")
            device = "cpu"
    else:
        logger.info("CUDA not available, using CPU for GTM optimization")

    def shannon_entropy(responsibilities: np.ndarray) -> float:
        """
        Compute the Shannon entropy (in percent) of a GTM landscape.
        """
        cumR = responsibilities.sum(axis=0)
        p = cumR / cumR.sum()
        nonzero = p > 0
        H = -np.sum(p[nonzero] * np.log(p[nonzero]))
        K = responsibilities.shape[1]
        E = H / np.log(K)
        return E

    # Define the objective function for hyperparameter optimization
    def objective(trial):
        # Suggest hyperparameters
        n_basis_functions_sqrt = trial.suggest_int("n_basis_functions_sqrt", 5, 25)
        n_nodes_n_basis_diff = trial.suggest_int("n_nodes_n_basis_diff", 5, 25)
        basis_width = trial.suggest_float("basis_width", 0.1, 10.0)
        reg_coeff = trial.suggest_float("reg_coeff", 0.1, 1000.0)

        num_basis_functions = n_basis_functions_sqrt**2
        num_nodes = (n_basis_functions_sqrt + n_nodes_n_basis_diff) ** 2

        # Initialize and fit GTM using ChemographyKit
        gtm = GTM(
            num_nodes=num_nodes,
            num_basis_functions=num_basis_functions,
            basis_width=basis_width,
            reg_coeff=reg_coeff,
            max_iter=200,
            device=device,
            standardize=False,
            pca_scale=True,
            pca_engine="torch",
        )

        try:
            gtm.fit(torch.from_numpy(X))
            resps, _ = gtm.project(torch.from_numpy(X))
            resps = resps.cpu().numpy()
            entropy = shannon_entropy(resps)
            logger.debug(f"Trial {trial.number}: entropy={entropy:.2f}")
            return entropy
        except Exception as e:
            logger.debug(f"Trial {trial.number} failed: {e}")
            # If the model fails to fit, return a low score
            return -np.inf

    # Initialize study
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

    # Run optimization with proper error handling
    logger.info(f"Running {n_trials_max} optimization trials...")
    study.optimize(objective, n_trials=n_trials_max, n_jobs=1, show_progress_bar=False)

    # Get best parameters
    best_params = study.best_params
    best_score = study.best_value

    if best_score == -np.inf:
        logger.error("All optimization trials failed. Using default parameters.")
        # Use conservative default parameters
        best_params = {
            "n_basis_functions_sqrt": 8,
            "n_nodes_n_basis_diff": 8,
            "basis_width": 1.0,
            "reg_coeff": 100.0,
        }
        best_score = 0.0

    # Fit final model with best parameters
    num_basis_functions = best_params["n_basis_functions_sqrt"] ** 2
    num_nodes = (best_params["n_basis_functions_sqrt"] + best_params["n_nodes_n_basis_diff"]) ** 2

    logger.info(
        f"Fitting final GTM model with {num_nodes} nodes and {num_basis_functions} basis functions"
    )

    gtm = GTM(
        num_nodes=num_nodes,
        num_basis_functions=num_basis_functions,
        basis_width=best_params["basis_width"],
        reg_coeff=best_params["reg_coeff"],
        max_iter=300,
        device=device,
        standardize=False,
        pca_scale=True,
        pca_engine="torch",
    )

    try:
        gtm.fit(torch.from_numpy(X.astype(np.float64)))
        logger.info(f"GTM model fitted successfully. Best score: {best_score:.2f}")
    except Exception as e:
        logger.error(f"Failed to fit final GTM model: {e}")
        raise

    return df, gtm, best_score


def optimize_gtm_model(
    df_csv_path: str, dataset_name: str, gtm_name: str, smiles_column: str, agent: Agent
) -> str:
    """
    Load a dataset of SMILES strings, optimize a Generative Topographic Mapping (GTM)
    model for entropy, store results in the agent's session state,
    and report the entropy score.

    Args:
        df_csv_path: Path to the CSV file containing the data table
        dataset_name: Key under which the cleaned DataFrame will be saved in agent.session_state
        gtm_name: Key under which the trained GTM model will be saved in agent.session_state
        smiles_column: Name of the column in the CSV that holds SMILES strings
        agent: The agent whose session_state dict will be updated

    Returns:
        Human-readable message reporting the best entropy score achieved

    Raises:
        FileNotFoundError: If df_csv_path does not point to an existing CSV file
        ValueError: If smiles_column is missing
    """
    # Validate inputs
    if not df_csv_path:
        raise ValueError("df_csv_path cannot be empty")
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    if not gtm_name:
        raise ValueError("gtm_name cannot be empty")
    if not smiles_column:
        raise ValueError("smiles_column cannot be empty")

    logger.info(f"Starting GTM optimization for {df_csv_path}")

    try:
        # Load the dataset
        logger.debug(f"Loading dataset from {df_csv_path}")
        with S3.open(df_csv_path, "r") as f:
            df = pd.read_csv(f)

        logger.info(f"Loaded dataset with shape {df.shape}")

        # Validate SMILES column exists
        if smiles_column not in df.columns:
            raise ValueError(
                f"Column '{smiles_column}' not found in dataset. Available columns: {list(df.columns)}"
            )

        # Clean the data
        initial_size = len(df)
        df = df.dropna(subset=[smiles_column])
        df = standardize_smiles_column(df, smiles_column)
        df = df.dropna(subset=[smiles_column]).reset_index(drop=True)
        final_size = len(df)

        if final_size == 0:
            raise ValueError(f"No valid SMILES found in column '{smiles_column}'")

        logger.info(
            f"Cleaned dataset: {initial_size} -> {final_size} rows ({initial_size - final_size} rows dropped)"
        )

        # Optimize GTM model
        logger.info("Optimizing GTM with entropy")
        df, gtm, best_score = optimize_gtm(df, smiles_column)

        # Store results in agent session
        if agent.session_state is None:
            agent.session_state = {}
        agent.session_state[dataset_name] = df
        agent.session_state[gtm_name] = gtm

        # Also set as the current session GTM model
        # Generate a path-like identifier for the optimized model
        optimized_model_path = f"{gtm_name}.pkl.gz"
        set_session_gtm_model(agent, gtm, optimized_model_path)

        logger.info(f"GTM optimization completed with entropy: {best_score:.1f}")
        return f"Entropy of the current study: {best_score:.1f}"

    except Exception as e:
        logger.error(f"Error in GTM optimization: {e}")
        raise


def calculate_gtm_ruggedness(dataset_name: str, gtm_name: str, agent: Agent) -> str:
    """
    Compute the Topographic Ruggedness Index (TRI) for a GTM model.

    Args:
        dataset_name: Key under which the DataFrame is stored in agent.session_state
        gtm_name: Key under which the GTM model is stored in agent.session_state
        agent: Agent instance whose session_state holds both the dataset and GTM

    Returns:
        Human-readable message reporting the TRI value

    Raises:
        KeyError: If dataset_name or gtm_name is not present in agent.session_state
        ValueError: If inputs are invalid
    """
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    if not gtm_name:
        raise ValueError("gtm_name cannot be empty")
    if not agent:
        raise ValueError("agent cannot be None")
    if agent.session_state is None:
        raise ValueError("agent.session_state is None - session state not initialized")

    logger.info(f"Calculating map ruggedness for dataset: {dataset_name}, GTM: {gtm_name}")

    try:
        # Validate session state contains required objects
        if gtm_name not in agent.session_state:
            available_keys = list(agent.session_state.keys())
            raise KeyError(
                f"GTM '{gtm_name}' not found in session state. Available: {available_keys}"
            )

        if dataset_name not in agent.session_state:
            available_keys = list(agent.session_state.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in session state. Available: {available_keys}"
            )

        gtm = agent.session_state[gtm_name]
        df = agent.session_state[dataset_name]

        # Calculate TRI
        logger.debug("Computing Topographic Ruggedness Index")
        tri_value = calculate_map_ruggedness(df, gtm)

        logger.info(f"TRI calculated: {tri_value}")
        return f"The TRI is {tri_value} for dataset: {dataset_name} and GTM: {gtm_name}"

    except Exception as e:
        logger.error(f"Error calculating map ruggedness: {e}")
        raise


def save_gtm_and_dataset(dataset_name: str, gtm_name: str, agent: Agent) -> str:
    """
    Save a GTM model and its associated dataset from the agent's session state.

    Args:
        dataset_name: Key under which the DataFrame is stored in agent.session_state
        gtm_name: Key under which the GTM model is stored in agent.session_state
        agent: Agent instance whose session_state contains both the dataset and GTM

    Returns:
        Message containing the paths to the saved files

    Raises:
        KeyError: If dataset_name or gtm_name is not present in agent.session_state
        IOError: If saving either file fails
    """
    if not dataset_name:
        raise ValueError("dataset_name cannot be empty")
    if not gtm_name:
        raise ValueError("gtm_name cannot be empty")
    if not agent:
        raise ValueError("agent cannot be None")
    if agent.session_state is None:
        raise ValueError("agent.session_state is None - session state not initialized")

    logger.info(f"Saving GTM model and dataset: {gtm_name}, {dataset_name}")

    try:
        # Validate session state
        if gtm_name not in agent.session_state:
            available_keys = list(agent.session_state.keys())
            raise KeyError(
                f"GTM '{gtm_name}' not found in session state. Available: {available_keys}"
            )

        if dataset_name not in agent.session_state:
            available_keys = list(agent.session_state.keys())
            raise KeyError(
                f"Dataset '{dataset_name}' not found in session state. Available: {available_keys}"
            )

        gtm = agent.session_state[gtm_name]
        df = agent.session_state[dataset_name]

        # Generate file paths
        saved_df_path = f"{dataset_name}{CSV_EXTENSION}"
        saved_gtm_path = f"{gtm_name}{PKL_GZ_EXTENSION}"

        # Save dataset
        logger.debug(f"Saving dataset to {saved_df_path}")
        with S3.open(saved_df_path, "w") as f:
            df.to_csv(f, sep="\t", index=False)

        # Save GTM model
        logger.debug(f"Saving GTM model to {saved_gtm_path}")
        with S3.open(saved_gtm_path, "wb") as f:
            with gzip.open(f, "wb") as gz:
                dill.dump(gtm, gz)

        logger.info(f"Successfully saved files: {saved_df_path}, {saved_gtm_path}")
        return f"dataset_path: {saved_df_path}; gtm_path: {saved_gtm_path}"

    except Exception as e:
        logger.error(f"Error saving GTM and data: {e}")
        raise


def load_dataframe_from_session(dataframe_name: str, session_key: str, agent: Agent) -> str:
    """
    Load a dataframe from the agent's session state into the pandas tools dataframes dictionary.

    Args:
        dataframe_name: Name to use for the dataframe in the pandas tools system
        session_key: Key under which the DataFrame is stored in agent.session_state
        agent: Agent instance whose session_state contains the dataframe

    Returns:
        Confirmation message with dataframe info

    Raises:
        KeyError: If session_key is not present in agent.session_state
        ValueError: If inputs are invalid
    """
    if not dataframe_name:
        raise ValueError("dataframe_name cannot be empty")
    if not session_key:
        raise ValueError("session_key cannot be empty")
    if not agent:
        raise ValueError("agent cannot be None")
    if agent.session_state is None:
        raise ValueError("agent.session_state is None - session state not initialized")

    logger.info(f"Loading dataframe '{session_key}' from session state as '{dataframe_name}'")

    try:
        if session_key not in agent.session_state:
            available_keys = list(agent.session_state.keys())
            raise KeyError(
                f"Dataframe '{session_key}' not found in session state. Available: {available_keys}"
            )

        df = agent.session_state[session_key]

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Object '{session_key}' is not a DataFrame, got {type(df)}")

        # Store in the pandas tools dataframes dictionary
        pandas_tool: Optional[PandasTools] = None
        tools_attr = getattr(agent, "tools", [])

        if isinstance(tools_attr, list):
            for tool in tools_attr:
                if isinstance(tool, PandasTools) or hasattr(tool, "dataframes"):
                    pandas_tool = tool  # type: ignore[assignment]
                    break
        elif isinstance(tools_attr, PandasTools):
            pandas_tool = tools_attr

        if pandas_tool and hasattr(pandas_tool, "dataframes"):
            pandas_tool.dataframes[dataframe_name] = df  # type: ignore[index]
            logger.info(f"Successfully loaded dataframe '{dataframe_name}' with shape {df.shape}")
            return f"✅ Dataframe '{dataframe_name}' loaded from session state. Shape: {df.shape}, Columns: {list(df.columns)}"

        logger.warning("Could not access pandas tools dataframes dictionary")
        return f"❌ Could not access pandas tools dataframes dictionary. Dataframe '{session_key}' exists in session state with shape {df.shape}"

    except Exception as e:
        logger.error(f"Error loading dataframe from session: {e}")
        raise


def load_gtm_density_matrix(dataset_file: str, gtm_file: str) -> str:
    """
    Load GTM model and dataset from S3 storage, return density matrix information.

    This function loads both the CSV dataset and the pickled GTM model from S3 storage
    (or falls back to local filesystem if S3 is not available), then computes and
    returns the density matrix information.

    Args:
        dataset_file: Path to the CSV dataset file (can be S3 URL, local path, or relative path)
        gtm_file: Path to the pickled GTM model file (can be S3 URL, local path, or relative path)

    Returns:
        Formatted string representation of the density DataFrame

    Raises:
        FileNotFoundError: If either file doesn't exist in S3 or locally
        ValueError: If files are invalid or empty
        Exception: If GTM loading or processing fails

    Examples:
        >>> # Load from S3 URLs
        >>> result = load_gtm_density_matrix("s3://bucket/data.csv", "s3://bucket/model.pkl.gz")

        >>> # Load from relative paths (session-scoped in S3)
        >>> result = load_gtm_density_matrix("data.csv", "model.pkl.gz")

        >>> # Load from local absolute paths
        >>> result = load_gtm_density_matrix("/path/to/data.csv", "/path/to/model.pkl.gz")
    """
    if not dataset_file:
        raise ValueError("dataset_file cannot be empty")
    if not gtm_file:
        raise ValueError("gtm_file cannot be empty")

    logger.info(
        f"Loading GTM density matrix from S3 storage: dataset={dataset_file}, model={gtm_file}"
    )

    try:
        # Load GTM model and dataset using S3 client (with automatic fallback to local filesystem)
        source, _ = load_gtm(dataset_file, gtm_file)

        if source is None or source.empty:
            raise ValueError("Loaded density matrix is empty")

        logger.info(f"Successfully loaded density matrix with shape {source.shape}")

        output = f"""
Density DataFrame:
{df_as_str(source)}
"""
        return output

    except Exception as e:
        logger.error(f"Error loading GTM density matrix from S3: {e}")
        raise


def create_activity_landscapes_tool(
    dataset: str,
    gtm_model: str,
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    chart_width: int = DEFAULT_CHART_WIDTH,
    chart_height: int = DEFAULT_CHART_HEIGHT,
) -> str:
    """
    Create activity landscapes from GTM model and dataset.

    Args:
        dataset: Path to the dataset file
        gtm_model: Path to the GTM model file
        node_threshold: Threshold below which nodes are excluded
        chart_width: Width of the output chart (pixels)
        chart_height: Height of the output chart (pixels)

    Returns:
        Success message with file paths

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If dataset or model files don't exist
    """
    # Validate inputs
    if not dataset:
        raise ValueError("dataset path cannot be empty")
    if not gtm_model:
        raise ValueError("gtm_model path cannot be empty")

    if not (0 < node_threshold <= 1):
        raise ValueError("node_threshold must be between 0 and 1")

    validate_positive_int(chart_width, "chart_width")
    validate_positive_int(chart_height, "chart_height")

    logger.info(f"Creating activity landscapes for {dataset} with model {gtm_model}")

    try:
        # Process the data
        source_activity = preprocess_gtm_activity_data(
            dataset, gtm_model, node_threshold=node_threshold
        )

        # Create the chart
        chart = create_activity_landscapes(
            source_activity, node_threshold, chart_width, chart_height
        )

        # Generate output paths
        s3_base = f"{Path(dataset).stem}_gtm_activity_landscape"
        s3_html = f"{s3_base}{HTML_EXTENSION}"
        s3_png = f"{s3_base}{PNG_EXTENSION}"

        # Save files
        logger.debug(f"Saving activity landscape charts to {s3_html} and {s3_png}")

        # Save HTML version
        with S3.open(s3_html, "w") as sf:
            sf.write(chart.to_html())

        # Save PNG version
        with S3.open(s3_png, "wb") as sf:
            chart.save(sf, format="png")

        logger.info(f"Successfully created activity landscapes: {s3_html}, {s3_png}")
        return f"GTM activity landscape saved to S3: `{S3.path(s3_html)}` and `{S3.path(s3_png)}`"

    except Exception as e:
        logger.error(f"Error creating activity landscapes: {e}")
        raise


def save_gtm_plot(
    dataset_file: str, gtm_model_file: str, mark_nodes: Optional[List[int]] = None
) -> str:
    """
    Generate and save a GTM density + points plot (HTML and PNG).

    Args:
        dataset_file: Path to the dataset file
        gtm_model_file: Path to the GTM model file
        mark_nodes: Optional list of node indices to label on the plot

    Returns:
        Success message with file paths

    Raises:
        ValueError: If inputs are invalid
        FileNotFoundError: If dataset or model files don't exist
    """
    # Validate inputs
    if not dataset_file:
        raise ValueError("dataset_file cannot be empty")
    if not gtm_model_file:
        raise ValueError("gtm_model_file cannot be empty")

    if mark_nodes is not None and not isinstance(mark_nodes, list):
        raise ValueError("mark_nodes must be a list or None")

    logger.info(f"Creating GTM plot for {dataset_file} with model {gtm_model_file}")

    try:
        # Derive output base name
        model_header = Path(gtm_model_file).with_suffix("").stem

        # Load data and prepare
        logger.debug("Loading GTM data and preparing coordinates")
        source, _ = load_gtm(dataset_file, gtm_model_file)
        _, df, _, resps = data_load_and_prep(dataset_file, gtm_model_file)

        # Compute charts and molecule info
        density_chart = altair_discrete_density_landscape(source, title=model_header)
        # resps now has the correct shape (n_molecules, n_nodes) for ChemographyKit
        coords = calculate_latent_coords(resps, correction=True, return_node=True)
        vis_info = encode_molecules(df, smiles_col_name=SMILES_COLUMN).reset_index()[
            [SMILES_COLUMN, "source", "image"]
        ]
        source_mols = pd.concat([coords, vis_info], axis=1)

        # Create points layer
        points = altair_points_chart(
            points_table=source_mols,
            num_nodes=resps.shape[1],  # n_nodes is the second dimension
            points_size=DEFAULT_POINTS_SIZE,
            points_opacity=DEFAULT_POINTS_OPACITY,
            tooltip_columns={SMILES_COLUMN: "Smile: ", "source": "Dataset: ", "image": None},
        )

        # Create optional labels layer for marked nodes
        layers = [density_chart, points]
        if mark_nodes:
            logger.debug(f"Adding labels for {len(mark_nodes)} marked nodes")
            text = (
                alt.Chart(source_mols)
                .mark_text(align="left", baseline="middle", dx=1)
                .encode(
                    x="x:Q",
                    y=alt.Y("y:Q", scale=alt.Scale(reverse=True)),
                    text=alt.condition(
                        alt.FieldOneOfPredicate(field="node_index", oneOf=mark_nodes),
                        "node_index:Q",
                        alt.value(""),
                    ),
                )
            )
            layers.append(text)

        # Combine and configure chart
        chart = alt.layer(*layers).properties(
            width=DEFAULT_CHART_WIDTH, height=DEFAULT_CHART_HEIGHT
        )
        chart = chart.configure_legend(
            labelFontSize=DEFAULT_LEGEND_FONT_SIZE,
            gradientVerticalMaxLength=DEFAULT_GRADIENT_MAX_LENGTH,
            gradientThickness=DEFAULT_GRADIENT_THICKNESS,
            tickCount=DEFAULT_TICK_COUNT,
        )

        # Generate output paths
        s3_html = f"{model_header}_gtm_plot{HTML_EXTENSION}"
        s3_png = f"{model_header}_gtm_plot{PNG_EXTENSION}"

        # Save files
        logger.debug(f"Saving GTM plot to {s3_html} and {s3_png}")

        with S3.open(s3_html, "w") as sf:
            sf.write(chart.to_html())

        with S3.open(s3_png, "wb") as sf:
            chart.save(sf, format="png")

        logger.info(f"Successfully created GTM plot: {s3_html}, {s3_png}")
        return f"GTM plot saved to S3: `{S3.path(s3_html)}` and `{S3.path(s3_png)}`"

    except Exception as e:
        logger.error(f"Error creating GTM plot: {e}")
        raise


# =============================================================================
# Data Operations
# =============================================================================


class GTMData:
    """Container for GTM data and associated molecular information."""

    def __init__(self):
        self.gtm = None
        self.df = None
        self.resps = None
        self.source = None
        self.source_NB = None
        self.coords_mols = None
        self.vis_info = None
        self.source_mols = None
        self.activity_table = None
        self.node_lookup_by_coords = None
        self.node_lookup_by_node = None


def load_and_prepare_gtm_data_with_model(
    dataset: str, gtm_model_path: str, gtm_model: Any
) -> GTMData:
    """
    Load dataset and project onto a pre-loaded GTM model, compute coordinates, and prepare source_mols.

    This variant accepts a pre-loaded GTM model object instead of loading from file.
    This is used when we want to reuse a model from session state.

    Args:
        dataset: Path to the dataset file (can be S3 URL, local path, or relative path)
        gtm_model_path: Path to the GTM model file (for logging/identification)
        gtm_model: Pre-loaded GTM model object

    Returns:
        GTMData object containing all loaded and prepared data

    Raises:
        ValueError: If dataset path is invalid
        FileNotFoundError: If dataset file doesn't exist
        Exception: If processing fails
    """
    if not dataset:
        raise ValueError("dataset path cannot be empty")

    logger.info(
        f"Loading dataset and projecting onto GTM model: dataset={dataset}, model={gtm_model_path}"
    )

    try:
        data = GTMData()
        data.gtm = gtm_model  # Use the provided model

        # Load dataset and prepare descriptors
        data_file = _ensure_suffix(dataset, ".csv")

        with S3.open(data_file, "r") as f:
            df = _read_csv_flexible(f)

        df = df.reset_index(drop=True)

        # Normalize SMILES column name to standard 'smi'
        df = normalize_smiles_column(df)
        df = standardize_smiles_column(df, SMILES_COLUMN)
        df = df.dropna(subset=[SMILES_COLUMN]).reset_index(drop=True)

        # Compute descriptors using autoencoder by default
        df, X, _ = _compute_descriptors(df, smiles_column=SMILES_COLUMN)
        df = df.rename(columns={"assay_chembl_id": "source"})

        # Project onto the GTM model
        resps, _ = gtm_model.project(torch.from_numpy(X).to(torch.double))
        resps = resps.cpu().numpy()

        data.df = df
        data.resps = resps

        # Compute molecule coordinates and visualization info
        data.coords_mols = calculate_latent_coords(resps, correction=True, return_node=True)
        data.vis_info = encode_molecules(data.df, smiles_col_name=SMILES_COLUMN).reset_index()

        # Combine coordinate and visualization data
        # NOTE: Excluding 'image' column to prevent context overflow when sampling is returned to LLM
        data.source_mols = pd.concat(
            [data.coords_mols, data.vis_info[[SMILES_COLUMN, "source"]]], axis=1
        )

        # Always compute density and neighborhood preservation
        # Validate that the dataset is compatible with the GTM model
        n_nodes = resps.shape[1]
        expected_nodes = gtm_model.num_nodes

        if n_nodes != expected_nodes:
            raise ValueError(
                f"GTM model incompatibility: Model grid has {expected_nodes} points, "
                f"but responses matrix has {n_nodes} points. The model may be corrupted."
            )

        # K-nearest neighbors for score calculations
        k_hit = 10
        nbrs_high = NearestNeighbors(n_neighbors=k_hit + 1).fit(X)
        _, high_dim_indexes = nbrs_high.kneighbors(X)
        high_dim_indexes = high_dim_indexes[:, 1:]

        NB_node = calculate_nn_preservation_per_sample(
            X_high_dim=X,
            X_low_dim=data.coords_mols[["x", "y"]].values,
            k_neighbors=k_hit,
            high_dim_indexes=high_dim_indexes,
        )

        # Creation of both density, magnification and neighborhood preservation landscapes
        density, density_NB = get_reg_density_matrix(resps, NB_node)

        data.source = density_to_table(density, node_threshold=0.1)
        data.source_NB = reg_density_to_table(density, density_NB, node_threshold=0.1)

        # Ensure source coordinates are integers (defensive: handle any float conversion)
        if "x" in data.source.columns:
            data.source["x"] = data.source["x"].astype(int)
        if "y" in data.source.columns:
            data.source["y"] = data.source["y"].astype(int)

        # Create lookup tables from source (which has integer coordinates, no 0.5 offset)
        data.node_lookup_by_coords, data.node_lookup_by_node = _create_node_lookup_tables(
            data.source
        )

        logger.info(
            f"Successfully loaded and projected dataset with {len(data.source_mols)} molecules"
        )
        return data

    except Exception as e:
        logger.error(f"Error loading GTM data with model: {e}")
        raise


def populate_gtm_data_from_latent_vectors(
    latent_vectors: np.ndarray,
    gtm_model: Any,
    scaler: Any,
    sequences: Optional[List[str]] = None,
    source_df: Optional[pd.DataFrame] = None,
) -> GTMData:
    """
    Populate a GTMData object from pre-computed latent vectors projected onto a latent GTM.

    This is the peptide/latent-space analogue of load_and_prepare_gtm_data_with_model.
    It projects latent vectors onto the GTM, computes coordinates, density, NN preservation,
    and builds lookup tables — everything needed for sampling operations.

    Args:
        latent_vectors: numpy array of shape (n_samples, latent_dim)
        gtm_model: Trained GTM model (from train_latent_gtm)
        scaler: StandardScaler fitted during training
        sequences: Optional list of peptide sequences (parallel to latent_vectors)
        source_df: Optional DataFrame with additional columns to include in source_mols

    Returns:
        Fully populated GTMData object ready for sampling
    """
    if latent_vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got {latent_vectors.ndim}D")

    n_samples = latent_vectors.shape[0]
    logger.info(f"Populating GTMData from {n_samples} latent vectors")

    data = GTMData()
    data.gtm = gtm_model

    # Project latent vectors onto GTM
    resps = project_latent_on_gtm(latent_vectors, gtm_model, scaler)
    data.resps = resps

    # Compute 2D coordinates from responsibilities
    data.coords_mols = calculate_latent_coords(resps, correction=True, return_node=True)

    # Build source_mols DataFrame with SEQUENCE column if available
    if sequences is not None:
        seq_df = pd.DataFrame({SEQUENCE_COLUMN: sequences})
        data.source_mols = pd.concat([data.coords_mols, seq_df], axis=1)
    elif source_df is not None and SEQUENCE_COLUMN in source_df.columns:
        data.source_mols = pd.concat(
            [data.coords_mols, source_df[[SEQUENCE_COLUMN]].reset_index(drop=True)],
            axis=1,
        )
    else:
        data.source_mols = data.coords_mols.copy()

    # Store the full DataFrame for reference
    if source_df is not None:
        data.df = source_df
    elif sequences is not None:
        data.df = pd.DataFrame({SEQUENCE_COLUMN: sequences})

    # Use scaled latent vectors as high-dimensional representation for NN computation
    X_scaled = scaler.transform(latent_vectors).astype(np.float64)

    # Validate GTM compatibility
    n_nodes = resps.shape[1]
    expected_nodes = gtm_model.num_nodes
    if n_nodes != expected_nodes:
        raise ValueError(
            f"GTM model incompatibility: Model has {expected_nodes} nodes, "
            f"but responsibilities matrix has {n_nodes}. The model may be corrupted."
        )

    # Compute density and NN preservation
    k_hit = min(10, n_samples - 1)
    if k_hit > 0:
        nbrs_high = NearestNeighbors(n_neighbors=k_hit + 1).fit(X_scaled)
        _, high_dim_indexes = nbrs_high.kneighbors(X_scaled)
        high_dim_indexes = high_dim_indexes[:, 1:]

        NB_node = calculate_nn_preservation_per_sample(
            X_high_dim=X_scaled,
            X_low_dim=data.coords_mols[["x", "y"]].values,
            k_neighbors=k_hit,
            high_dim_indexes=high_dim_indexes,
        )

        density, density_NB = get_reg_density_matrix(resps, NB_node)
        data.source = density_to_table(density, node_threshold=0.1)
        data.source_NB = reg_density_to_table(density, density_NB, node_threshold=0.1)
    else:
        density = get_density_matrix(resps)
        data.source = density_to_table(density, node_threshold=0.1)

    # Ensure integer coordinates in density table
    if "x" in data.source.columns:
        data.source["x"] = data.source["x"].astype(int)
    if "y" in data.source.columns:
        data.source["y"] = data.source["y"].astype(int)

    # Create coordinate-to-node lookup tables
    data.node_lookup_by_coords, data.node_lookup_by_node = _create_node_lookup_tables(data.source)

    logger.info(
        f"Successfully populated GTMData with {len(data.source_mols)} samples, "
        f"{len(data.source)} density nodes"
    )
    return data


def load_and_prepare_gtm_data(
    dataset: str, gtm_model: str, *, compute_density: bool = False
) -> GTMData:
    """
    Load GTM model and molecular data from S3 storage, compute coordinates, and prepare source_mols.

    This function loads both the CSV dataset and the pickled GTM model from S3 storage
    (or falls back to local filesystem if S3 is not available), then computes molecular
    coordinates, density, and neighborhood preservation tables.

    Args:
        dataset: Path to the dataset file (can be S3 URL, local path, or relative path)
        gtm_model: Path to the GTM model file (can be S3 URL, local path, or relative path)
        compute_density: Deprecated. Density and neighborhood preservation are always computed.

    Returns:
        GTMData object containing all loaded and prepared data

    Raises:
        ValueError: If dataset or gtm_model paths are invalid
        FileNotFoundError: If files don't exist in S3 or locally
        Exception: If GTM loading or processing fails
    """
    if not dataset:
        raise ValueError("dataset path cannot be empty")
    if not gtm_model:
        raise ValueError("gtm_model path cannot be empty")

    logger.info(f"Loading GTM data: dataset={dataset}, model={gtm_model}")

    try:
        data = GTMData()

        # Load GTM and dataset
        data.gtm, data.df, _, data.resps = data_load_and_prep(dataset, gtm_model)
        data.source, data.source_NB = load_gtm(dataset, gtm_model)

        # Ensure source coordinates are integers (defensive: handle any float conversion)
        if "x" in data.source.columns:
            data.source["x"] = data.source["x"].astype(int)
        if "y" in data.source.columns:
            data.source["y"] = data.source["y"].astype(int)

        # Compute molecule coordinates and visualization info
        # data.resps now has the correct shape (n_molecules, n_nodes) for ChemographyKit
        data.coords_mols = calculate_latent_coords(data.resps, correction=True, return_node=True)
        data.vis_info = encode_molecules(data.df, smiles_col_name=SMILES_COLUMN).reset_index()

        # Combine coordinate and visualization data
        # Select only columns that exist in vis_info
        # NOTE: Excluding 'image' to prevent context overflow when sampling is returned to LLM
        vis_cols = [SMILES_COLUMN, "source"]
        available_vis_cols = [col for col in vis_cols if col in data.vis_info.columns]
        data.source_mols = pd.concat([data.coords_mols, data.vis_info[available_vis_cols]], axis=1)

        # Create lookup tables from source (which has integer coordinates, no 0.5 offset)
        data.node_lookup_by_coords, data.node_lookup_by_node = _create_node_lookup_tables(
            data.source
        )

        logger.info(f"Successfully loaded GTM data with {len(data.source_mols)} molecules")
        return data

    except Exception as e:
        logger.error(f"Error loading GTM data: {e}")
        raise


# =============================================================================
# Analysis Operations
# =============================================================================


def analyze_scaffolds_in_nodes(source_mols: pd.DataFrame, list_of_nodes: List[int]) -> str:
    """
    Analyze molecular scaffolds in selected GTM nodes.

    Args:
        source_mols: DataFrame containing molecular data with node_index column
        list_of_nodes: List of node indices to analyze

    Returns:
        String representation of scaffold frequency table

    Raises:
        ValueError: If list_of_nodes is empty or invalid
        AttributeError: If source_mols is None
    """
    if not list_of_nodes:
        raise ValueError("list_of_nodes cannot be empty")

    if not isinstance(list_of_nodes, list) or not all(
        isinstance(node, int) for node in list_of_nodes
    ):
        raise ValueError("list_of_nodes must be a list of integers")

    if source_mols is None:
        raise AttributeError("source_mols cannot be None")

    logger.info(f"Analyzing scaffolds in {len(list_of_nodes)} nodes: {list_of_nodes}")

    try:
        # Get molecules from specified nodes
        df_specific = view_molecules_by_nodes(source_mols, list_of_nodes)

        if df_specific.empty:
            logger.warning(f"No molecules found in nodes {list_of_nodes}")
            return "No molecules found in the specified nodes"

        # Calculate Murcko scaffolds
        df_specific["scaffold_smi"] = df_specific[SMILES_COLUMN].apply(
            MurckoScaffold.MurckoScaffoldSmiles
        )

        # Create frequency table
        scaffold_df = value_counts_df(df_specific, "scaffold_smi")

        # Add scaffold ID for reference
        scaffold_df["Scaffold ID"] = scaffold_df.index.astype(str)

        logger.info(f"Found {len(scaffold_df)} unique scaffolds in specified nodes")
        return df_as_str(scaffold_df)

    except Exception as e:
        logger.error(f"Error analyzing scaffolds: {e}")
        raise


def check_source_datasets_in_nodes(source_mols: pd.DataFrame, list_of_nodes: List[int]) -> str:
    """
    Analyze data source distribution in selected GTM nodes.

    Args:
        source_mols: DataFrame containing molecular data with node_index and source columns
        list_of_nodes: List of node indices to analyze

    Returns:
        String representation of source frequency table

    Raises:
        ValueError: If list_of_nodes is empty or invalid
        AttributeError: If source_mols is None
    """
    if not list_of_nodes:
        raise ValueError("list_of_nodes cannot be empty")

    if not isinstance(list_of_nodes, list) or not all(
        isinstance(node, int) for node in list_of_nodes
    ):
        raise ValueError("list_of_nodes must be a list of integers")

    if source_mols is None:
        raise AttributeError("source_mols cannot be None")

    logger.info(f"Checking source datasets in {len(list_of_nodes)} nodes: {list_of_nodes}")

    try:
        # Get molecules from specified nodes
        df_specific = view_molecules_by_nodes(source_mols, list_of_nodes)

        if df_specific.empty:
            logger.warning(f"No molecules found in nodes {list_of_nodes}")
            return "No molecules found in the specified nodes"

        # Get source distribution
        source_counts = df_specific.source.value_counts()

        logger.info(f"Found molecules from {len(source_counts)} different sources")
        return str(source_counts)

    except Exception as e:
        logger.error(f"Error checking source datasets: {e}")
        raise


def _create_node_lookup_tables(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create coordinate-to-node and node-to-coordinate lookup tables from source DataFrame.

    Ensures coordinates are integers (no 0.5 offset) for consistent lookups.

    Args:
        source: DataFrame with 'x', 'y', and 'nodes' columns (integer coordinates)

    Returns:
        tuple: (node_lookup_by_coords, node_lookup_by_node)
            - node_lookup_by_coords: DataFrame indexed by (x, y) MultiIndex with 'nodes' column
            - node_lookup_by_node: DataFrame indexed by 'nodes' with 'x' and 'y' columns
    """
    if source is None or source.empty:
        raise ValueError("source DataFrame cannot be None or empty")

    if "x" not in source.columns or "y" not in source.columns or "nodes" not in source.columns:
        raise ValueError("source DataFrame must have 'x', 'y', and 'nodes' columns")

    # Ensure coordinates are integers (no float offsets)
    lookup_df = source[["x", "y", "nodes"]].copy()
    lookup_df["x"] = lookup_df["x"].astype(int)
    lookup_df["y"] = lookup_df["y"].astype(int)

    # Create coordinate-to-node lookup (MultiIndex by x, y)
    node_lookup_by_coords = lookup_df.set_index(["x", "y"])[["nodes"]].copy()

    # Create node-to-coordinate lookup (indexed by nodes)
    node_lookup_by_node = lookup_df[["nodes", "x", "y"]].drop_duplicates().set_index("nodes")

    return node_lookup_by_coords, node_lookup_by_node


def get_node_id_from_coords(source: pd.DataFrame, x: int, y: int) -> str:
    """
    Look up which GTM node corresponds to the given x,y coordinates.

    Args:
        source: DataFrame containing coordinate-to-node mapping with x, y, and nodes columns
        x: X coordinate
        y: Y coordinate

    Returns:
        String representation of the node ID

    Raises:
        AttributeError: If source is None
        ValueError: If coordinates are invalid or not found
    """
    if not isinstance(x, int) or not isinstance(y, int):
        raise ValueError("x and y must be integers")

    if source is None:
        raise AttributeError("source DataFrame cannot be None")

    logger.debug(f"Looking up node ID for coordinates ({x}, {y})")

    try:
        # Find matching coordinates
        matches = source[(source.x == x) & (source.y == y)]

        if matches.empty:
            available_coords = source[["x", "y"]].drop_duplicates()
            raise ValueError(
                f"No node found at coordinates ({x}, {y}). "
                f"Available coordinates: {available_coords.values.tolist()[:10]}..."
            )

        node_id = int(matches.nodes.values[0])
        logger.debug(f"Found node ID {node_id} at coordinates ({x}, {y})")
        return str(node_id)

    except Exception as e:
        logger.error(f"Error looking up node ID: {e}")
        raise


# =============================================================================
# Utility Functions
# =============================================================================


def select_nodes_by_density(
    density_table: pd.DataFrame,
    top_n: int = 5,
    min_density: float | None = None,
    use_filtered: bool = True,
) -> List[int]:
    """
    Select nodes from a density table based on density values.

    Args:
        density_table: DataFrame with 'nodes' column and density columns ('density' or 'filtered_density')
        top_n: Number of top nodes to select (by highest density)
        min_density: Optional minimum density threshold to filter nodes
        use_filtered: If True, use 'filtered_density' column; otherwise use 'density' column

    Returns:
        List of node IDs sorted by density (highest first)

    Raises:
        ValueError: If density table is invalid or required columns are missing
    """
    if density_table is None or density_table.empty:
        raise ValueError("density_table cannot be None or empty")

    if "nodes" not in density_table.columns:
        raise ValueError("density_table must contain 'nodes' column")

    # Determine which density column to use
    density_col = "filtered_density" if use_filtered else "density"
    if density_col not in density_table.columns:
        # Fallback to 'density' if 'filtered_density' is not available
        if use_filtered and "density" in density_table.columns:
            logger.warning("Column 'filtered_density' not found, using 'density' instead")
            density_col = "density"
        else:
            raise ValueError(f"density_table must contain '{density_col}' column")

    # Filter by minimum density if specified
    filtered = density_table.copy()
    if min_density is not None:
        filtered = filtered[filtered[density_col] >= min_density]

    if filtered.empty:
        logger.warning(f"No nodes found matching density criteria (min_density={min_density})")
        return []

    # Sort by density (descending) and select top N
    sorted_nodes = filtered.sort_values(by=density_col, ascending=False)
    top_nodes = sorted_nodes.head(top_n)

    node_list = top_nodes["nodes"].tolist()
    logger.debug(f"Selected {len(node_list)} nodes by density: {node_list[:10]}")
    return node_list


def select_nodes_by_activity(
    activity_table: pd.DataFrame,
    activity_column: str | None = None,
    top_n: int = 5,
    min_value: float | None = None,
    ascending: bool = False,
) -> List[int]:
    """
    Select nodes from an activity table based on activity values.

    Args:
        activity_table: DataFrame with 'nodes' column and activity columns
        activity_column: Name of the activity column to use. If None, automatically detects
                        probability columns (ending with '_prob') or numeric columns.
        top_n: Number of top nodes to select
        min_value: Optional minimum activity value to filter nodes
        ascending: If False, select highest values; if True, select lowest values

    Returns:
        List of node IDs sorted by activity (best first)

    Raises:
        ValueError: If activity table is invalid or required columns are missing
    """
    if activity_table is None or activity_table.empty:
        raise ValueError("activity_table cannot be None or empty")

    if "nodes" not in activity_table.columns:
        raise ValueError("activity_table must contain 'nodes' column")

    # Auto-detect activity column if not specified
    if activity_column is None:
        # Look for probability columns first (ending with '_prob')
        prob_cols = [
            col for col in activity_table.columns if col.endswith("_prob") and col != "nodes"
        ]
        if prob_cols:
            activity_column = prob_cols[0]
            logger.debug(f"Auto-detected activity column: {activity_column}")
        else:
            # Look for numeric columns (excluding 'nodes')
            numeric_cols = activity_table.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != "nodes"]
            if numeric_cols:
                activity_column = numeric_cols[0]
                logger.debug(f"Auto-detected activity column: {activity_column}")
            else:
                raise ValueError(
                    "Could not auto-detect activity column. Please specify activity_column."
                )

    if activity_column not in activity_table.columns:
        available_cols = [col for col in activity_table.columns if col != "nodes"]
        raise ValueError(
            f"Activity column '{activity_column}' not found. Available columns: {available_cols}"
        )

    # Filter by minimum value if specified
    filtered = activity_table.copy()
    if min_value is not None:
        filtered = filtered[filtered[activity_column] >= min_value]

    if filtered.empty:
        logger.warning(
            f"No nodes found matching activity criteria (min_value={min_value}, column={activity_column})"
        )
        return []

    # Sort by activity column and select top N
    sorted_nodes = filtered.sort_values(by=activity_column, ascending=ascending)
    top_nodes = sorted_nodes.head(top_n)

    node_list = top_nodes["nodes"].tolist()
    logger.debug(f"Selected {len(node_list)} nodes by activity: {node_list[:10]}")
    return node_list


def sample_molecules_from_nodes(
    source_mols: pd.DataFrame,
    node_ids: List[int],
    sample_size: int | None = None,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Sample molecules from source_mols that belong to the specified nodes.

    Args:
        source_mols: DataFrame containing molecular data with 'node_index' column
        node_ids: List of node indices to sample from
        sample_size: Optional maximum number of molecules to sample (None = all)
        random_state: Optional random seed for reproducible sampling

    Returns:
        DataFrame containing sampled molecules

    Raises:
        ValueError: If source_mols is invalid or node_ids is empty
    """
    if source_mols is None or source_mols.empty:
        raise ValueError("source_mols cannot be None or empty")

    if "node_index" not in source_mols.columns:
        raise ValueError("source_mols must contain 'node_index' column")

    if not node_ids:
        logger.warning("node_ids is empty, returning empty DataFrame")
        return pd.DataFrame()

    # Filter molecules from specified nodes
    filtered = source_mols[source_mols["node_index"].isin(node_ids)].copy()

    if filtered.empty:
        logger.warning(f"No molecules found in nodes: {node_ids}")
        return pd.DataFrame()

    # Sample if sample_size is specified
    if sample_size is not None and sample_size > 0:
        if len(filtered) > sample_size:
            filtered = filtered.sample(n=sample_size, random_state=random_state)

    logger.debug(f"Sampled {len(filtered)} molecules from {len(node_ids)} nodes")
    return filtered


def sample_molecules_by_coordinates(
    source_mols: pd.DataFrame,
    lookup_table: pd.DataFrame,
    coordinates: Iterable[Tuple[int, int] | Sequence[int] | dict],
    sample_size: int | None = None,
    random_state: int | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Sample molecules located at the provided coordinate pairs.

    Args:
        source_mols: DataFrame containing molecular data with 'node_index' column
        lookup_table: DataFrame mapping coordinates to nodes (with 'x', 'y', 'nodes' columns or MultiIndex)
        coordinates: Iterable of coordinate pairs (x, y) or dicts with 'x' and 'y' keys
        sample_size: Optional maximum number of molecules to sample per coordinate (None = all)
        random_state: Optional random seed for reproducible sampling
        allow_missing: If True, skip missing coordinates; if False, raise error for missing coordinates

    Returns:
        DataFrame containing sampled molecules

    Raises:
        ValueError: If inputs are invalid or coordinates are missing (unless allow_missing=True)
    """
    if source_mols is None or source_mols.empty:
        raise ValueError("source_mols cannot be None or empty")

    if "node_index" not in source_mols.columns:
        raise ValueError("source_mols must contain 'node_index' column")

    # Resolve lookup table structure
    if isinstance(lookup_table.index, pd.MultiIndex):
        # MultiIndex with (x, y) coordinates
        coord_df = lookup_table.reset_index()
        if "x" not in coord_df.columns or "y" not in coord_df.columns:
            # MultiIndex names might be different
            if len(lookup_table.index.names) >= 2:
                coord_df = coord_df.rename(
                    columns={
                        lookup_table.index.names[0]: "x",
                        lookup_table.index.names[1]: "y",
                    }
                )
    else:
        coord_df = lookup_table.copy()
        if "x" not in coord_df.columns or "y" not in coord_df.columns:
            raise ValueError(
                "lookup_table must have 'x' and 'y' columns or a MultiIndex with coordinate names"
            )

    if "nodes" not in coord_df.columns:
        raise ValueError("lookup_table must contain 'nodes' column")

    # Convert coordinates to a list of (x, y) tuples with integer coordinates
    # Handle float coordinates by rounding to nearest integer (removes 0.5 offset)
    coord_list = []
    for coord in coordinates:
        if isinstance(coord, dict):
            if "x" not in coord or "y" not in coord:
                raise ValueError("Coordinate dict must have 'x' and 'y' keys")
            # Convert to int, handling float coordinates (e.g., 7.5 -> 8, 14.5 -> 15)
            x = int(round(float(coord["x"])))
            y = int(round(float(coord["y"])))
            coord_list.append((x, y))
        elif isinstance(coord, (tuple, list, Sequence)):
            if len(coord) < 2:
                raise ValueError("Coordinate must have at least 2 elements (x, y)")
            # Convert to int, handling float coordinates (e.g., 7.5 -> 8, 14.5 -> 15)
            x = int(round(float(coord[0])))
            y = int(round(float(coord[1])))
            coord_list.append((x, y))
        else:
            raise ValueError(f"Invalid coordinate format: {type(coord)}")

    # Find node IDs for each coordinate
    node_ids = []
    missing_coords = []
    for x, y in coord_list:
        matches = coord_df[(coord_df["x"] == x) & (coord_df["y"] == y)]
        if matches.empty:
            missing_coords.append((x, y))
        else:
            node_ids.extend(matches["nodes"].tolist())

    if missing_coords and not allow_missing:
        raise ValueError(
            f"Coordinates not found in lookup table: {missing_coords[:10]}{'...' if len(missing_coords) > 10 else ''}"
        )
    elif missing_coords:
        logger.warning(
            f"Skipping {len(missing_coords)} missing coordinates: {missing_coords[:10]}{'...' if len(missing_coords) > 10 else ''}"
        )

    if not node_ids:
        logger.warning("No nodes found for the provided coordinates")
        return pd.DataFrame()

    # Remove duplicates while preserving order
    unique_node_ids = list(dict.fromkeys(node_ids))

    # Sample molecules from the found nodes
    return sample_molecules_from_nodes(
        source_mols, unique_node_ids, sample_size=sample_size, random_state=random_state
    )


def view_molecules_by_nodes(df: pd.DataFrame, node_list: List[int]) -> pd.DataFrame:
    """
    Filters molecules based on node index.

    Args:
        df: The DataFrame containing molecule data with 'node_index' column
        node_list: A list of node indices to filter by

    Returns:
        A filtered DataFrame containing only the molecules assigned to the specified nodes
    """
    return df[df["node_index"].isin(node_list)].copy()


def encode_molecules(
    df: pd.DataFrame,
    smiles_col_name: str = "smi",
) -> pd.DataFrame:
    """
    Prepare molecular data for visualization by attaching base64-encoded molecule images.

    Args:
        df: DataFrame containing SMILES strings
        smiles_col_name: Name of the column containing SMILES strings (default: 'smi')

    Returns:
        DataFrame with molecular data ready for analysis
    """

    if smiles_col_name not in df.columns:
        raise ValueError(f"Column '{smiles_col_name}' not found in dataframe")

    encoded = df.copy()

    smiles_series = encoded[smiles_col_name]

    # Ensure canonical 'smi' column is present for downstream consumers
    if "smi" not in encoded.columns or smiles_col_name != "smi":
        encoded["smi"] = smiles_series

    if "source" not in encoded.columns:
        encoded["source"] = "unknown"

    def _smiles_to_data_uri(smiles: Any) -> Optional[str]:
        if not isinstance(smiles, str) or not smiles:
            return None
        try:
            png_bytes = smiles_to_png_bytes(smiles)
            b64 = base64.b64encode(png_bytes).decode("utf-8")
            return f"data:image/png;base64,{b64}"
        except Exception:
            logger.debug("Failed to encode molecule image for %s", smiles)
            return None

    encoded["image"] = smiles_series.apply(_smiles_to_data_uri)

    return encoded


# =============================================================================
# Latent-Space GTM Operations (for Peptide WAE integration)
# =============================================================================

# Session state keys for latent GTM
SESSION_LATENT_GTM_MODEL_KEY = "_current_latent_gtm_model"
SESSION_LATENT_GTM_SCALER_KEY = "_current_latent_gtm_scaler"


def train_latent_gtm(
    latent_vectors: np.ndarray,
    config: Optional[dict] = None,
) -> tuple:
    """
    Train a GTM on pre-computed latent vectors (e.g. from Peptide WAE).

    Follows the pattern from the Peptides_WAE notebook: StandardScaler → torch float64 → GTM.fit().
    Uses Optuna hyperparameter optimization identical to optimize_gtm but skips descriptor computation.

    Args:
        latent_vectors: numpy array of shape (n_samples, latent_dim)
        config: Optional dict with GTM hyperparameters. If None, uses Optuna optimization.
                Supported keys: n_basis_functions_sqrt, n_nodes_n_basis_diff, basis_width, reg_coeff

    Returns:
        tuple: (gtm_model, scaler, best_score) where:
            - gtm_model: Trained GTM model
            - scaler: StandardScaler fitted on the latent vectors
            - best_score: Shannon entropy score of the trained model
    """
    from sklearn.preprocessing import StandardScaler

    if latent_vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got {latent_vectors.ndim}D")

    n_samples, latent_dim = latent_vectors.shape
    logger.info(f"Training latent GTM on {n_samples} samples with {latent_dim} dimensions")

    # Scale the latent vectors
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(latent_vectors).astype(np.float64)

    # Check CUDA availability
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.tensor([1.0]).cuda()
            device = "cuda"
            logger.info("Using CUDA device for latent GTM training")
        except Exception:
            logger.info("CUDA not functional, using CPU")

    def shannon_entropy(responsibilities: np.ndarray) -> float:
        cumR = responsibilities.sum(axis=0)
        p = cumR / cumR.sum()
        nonzero = p > 0
        H = -np.sum(p[nonzero] * np.log(p[nonzero]))
        K = responsibilities.shape[1]
        return H / np.log(K)

    if config is not None:
        # Direct training with provided config
        n_basis_sqrt = config.get("n_basis_functions_sqrt", 10)
        n_diff = config.get("n_nodes_n_basis_diff", 10)
        basis_width = config.get("basis_width", 1.0)
        reg_coeff = config.get("reg_coeff", 1.0)

        num_basis = n_basis_sqrt**2
        num_nodes = (n_basis_sqrt + n_diff) ** 2

        gtm = GTM(
            num_basis_functions=num_basis,
            num_nodes=num_nodes,
            basis_width=basis_width,
            reg_coeff=reg_coeff,
            device=device,
        )
        gtm.fit(torch.from_numpy(X_scaled).to(torch.float64), n_iterations=200)
        resps, _ = gtm.project(torch.from_numpy(X_scaled).to(torch.float64))
        score = shannon_entropy(resps.cpu().numpy())

        logger.info(f"Latent GTM trained with entropy: {score:.4f}")
        return gtm, scaler, score

    # Optuna optimization
    n_trials_max = 10

    def objective(trial):
        n_basis_sqrt = trial.suggest_int("n_basis_functions_sqrt", 5, 25)
        n_diff = trial.suggest_int("n_nodes_n_basis_diff", 5, 25)
        basis_width = trial.suggest_float("basis_width", 0.1, 10.0)
        reg_coeff = trial.suggest_float("reg_coeff", 0.1, 1000.0)

        num_basis = n_basis_sqrt**2
        num_nodes = (n_basis_sqrt + n_diff) ** 2

        gtm_trial = GTM(
            num_basis_functions=num_basis,
            num_nodes=num_nodes,
            basis_width=basis_width,
            reg_coeff=reg_coeff,
            device=device,
        )
        gtm_trial.fit(torch.from_numpy(X_scaled).to(torch.float64), n_iterations=200)
        resps, _ = gtm_trial.project(torch.from_numpy(X_scaled).to(torch.float64))
        return shannon_entropy(resps.cpu().numpy())

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials_max)

    best = study.best_params
    num_basis = best["n_basis_functions_sqrt"] ** 2
    num_nodes = (best["n_basis_functions_sqrt"] + best["n_nodes_n_basis_diff"]) ** 2

    gtm = GTM(
        num_basis_functions=num_basis,
        num_nodes=num_nodes,
        basis_width=best["basis_width"],
        reg_coeff=best["reg_coeff"],
        device=device,
    )
    gtm.fit(torch.from_numpy(X_scaled).to(torch.float64), n_iterations=200)
    resps, _ = gtm.project(torch.from_numpy(X_scaled).to(torch.float64))
    best_score = shannon_entropy(resps.cpu().numpy())

    logger.info(f"Latent GTM optimized with best entropy: {best_score:.4f}")
    return gtm, scaler, best_score


def project_latent_on_gtm(
    latent_vectors: np.ndarray,
    gtm_model: Any,
    scaler: Any,
) -> np.ndarray:
    """
    Project latent vectors onto an existing WAE-trained GTM.

    Args:
        latent_vectors: numpy array of shape (n_samples, latent_dim)
        gtm_model: Trained GTM model (from train_latent_gtm)
        scaler: StandardScaler fitted during training (from train_latent_gtm)

    Returns:
        responsibilities: numpy array of shape (n_samples, n_nodes)
    """
    if latent_vectors.ndim != 2:
        raise ValueError(f"Expected 2D array, got {latent_vectors.ndim}D")

    X_scaled = scaler.transform(latent_vectors).astype(np.float64)
    resps, _ = gtm_model.project(torch.from_numpy(X_scaled).to(torch.float64))
    return resps.cpu().numpy()


def load_dbaasp_data(
    dbaasp_path: Optional[str] = None,
) -> tuple:
    """
    Load and parse the DBAASP antimicrobial peptide activity dataset.

    Expects a CSV with a SEQUENCE column and binary activity columns for each organism
    (1 = active, 0 = inactive).

    Args:
        dbaasp_path: Path to the DBAASP CSV file. If None, uses DEFAULT_DBAASP_DATA_PATH.

    Returns:
        tuple: (df, sequences, organism_activity, eligible_organisms) where:
            - df: Full DataFrame
            - sequences: List of peptide sequences (space-separated amino acid codes)
            - organism_activity: Dict mapping organism name → array of binary labels
            - eligible_organisms: List of organism names with >= MIN_ORGANISM_DATA_POINTS data points
    """
    path = dbaasp_path or DEFAULT_DBAASP_DATA_PATH

    logger.info(f"Loading DBAASP data from: {path}")
    try:
        with S3.open(path, "r") as f:
            df = pd.read_csv(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"DBAASP data file not found at {path}. "
            f"Download it from the wae_peptides HuggingFace repository or provide a custom path."
        ) from e

    # Find sequence column (case-insensitive)
    seq_col = None
    for col in df.columns:
        if col.upper() == SEQUENCE_COLUMN:
            seq_col = col
            break

    if seq_col is None:
        raise ValueError(
            f"No '{SEQUENCE_COLUMN}' column found in DBAASP data. "
            f"Available columns: {list(df.columns)}"
        )

    sequences = df[seq_col].tolist()

    # Identify organism activity columns (binary 0/1 columns, excluding SEQUENCE)
    organism_activity = {}
    eligible_organisms = []

    for col in df.columns:
        if col.upper() == SEQUENCE_COLUMN:
            continue
        # Check if column is binary (0/1 values only)
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            labels = df[col].values.astype(int)
            organism_activity[col] = labels
            n_active = int(labels.sum())
            n_total = len(labels)
            if n_total >= MIN_ORGANISM_DATA_POINTS:
                eligible_organisms.append(col)
                logger.info(f"  Organism '{col}': {n_active} active / {n_total} total (eligible)")
            else:
                logger.debug(f"  Organism '{col}': {n_active} active / {n_total} total (too few)")

    logger.info(
        f"DBAASP data loaded: {len(sequences)} sequences, "
        f"{len(organism_activity)} organisms, {len(eligible_organisms)} eligible"
    )
    return df, sequences, organism_activity, eligible_organisms


def create_peptide_activity_landscape(
    responsibilities: np.ndarray,
    activity_labels: np.ndarray,
    organism_name: str,
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    chart_width: int = DEFAULT_CHART_WIDTH,
    chart_height: int = DEFAULT_CHART_HEIGHT,
) -> tuple:
    """
    Create a classification activity landscape for a single organism on the WAE GTM.

    Uses ChemographyKit's get_class_density_matrix and altair_discrete_class_landscape
    to produce a binary classification landscape (active/inactive).

    Args:
        responsibilities: numpy array of shape (n_samples, n_nodes) from GTM projection
        activity_labels: numpy array of binary labels (0=inactive, 1=active) for this organism
        organism_name: Name of the organism (for chart title)
        node_threshold: Threshold below which nodes are excluded
        chart_width: Width of the output chart (pixels)
        chart_height: Height of the output chart (pixels)

    Returns:
        tuple: (chart, landscape_table) where:
            - chart: Altair chart object
            - landscape_table: DataFrame with landscape data (x, y, nodes, prob, etc.)
    """
    # Use ChemographyKit to compute classification landscape
    class_density = get_class_density_matrix(responsibilities, activity_labels)
    landscape_table = class_density_to_table(class_density, node_threshold=node_threshold)

    # Create the chart
    chart = altair_discrete_class_landscape(
        landscape_table,
        title=f"Antimicrobial Activity Landscape: {organism_name}",
    )
    chart = configure_chart(chart, chart_width, chart_height)

    return chart, landscape_table


def create_peptide_activity_landscapes_tool(
    dbaasp_path: Optional[str],
    latent_vectors: np.ndarray,
    gtm_model: Any,
    scaler: Any,
    organism: str = "all",
    node_threshold: float = DEFAULT_NODE_THRESHOLD,
    chart_width: int = DEFAULT_CHART_WIDTH,
    chart_height: int = DEFAULT_CHART_HEIGHT,
) -> str:
    """
    End-to-end tool: Load DBAASP data, project onto WAE GTM, create activity landscapes.

    Args:
        dbaasp_path: Path to DBAASP CSV. If None, uses default.
        latent_vectors: Pre-computed latent vectors for all sequences in DBAASP data.
        gtm_model: Trained latent GTM model.
        scaler: StandardScaler from latent GTM training.
        organism: Organism name to create landscape for. Use "all" for all eligible organisms.
        node_threshold: Threshold below which nodes are excluded.
        chart_width: Width of the output chart (pixels).
        chart_height: Height of the output chart (pixels).

    Returns:
        Summary message with paths to saved landscape files.
    """
    # Load DBAASP data
    df, sequences, organism_activity, eligible_organisms = load_dbaasp_data(dbaasp_path)

    # Project latent vectors onto GTM
    resps = project_latent_on_gtm(latent_vectors, gtm_model, scaler)

    # Determine which organisms to process
    if organism.lower() == "all":
        organisms_to_process = eligible_organisms
    else:
        # Find matching organism (case-insensitive partial match)
        matched = [o for o in organism_activity.keys() if organism.lower() in o.lower()]
        if not matched:
            available = ", ".join(eligible_organisms[:10])
            raise ValueError(f"Organism '{organism}' not found. Available organisms: {available}")
        organisms_to_process = matched

    if not organisms_to_process:
        return "No organisms with sufficient data found for landscape generation."

    saved_files = []
    for org_name in organisms_to_process:
        labels = organism_activity[org_name]

        try:
            chart, landscape_table = create_peptide_activity_landscape(
                resps, labels, org_name, node_threshold, chart_width, chart_height
            )

            # Save files
            safe_name = org_name.replace(" ", "_").replace(".", "").lower()
            base = f"peptide_activity_landscape_{safe_name}"
            html_path = f"{base}{HTML_EXTENSION}"
            png_path = f"{base}{PNG_EXTENSION}"
            csv_path = f"{base}{CSV_EXTENSION}"

            with S3.open(html_path, "w") as sf:
                sf.write(chart.to_html())

            with S3.open(png_path, "wb") as sf:
                chart.save(sf, format="png")

            with S3.open(csv_path, "w") as sf:
                landscape_table.to_csv(sf, index=False)

            saved_files.append(
                f"**{org_name}**: `{S3.path(html_path)}`, `{S3.path(png_path)}`, `{S3.path(csv_path)}`"
            )
            logger.info(f"Created peptide activity landscape for {org_name}")

        except Exception as e:
            logger.warning(f"Failed to create landscape for {org_name}: {e}")
            saved_files.append(f"**{org_name}**: Failed - {e}")

    summary = f"Created {len(saved_files)} peptide activity landscape(s):\n"
    summary += "\n".join(saved_files)
    return summary
