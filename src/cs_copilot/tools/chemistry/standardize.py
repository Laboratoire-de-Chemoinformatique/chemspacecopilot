from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)
_STANDARDIZE_CACHE_SIZE = 16_384
_STANDARDIZERS: Optional[
    tuple[
        rdMolStandardize.Uncharger,
        rdMolStandardize.TautomerEnumerator,
        rdMolStandardize.LargestFragmentChooser,
    ]
] = None


def _get_standardizers() -> tuple[
    rdMolStandardize.Uncharger,
    rdMolStandardize.TautomerEnumerator,
    rdMolStandardize.LargestFragmentChooser,
]:
    """Create RDKit standardizer helpers lazily per worker process."""
    global _STANDARDIZERS

    if _STANDARDIZERS is None:
        cleanup_params = rdMolStandardize.CleanupParameters()
        _STANDARDIZERS = (
            rdMolStandardize.Uncharger(),
            rdMolStandardize.TautomerEnumerator(),
            rdMolStandardize.LargestFragmentChooser(cleanup_params),
        )
    return _STANDARDIZERS


def _resolve_worker_count(max_workers: Optional[int], string_row_count: int) -> int:
    if string_row_count <= 0:
        return 1

    if max_workers is not None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1 or None")
        return min(max_workers, string_row_count)

    cpu_count = os.cpu_count() or 1
    return min(string_row_count, cpu_count)


def _chunk_size(item_count: int, worker_count: int) -> int:
    return max(1, math.ceil(item_count / worker_count))


def _standardize_chunk(smiles_values: list[str]) -> list[Optional[str]]:
    return [standardize_smiles(smiles) for smiles in smiles_values]


@lru_cache(maxsize=_STANDARDIZE_CACHE_SIZE)
def _standardize_smiles_cached(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        uncharger, tautomer_enumerator, fragment_chooser = _get_standardizers()
        parent = fragment_chooser.choose(clean_mol)
        uncharged = uncharger.uncharge(parent)
        tautomer = tautomer_enumerator.Canonicalize(uncharged)

        return Chem.MolToSmiles(tautomer, canonical=True)
    except Exception:
        return None


def standardize_smiles(smiles: str) -> Optional[str]:
    if not isinstance(smiles, str):
        return None
    return _standardize_smiles_cached(smiles)


def standardize_smiles_column(
    df: pd.DataFrame,
    col_name: str,
    *,
    max_workers: Optional[int] = None,
    min_parallel_rows: int = 64,
) -> pd.DataFrame:
    """Apply SMILES standardization to a DataFrame column in-place.

    Rows where standardization fails (invalid SMILES) will have the column value
    set to None/NaN so callers can drop them with dropna(subset=[col_name]).

    Args:
        df: DataFrame containing the SMILES column.
        col_name: Name of the column holding SMILES strings.

    Returns:
        The same DataFrame with the column values replaced by standardized SMILES.
    """
    if min_parallel_rows < 1:
        raise ValueError("min_parallel_rows must be at least 1")

    column_values = df[col_name].tolist()
    total_rows = len(column_values)
    string_positions = [pos for pos, value in enumerate(column_values) if isinstance(value, str)]
    string_values = [column_values[pos] for pos in string_positions]
    string_row_count = len(string_values)
    worker_count = _resolve_worker_count(max_workers, string_row_count)
    mode = "processes" if string_row_count >= min_parallel_rows and worker_count > 1 else "serial"
    serial_fallback = False

    logger.info(
        "Standardizing SMILES column '%s': total_rows=%d string_rows=%d mode=%s workers=%d",
        col_name,
        total_rows,
        string_row_count,
        mode,
        worker_count,
    )
    started_at = time.perf_counter()

    standardized_values: list[Optional[str]]
    if mode == "processes":
        chunk_size = _chunk_size(string_row_count, worker_count)
        chunks = [
            string_values[start : start + chunk_size]
            for start in range(0, string_row_count, chunk_size)
        ]
        try:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                standardized_values = [
                    standardized
                    for chunk_result in executor.map(_standardize_chunk, chunks)
                    for standardized in chunk_result
                ]
        except Exception:
            logger.warning(
                "Process-based SMILES standardization failed for column '%s'; "
                "falling back to serial",
                col_name,
                exc_info=True,
            )
            standardized_values = [standardize_smiles(smiles) for smiles in string_values]
            serial_fallback = True
    else:
        standardized_values = [standardize_smiles(smiles) for smiles in string_values]

    result_values: list[Optional[str]] = [None] * total_rows
    for pos, standardized in zip(string_positions, standardized_values, strict=True):
        result_values[pos] = standardized

    df[col_name] = result_values

    success_count = sum(value is not None for value in standardized_values)
    failure_count = string_row_count - success_count
    elapsed_s = time.perf_counter() - started_at
    logger.info(
        "Finished standardizing SMILES column '%s': elapsed_s=%.3f success_count=%d "
        "failure_count=%d serial_fallback=%s",
        col_name,
        elapsed_s,
        success_count,
        failure_count,
        serial_fallback,
    )

    return df
