"""Helpers for detecting SMILES columns in tabular data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

SMILES_COLUMN_EXACT_NAMES = ("smi", "smiles", "SMILES", "Smiles")


def smiles_column_exact_names(preferred_name: str | None = None) -> tuple[str, ...]:
    """Return exact SMILES column names with an optional target-specific preference."""
    if preferred_name is None:
        return SMILES_COLUMN_EXACT_NAMES

    remaining_names = [name for name in SMILES_COLUMN_EXACT_NAMES if name != preferred_name]
    return (preferred_name, *remaining_names)


def find_smiles_column_name(
    columns: Iterable[object],
    *,
    exact_names: Sequence[str] = SMILES_COLUMN_EXACT_NAMES,
) -> str | None:
    """Find a SMILES column by exact priority first, then by case-insensitive containment."""
    column_names = list(columns)

    for exact_name in exact_names:
        if exact_name in column_names:
            return exact_name

    for column_name in column_names:
        if isinstance(column_name, str) and "smiles" in column_name.lower():
            return column_name

    return None


def format_smiles_column_expectation(exact_names: Sequence[str]) -> str:
    """Return a concise human-readable description of accepted SMILES column names."""
    return (
        f"exact names in priority order: {list(exact_names)}, "
        "or any column containing 'smiles' (case-insensitive)"
    )
