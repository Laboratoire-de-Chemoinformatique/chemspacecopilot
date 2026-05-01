"""Unit tests for shared SMILES column detection."""

import pandas as pd
import pytest

from cs_copilot.tools.chemography.gtm_operations import (
    find_smiles_column,
    normalize_smiles_column,
)


def test_find_smiles_column_detects_canonical_smiles():
    df = pd.DataFrame({"canonical_smiles": ["CCO"]})

    assert find_smiles_column(df) == "canonical_smiles"


def test_find_smiles_column_detects_mixed_case_smiles_containment():
    df = pd.DataFrame({"Standardized_SMILES": ["CCO"]})

    assert find_smiles_column(df) == "Standardized_SMILES"


def test_find_smiles_column_prefers_exact_smi_over_derived_name():
    df = pd.DataFrame({"canonical_smiles": ["CCN"], "smi": ["CCO"]})

    assert find_smiles_column(df) == "smi"


def test_normalize_smiles_column_renames_detected_column_to_smi():
    df = pd.DataFrame({"canonical_smiles": ["CCO"], "activity": [1.0]})

    result = normalize_smiles_column(df)

    assert "smi" in result.columns
    assert "canonical_smiles" not in result.columns
    assert result["smi"].tolist() == ["CCO"]
    assert "canonical_smiles" in df.columns


def test_find_smiles_column_raises_clear_error_for_missing_column():
    df = pd.DataFrame({"molecule_id": ["CHEMBL1"]})

    with pytest.raises(ValueError, match="containing 'smiles'"):
        find_smiles_column(df)
