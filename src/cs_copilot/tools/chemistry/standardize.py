from __future__ import annotations

from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def standardize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean_mol)
        uncharged = _UNCHARGER.uncharge(parent)
        tautomer = _TAUTOMER_ENUMERATOR.Canonicalize(uncharged)

        return Chem.MolToSmiles(tautomer, canonical=True)
    except Exception:
        return None


def standardize_smiles_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Apply SMILES standardization to a DataFrame column in-place.

    Rows where standardization fails (invalid SMILES) will have the column value
    set to None/NaN so callers can drop them with dropna(subset=[col_name]).

    Args:
        df: DataFrame containing the SMILES column.
        col_name: Name of the column holding SMILES strings.

    Returns:
        The same DataFrame with the column values replaced by standardized SMILES.
    """

    df[col_name] = df[col_name].apply(
        lambda s: standardize_smiles(s) if isinstance(s, str) else None
    )
    return df
