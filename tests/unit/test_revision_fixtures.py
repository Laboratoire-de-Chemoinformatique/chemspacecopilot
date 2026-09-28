"""Prospective frozen inputs preserve curated scientific thresholds."""

from pathlib import Path
import importlib.util

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "revision_fixtures", Path(__file__).parents[2] / "scripts/prepare_revision_fixtures.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_curated_view_preserves_intermediate_gap_and_continuous_values():
    frame = pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "pIC50": [7.0, 5.0, 5.5],
            "activity_class": ["active", "inactive", "intermediate"],
            "activity_binary": [1, 0, None],
            "standard_type": ["IC50"] * 3,
            "standard_units": ["nM"] * 3,
            "standard_value": [100, 10000, 3162],
        }
    )
    view = module.curated_view(frame)
    assert view["activity_comment"].iloc[:2].tolist() == ["active", "inactive"]
    assert pd.isna(view["activity_comment"].iloc[2])
    assert view["pIC50"].tolist() == frame["pIC50"].tolist()
    assert view["median_IC50_nM"].tolist() == [100, 10000, 3162]
    assert "standard_type" not in view and "standard_value" not in view
    assert "canonical_smiles" in frame and "activity_comment" not in frame


def test_missing_curated_labels_cannot_silently_be_inferred():
    with pytest.raises(ValueError, match="Missing curated fields"):
        module.curated_view(pd.DataFrame({"canonical_smiles": ["CCO"]}))
