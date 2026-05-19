#!/usr/bin/env python
"""Tests for generic activity column inference."""

import pandas as pd
import pytest

from cs_copilot.tools.chemistry.activity_schema import (
    activity_series_for_landscape,
    build_compound_memory_preview,
    infer_activity_mapping,
    normalize_activity_labels,
)


def test_infers_user_ic50_nm_as_lower_is_better_regression():
    df = pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN"],
            "IC50_nM": [10.0, 100.0],
        }
    )

    mapping = infer_activity_mapping(df)
    values, kind, mapping = activity_series_for_landscape(df, mapping)

    assert mapping.smiles_column == "canonical_smiles"
    assert mapping.activity_column == "IC50_nM"
    assert mapping.activity_kind == "regression"
    assert mapping.activity_semantics == "lower_is_better"
    assert mapping.detected_units == "nm"
    assert kind == "regression"
    assert values.tolist() == pytest.approx([8.0, 7.0])


def test_infers_user_label_activity_as_classification():
    df = pd.DataFrame({"SMILES": ["CCO", "CCN"], "label": ["Active", "inactive"]})

    mapping = infer_activity_mapping(df)
    labels, kind, _mapping = activity_series_for_landscape(df, mapping)

    assert mapping.smiles_column == "SMILES"
    assert mapping.activity_column == "label"
    assert mapping.activity_kind == "classification"
    assert mapping.activity_semantics == "label"
    assert kind == "classification"
    assert labels.tolist() == ["active", "inactive"]


def test_infers_chembl_pchembl_as_higher_is_better_regression():
    df = pd.DataFrame(
        {
            "smi": ["CCO", "CCN"],
            "pchembl_value": [6.1, 8.2],
            "molecule_chembl_id": ["CHEMBL1", "CHEMBL2"],
        }
    )

    mapping = infer_activity_mapping(df)
    preview = build_compound_memory_preview(df, mapping)

    assert mapping.source_format == "chembl"
    assert mapping.activity_column == "pchembl_value"
    assert mapping.activity_semantics == "higher_is_better"
    assert preview[0]["smiles"] == "CCN"
    assert preview[0]["activity"]["score"] == 8.2


def test_compound_preview_allows_smiles_without_activity():
    df = pd.DataFrame({"smiles": ["CCO", "CCN"], "name": ["a", "b"]})

    mapping = infer_activity_mapping(df)
    preview = build_compound_memory_preview(df, mapping)

    assert mapping.activity_column is None
    assert preview == [{"smiles": "CCO"}, {"smiles": "CCN"}]


def test_normalize_activity_labels_handles_common_variants():
    labels = normalize_activity_labels(pd.Series(["Hit", "not active", "unknown"]))

    assert labels.tolist()[:2] == ["active", "inactive"]
    assert pd.isna(labels.iloc[2])
