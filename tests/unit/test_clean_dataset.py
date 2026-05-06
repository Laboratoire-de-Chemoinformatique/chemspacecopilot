#!/usr/bin/env python
# coding: utf-8
"""Tests for clean molecular dataset artifact preparation."""

import pandas as pd

from cs_copilot.storage import S3
from cs_copilot.storage import client as storage_client
from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset


def test_prepare_clean_dataset_merges_stereo_duplicates_and_writes_artifacts(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    old_prefix = S3.current_prefix()
    S3.set_session_prefix("unit-clean-dataset")

    try:
        df = pd.DataFrame(
            {
                "compound_id": ["cmp_a", "cmp_b", "cmp_c", "bad"],
                "canonical_smiles": [
                    "F[C@H](Cl)Br",
                    "F[C@@H](Cl)Br",
                    "CCO",
                    "not-a-smiles",
                ],
                "IC50_nM": [10.0, 100.0, 50.0, 5.0],
                "assay_chembl_id": ["assay_1", "assay_2", "assay_3", "assay_4"],
            }
        )

        result = prepare_clean_dataset(
            df,
            source_name="stereo_test",
            smiles_column="canonical_smiles",
        )

        clean = result.clean_df
        assert clean.shape[0] == 2
        assert "morgan_fingerprint" not in clean.columns
        assert result.standardization_summary["invalid_smiles_rows"] == 1
        assert result.standardization_summary["duplicate_rows_after_standardization"] == 1
        assert result.standardization_summary["raw_smiles_collapse_groups"] == 1
        assert "/01_chemical_space/datasets/clean/" in result.clean_dataset_path
        assert "/01_chemical_space/descriptors/" in result.descriptor_parquet_path
        assert "/01_chemical_space/standardization/" in result.standardization_report_path

        merged = clean.loc[clean["compound_ids"] == "cmp_a|cmp_b"].iloc[0]
        assert merged["activity_final"] == 7.5
        assert merged["activity_n"] == 2
        assert merged["assay_chembl_ids"] == "assay_1|assay_2"

        clean_from_disk = pd.read_csv(result.clean_dataset_path)
        assert "morgan_fingerprint" not in clean_from_disk.columns
        assert "activity_final" in clean_from_disk.columns

        descriptor_df = pd.read_parquet(result.descriptor_parquet_path)
        assert "morgan_fingerprint" in descriptor_df.columns
        assert "activity_final" in descriptor_df.columns
        assert descriptor_df.shape[0] == clean.shape[0]

        with open(result.standardization_report_path) as report_handle:
            report = report_handle.read()
        assert "Raw SMILES Collapse Examples" in report
        assert "Descriptor Parquet" in report
    finally:
        S3.set_session_prefix(old_prefix)
