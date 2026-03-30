#!/usr/bin/env python
"""Unit tests for SMILES standardization helpers."""

import importlib.util
import logging
from pathlib import Path

import pandas as pd

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src/cs_copilot/tools/chemistry/standardize.py"
_SPEC = importlib.util.spec_from_file_location("test_standardize_module", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_standardize_smiles_cached = _MODULE._standardize_smiles_cached
standardize_smiles = _MODULE.standardize_smiles
standardize_smiles_column = _MODULE.standardize_smiles_column


class TestStandardizeSmilesColumn:
    def test_standardize_smiles_uses_cache_for_repeated_inputs(self):
        _standardize_smiles_cached.cache_clear()

        first = standardize_smiles("C[NH3+]")
        second = standardize_smiles("C[NH3+]")
        cache_info = _standardize_smiles_cached.cache_info()

        assert first == "CN"
        assert second == "CN"
        assert cache_info.hits >= 1
        assert cache_info.currsize >= 1

    def test_standardize_smiles_keeps_largest_fragment_behavior(self):
        assert standardize_smiles("CC(=O)O.[Na+]") == "CC(=O)O"

    def test_serial_standardization_handles_mixed_inputs_in_place(self):
        df = pd.DataFrame({"smiles": ["CCO", "not-a-smiles", None, 123, "C[NH3+]"]})
        original_df = df

        result = standardize_smiles_column(df, "smiles", min_parallel_rows=1000)

        expected = [
            standardize_smiles("CCO"),
            standardize_smiles("not-a-smiles"),
            None,
            None,
            standardize_smiles("C[NH3+]"),
        ]
        assert result is original_df
        normalized_result = [
            None if pd.isna(value) else value for value in result["smiles"].tolist()
        ]
        assert normalized_result == expected

    def test_threaded_standardization_matches_serial_and_preserves_order(self):
        smiles_values = ["CCO", "CCN", "invalid", "c1ccccc1O", None, 123, "CC(=O)O", "O=C([O-])C"]
        serial_df = pd.DataFrame({"smiles": smiles_values})
        threaded_df = pd.DataFrame({"smiles": smiles_values})

        serial_result = standardize_smiles_column(serial_df, "smiles", min_parallel_rows=1000)
        threaded_result = standardize_smiles_column(
            threaded_df,
            "smiles",
            max_workers=2,
            min_parallel_rows=1,
        )

        assert threaded_result["smiles"].tolist() == serial_result["smiles"].tolist()

    def test_summary_logging_reports_mode_and_counts(self, caplog):
        df = pd.DataFrame({"smiles": ["CCO", "invalid", None, "CCN"]})

        with caplog.at_level(logging.INFO):
            standardize_smiles_column(df, "smiles", max_workers=2, min_parallel_rows=1)

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Standardizing SMILES column 'smiles': total_rows=4 string_rows=3 mode=threaded workers=2"
            in message
            for message in messages
        )
        assert any(
            "Finished standardizing SMILES column 'smiles':" in message
            and "success_count=2" in message
            and "failure_count=1" in message
            and "serial_fallback=False" in message
            for message in messages
        )

    def test_small_inputs_remain_serial(self, caplog):
        df = pd.DataFrame({"smiles": ["CCO", "CCN", None]})

        with caplog.at_level(logging.INFO):
            standardize_smiles_column(df, "smiles", max_workers=4, min_parallel_rows=10)

        messages = [record.getMessage() for record in caplog.records]
        assert any("mode=serial workers=2" in message for message in messages)
