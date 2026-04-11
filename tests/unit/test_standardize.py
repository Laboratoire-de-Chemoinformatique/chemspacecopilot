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

_resolve_worker_count = _MODULE._resolve_worker_count
_standardize_smiles_cached = _MODULE._standardize_smiles_cached
standardize_smiles = _MODULE.standardize_smiles
standardize_smiles_column = _MODULE.standardize_smiles_column


class InlineExecutor:
    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, iterable):
        return [func(item) for item in iterable]


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

    def test_default_worker_count_uses_all_detected_cpus(self, monkeypatch):
        monkeypatch.setattr(_MODULE.os, "cpu_count", lambda: 48)

        assert _resolve_worker_count(None, 10_000) == 48
        assert _resolve_worker_count(None, 4) == 4

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

    def test_process_standardization_matches_serial_and_preserves_order(self, monkeypatch):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles_values = ["CCO", "CCN", "invalid", "c1ccccc1O", None, 123, "CC(=O)O", "O=C([O-])C"]
        serial_df = pd.DataFrame({"smiles": smiles_values})
        parallel_df = pd.DataFrame({"smiles": smiles_values})

        serial_result = standardize_smiles_column(serial_df, "smiles", min_parallel_rows=1000)
        parallel_result = standardize_smiles_column(
            parallel_df,
            "smiles",
            max_workers=2,
            min_parallel_rows=1,
        )

        assert parallel_result["smiles"].tolist() == serial_result["smiles"].tolist()

    def test_default_threshold_uses_process_mode(self, monkeypatch, caplog):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)
        df = pd.DataFrame({"smiles": ["CCO"] * 64})

        with caplog.at_level(logging.INFO):
            standardize_smiles_column(df, "smiles", max_workers=2)

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Standardizing SMILES column 'smiles': total_rows=64 string_rows=64 "
            "mode=processes workers=2" in message
            for message in messages
        )

    def test_summary_logging_reports_mode_and_counts(self, monkeypatch, caplog):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)
        df = pd.DataFrame({"smiles": ["CCO", "invalid", None, "CCN"]})

        with caplog.at_level(logging.INFO):
            standardize_smiles_column(df, "smiles", max_workers=2, min_parallel_rows=1)

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Standardizing SMILES column 'smiles': total_rows=4 string_rows=3 "
            "mode=processes workers=2" in message
            for message in messages
        )
        assert any(
            "Finished standardizing SMILES column 'smiles':" in message
            and "success_count=2" in message
            and "failure_count=1" in message
            and "serial_fallback=False" in message
            for message in messages
        )

    def test_process_fallback_reuses_serial_path(self, monkeypatch, caplog):
        class FailingExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, iterable):
                raise RuntimeError("executor failed")

        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", FailingExecutor)
        df = pd.DataFrame({"smiles": ["CCO", "invalid", None, "CCN"]})

        with caplog.at_level(logging.INFO):
            result = standardize_smiles_column(df, "smiles", max_workers=2, min_parallel_rows=1)

        expected = [
            standardize_smiles("CCO"),
            standardize_smiles("invalid"),
            None,
            standardize_smiles("CCN"),
        ]
        normalized_result = [
            None if pd.isna(value) else value for value in result["smiles"].tolist()
        ]
        assert normalized_result == expected

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Process-based SMILES standardization failed for column 'smiles'; "
            "falling back to serial" in message
            for message in messages
        )
        assert any(
            "Finished standardizing SMILES column 'smiles':" in message
            and "serial_fallback=True" in message
            for message in messages
        )

    def test_small_inputs_remain_serial_by_default(self, caplog):
        df = pd.DataFrame({"smiles": ["CCO", "CCN", None]})

        with caplog.at_level(logging.INFO):
            standardize_smiles_column(df, "smiles", max_workers=2)

        messages = [record.getMessage() for record in caplog.records]
        assert any("mode=serial workers=2" in message for message in messages)
