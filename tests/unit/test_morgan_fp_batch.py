#!/usr/bin/env python
"""Unit tests for the parallel Morgan count fingerprint batch helper."""

from __future__ import annotations

import logging

import numpy as np
import pytest

from cs_copilot.tools.chemistry import MolecularDescriptorEncoder
from cs_copilot.tools.chemistry import base_chemistry as _MODULE
from cs_copilot.tools.chemistry.base_chemistry import (
    _resolve_worker_count,
    calc_morgan_fp,
    calc_morgan_fp_batch,
)


class InlineExecutor:
    """In-process stand-in for ``ProcessPoolExecutor`` used by the tests.

    Supports the two-iterable form ``executor.map(func, chunks, nbits_values)``
    that ``calc_morgan_fp_batch`` uses internally.
    """

    def __init__(self, max_workers):
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, *iterables):
        return [func(*args) for args in zip(*iterables, strict=True)]


class TestCalcMorganFpBatch:
    def test_batch_matches_serial_and_preserves_order(self, monkeypatch):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles = [
            "CCO",
            "c1ccccc1O",
            "invalid",
            "CC(=O)O",
            "O=C([O-])C",
            "CCN",
            "not-a-smiles",
            "CCOCC",
        ]
        nbits = 256

        serial_reference = [calc_morgan_fp(smi, nbits) for smi in smiles]
        parallel_result = calc_morgan_fp_batch(smiles, nbits, max_workers=3, min_parallel_rows=1)

        assert len(parallel_result) == len(serial_reference)
        for parallel_fp, serial_fp in zip(parallel_result, serial_reference, strict=True):
            if serial_fp is None:
                assert parallel_fp is None
            else:
                assert parallel_fp is not None
                np.testing.assert_array_equal(parallel_fp, serial_fp)

    def test_small_input_stays_serial_by_default(self, monkeypatch, caplog):
        # Guard: if serial routing is broken the executor would be entered and
        # blow up, making the test fail loudly.
        class FailIfUsed:
            def __init__(self, *args, **kwargs):
                raise AssertionError("ProcessPoolExecutor should not be used for small inputs")

        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", FailIfUsed)

        smiles = ["CCO", "CCN", "CCOCC"]
        with caplog.at_level(logging.INFO, logger=_MODULE.__name__):
            result = calc_morgan_fp_batch(smiles, nbits=128, max_workers=4)

        assert len(result) == 3
        assert all(fp is not None for fp in result)
        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Computing Morgan fingerprints: total_rows=3 mode=serial workers=3 nbits=128" in msg
            for msg in messages
        )

    def test_invalid_smiles_yield_none_in_batch(self, monkeypatch):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles = ["CCO", "definitely-not-a-smiles", "c1ccccc1"]
        result = calc_morgan_fp_batch(smiles, nbits=64, max_workers=2, min_parallel_rows=1)

        assert result[0] is not None
        assert result[1] is None
        assert result[2] is not None

    def test_process_fallback_reuses_serial_path(self, monkeypatch, caplog):
        class FailingExecutor:
            def __init__(self, max_workers):
                self.max_workers = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, *iterables):
                raise RuntimeError("executor failed")

        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", FailingExecutor)

        smiles = ["CCO", "invalid", "CCN"]
        with caplog.at_level(logging.INFO, logger=_MODULE.__name__):
            result = calc_morgan_fp_batch(smiles, nbits=64, max_workers=2, min_parallel_rows=1)

        reference = [calc_morgan_fp(smi, 64) for smi in smiles]
        assert len(result) == 3
        for got, expected in zip(result, reference, strict=True):
            if expected is None:
                assert got is None
            else:
                np.testing.assert_array_equal(got, expected)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Process-based Morgan fingerprint computation failed" in msg for msg in messages)
        assert any("serial_fallback=True" in msg for msg in messages)

    def test_summary_logging_reports_counts(self, monkeypatch, caplog):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles = ["CCO", "invalid", "c1ccccc1", "CCN"]
        with caplog.at_level(logging.INFO, logger=_MODULE.__name__):
            calc_morgan_fp_batch(smiles, nbits=128, max_workers=2, min_parallel_rows=1)

        messages = [record.getMessage() for record in caplog.records]
        assert any(
            "Computing Morgan fingerprints: total_rows=4 mode=processes workers=2 nbits=128" in msg
            for msg in messages
        )
        assert any(
            "Finished Morgan fingerprints:" in msg
            and "success_count=3" in msg
            and "failure_count=1" in msg
            and "serial_fallback=False" in msg
            for msg in messages
        )

    def test_max_workers_must_be_positive(self):
        with pytest.raises(ValueError):
            _resolve_worker_count(0, 10)

    def test_min_parallel_rows_must_be_positive(self):
        with pytest.raises(ValueError):
            calc_morgan_fp_batch(["CCO"], nbits=32, min_parallel_rows=0)


class TestEncoderMorganIntegration:
    def test_encoder_morgan_matches_serial_reference(self, monkeypatch):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "CCOCC", "c1ccccc1O"]
        nbits = 512

        encoder = MolecularDescriptorEncoder()
        matrix = encoder.encode(smiles, descriptor_type="morgan", nbits=nbits)

        assert matrix.shape == (len(smiles), nbits)
        # Serial reference computed directly
        reference = np.vstack([calc_morgan_fp(smi, nbits).astype(np.float64) for smi in smiles])
        np.testing.assert_array_equal(matrix, reference)

    def test_encoder_substitutes_zero_vector_for_invalid_smiles(self, monkeypatch, caplog):
        monkeypatch.setattr(_MODULE, "ProcessPoolExecutor", InlineExecutor)

        smiles = ["CCO", "not-a-smiles", "CCN"]
        nbits = 64

        encoder = MolecularDescriptorEncoder()
        with caplog.at_level(logging.WARNING):
            matrix = encoder.encode(smiles, descriptor_type="morgan", nbits=nbits)

        assert matrix.shape == (3, nbits)
        np.testing.assert_array_equal(matrix[1], np.zeros(nbits, dtype=np.float64))
        assert any(
            "Invalid SMILES 'not-a-smiles'" in record.getMessage() for record in caplog.records
        )
