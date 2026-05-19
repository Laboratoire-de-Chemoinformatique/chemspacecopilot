"""Unit tests for GTM dataset preprocessing reuse."""

from __future__ import annotations

import gzip
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cs_copilot.tools.chemography import gtm_operations


def _local_s3_open_factory(tmp_path: Path):
    def _open(path: str, mode: str = "r"):
        target = Path(path)
        if not target.is_absolute():
            target = tmp_path / target.name
        target.parent.mkdir(parents=True, exist_ok=True)
        return open(target, mode)

    return _open


def _patch_fast_gtm_optimization(monkeypatch, tmp_path: Path):
    calls = {"standardize": 0, "descriptors": 0, "optimize": []}

    def fake_standardize_smiles_column(df, col_name, **_kwargs):
        calls["standardize"] += 1
        df[col_name] = df[col_name].map(lambda value: f"std:{value}")
        return df

    def fake_compute_descriptors(
        df,
        smiles_column="smi",
        descriptor_type=None,
        descriptor_column=None,
    ):
        calls["descriptors"] += 1
        column = descriptor_column
        if column is None:
            column = (
                "autoencoder_embedding"
                if descriptor_type == "autoencoder"
                else "morgan_fingerprint"
            )
        matrix = np.arange(len(df) * 2, dtype=np.float64).reshape(len(df), 2)
        df = df.copy()
        df[column] = [row.tolist() for row in matrix]
        return df, matrix, column

    def fake_optimize_gtm(
        df,
        smiles_column="smi",
        strategy="low",
        *,
        descriptor_type=None,
        agent=None,
        X=None,
        descriptor_column=None,
    ):
        calls["optimize"].append(
            {
                "df": df.copy(),
                "X": None if X is None else X.copy(),
                "descriptor_column": descriptor_column,
                "descriptor_type": descriptor_type,
            }
        )
        return df, object(), 0.5

    monkeypatch.setattr(gtm_operations.S3, "open", _local_s3_open_factory(tmp_path))
    monkeypatch.setattr(
        gtm_operations,
        "standardize_smiles_column",
        fake_standardize_smiles_column,
    )
    monkeypatch.setattr(gtm_operations, "_compute_descriptors", fake_compute_descriptors)
    monkeypatch.setattr(gtm_operations, "optimize_gtm", fake_optimize_gtm)
    return calls


def test_optimize_gtm_model_reuses_prepared_dataset_cache(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(csv_path, index=False)
    agent = SimpleNamespace(session_state={})
    calls = _patch_fast_gtm_optimization(monkeypatch, tmp_path)

    gtm_operations.optimize_gtm_model(str(csv_path), "dataset_1", "gtm_1", "smiles", agent)
    gtm_operations.optimize_gtm_model(str(csv_path), "dataset_2", "gtm_2", "smiles", agent)

    assert calls["standardize"] == 1
    assert calls["descriptors"] == 1
    assert len(calls["optimize"]) == 2
    assert all(call["X"] is not None for call in calls["optimize"])

    stored = agent.session_state["dataset_1"]
    assert stored["raw_smiles"].tolist() == ["C", "CC"]
    assert stored["smiles"].tolist() == ["std:C", "std:CC"]


def test_optimize_gtm_model_cache_misses_when_descriptor_type_changes(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(csv_path, index=False)
    agent = SimpleNamespace(session_state={})
    calls = _patch_fast_gtm_optimization(monkeypatch, tmp_path)

    gtm_operations.optimize_gtm_model(
        str(csv_path),
        "morgan_dataset",
        "morgan_gtm",
        "smiles",
        agent,
        descriptor_type="morgan",
    )
    gtm_operations.optimize_gtm_model(
        str(csv_path),
        "ae_dataset",
        "ae_gtm",
        "smiles",
        agent,
        descriptor_type="autoencoder",
    )

    assert calls["standardize"] == 2
    assert calls["descriptors"] == 2


def test_optimize_gtm_model_cache_misses_when_dataset_changes(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(csv_path, index=False)
    agent = SimpleNamespace(session_state={})
    calls = _patch_fast_gtm_optimization(monkeypatch, tmp_path)

    gtm_operations.optimize_gtm_model(str(csv_path), "dataset_1", "gtm_1", "smiles", agent)
    pd.DataFrame({"smiles": ["C", "CC", "CCC"]}).to_csv(csv_path, index=False)
    gtm_operations.optimize_gtm_model(str(csv_path), "dataset_2", "gtm_2", "smiles", agent)

    assert calls["standardize"] == 2
    assert calls["descriptors"] == 2


def test_optimize_gtm_model_preserves_existing_raw_smiles_column(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame({"smiles": ["C"], "raw_smiles": ["existing"]}).to_csv(csv_path, index=False)
    agent = SimpleNamespace(session_state={})
    _patch_fast_gtm_optimization(monkeypatch, tmp_path)

    gtm_operations.optimize_gtm_model(str(csv_path), "dataset", "gtm", "smiles", agent)

    stored = agent.session_state["dataset"]
    assert stored["raw_smiles"].tolist() == ["existing"]
    assert stored["raw_smiles_input"].tolist() == ["C"]
    assert stored["smiles"].tolist() == ["std:C"]


def test_optimize_gtm_accepts_precomputed_descriptors(monkeypatch):
    def fail_compute_descriptors(*_args, **_kwargs):
        raise AssertionError("descriptors should not be recomputed")

    class FakeGTM:
        def __init__(self, num_nodes, **_kwargs):
            self.num_nodes = num_nodes

        def fit(self, _tensor):
            return None

        def project(self, tensor):
            responsibilities = gtm_operations.torch.ones(
                (tensor.shape[0], self.num_nodes),
                dtype=gtm_operations.torch.float64,
            )
            return responsibilities, None

    monkeypatch.setattr(gtm_operations, "_compute_descriptors", fail_compute_descriptors)
    monkeypatch.setattr(gtm_operations, "GTM", FakeGTM)

    df = pd.DataFrame(
        {
            "smi": ["C", "CC"],
            "morgan_fingerprint": [[1.0, 0.0], [0.0, 1.0]],
        }
    )
    X = np.array([[1.0, 0.0], [0.0, 1.0]])

    result_df, gtm, best_score = gtm_operations.optimize_gtm(
        df,
        smiles_column="smi",
        strategy="low",
        X=X,
        descriptor_column="morgan_fingerprint",
    )

    assert result_df["morgan_fingerprint"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert gtm.num_nodes > 0
    assert best_score == pytest.approx(1.0)


def test_compute_descriptors_reuses_existing_descriptor_lists(monkeypatch):
    class FailingEncoder:
        def __init__(self, default_descriptor="morgan"):
            self.default_descriptor = default_descriptor

        def column_name(self):
            return "morgan_fingerprint"

        def encode(self, _smiles):
            raise AssertionError("descriptors should not be recomputed")

    monkeypatch.setattr(gtm_operations, "MolecularDescriptorEncoder", FailingEncoder)

    df = pd.DataFrame(
        {
            "smi": ["C", "CC"],
            "morgan_fingerprint": [[1.0, 0.0], [0.0, 1.0]],
        }
    )

    result_df, matrix, column = gtm_operations._compute_descriptors(df, smiles_column="smi")

    assert result_df is df
    assert column == "morgan_fingerprint"
    np.testing.assert_array_equal(matrix, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_compute_descriptors_reuses_existing_descriptor_strings(monkeypatch):
    class FailingEncoder:
        def __init__(self, default_descriptor="morgan"):
            self.default_descriptor = default_descriptor

        def column_name(self):
            return "morgan_fingerprint"

        def encode(self, _smiles):
            raise AssertionError("descriptors should not be recomputed")

    monkeypatch.setattr(gtm_operations, "MolecularDescriptorEncoder", FailingEncoder)

    df = pd.DataFrame(
        {
            "smi": ["C", "CC"],
            "morgan_fingerprint": ["[1.0, 0.0]", "[0.0, 1.0]"],
        }
    )

    _result_df, matrix, column = gtm_operations._compute_descriptors(df, smiles_column="smi")

    assert column == "morgan_fingerprint"
    np.testing.assert_array_equal(matrix, np.array([[1.0, 0.0], [0.0, 1.0]]))


def test_data_load_and_prep_reuses_prepared_dataset_cache(monkeypatch, tmp_path):
    csv_path = tmp_path / "dataset.csv"
    gtm_path = tmp_path / "model.pkl.gz"
    pd.DataFrame({"smiles": ["C", "CC"]}).to_csv(csv_path, index=False)
    with gzip.open(gtm_path, "wb") as handle:
        handle.write(b"dummy")

    class FakeGTM:
        num_nodes = 4

        def project(self, tensor):
            return (
                gtm_operations.torch.ones(
                    (tensor.shape[0], self.num_nodes),
                    dtype=gtm_operations.torch.float64,
                ),
                None,
            )

    calls = _patch_fast_gtm_optimization(monkeypatch, tmp_path)
    monkeypatch.setattr(gtm_operations.dill, "load", lambda _handle: FakeGTM())
    agent = SimpleNamespace(session_state={})

    gtm_operations.data_load_and_prep(str(csv_path), str(gtm_path), agent=agent)
    gtm_operations.data_load_and_prep(str(csv_path), str(gtm_path), agent=agent)

    assert calls["standardize"] == 1
    assert calls["descriptors"] == 1
