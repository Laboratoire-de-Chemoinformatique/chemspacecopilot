"""Trusted GTM loading must work for actual compressed bytes and CUDA archives."""

from __future__ import annotations

import gzip
from types import SimpleNamespace

import dill
import pytest
import torch

from cs_copilot.tools.chemography import gtm_operations


@pytest.mark.parametrize(
    "compressed,suffix", [(False, ".pkl.gz"), (True, ".pkl"), (True, ".pkl.gz")]
)
def test_compression_is_detected_from_bytes_not_filename(tmp_path, monkeypatch, compressed, suffix):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / f"map{suffix}"
    payload = dill.dumps(SimpleNamespace(device="cpu", value=17))
    path.write_bytes(gzip.compress(payload) if compressed else payload)
    loaded = gtm_operations.load_gtm_model(str(path))
    assert loaded.value == 17
    assert loaded.device == "cpu"


def test_embedded_cuda_storages_and_model_device_are_remapped_for_projection(tmp_path, monkeypatch):
    model = gtm_operations.GTM(
        num_nodes=4,
        num_basis_functions=4,
        basis_width=1.0,
        reg_coeff=0.1,
        device="cpu",
        standardize=False,
        max_iter=2,
    )
    # A tiny fitted map exercises real chemographykit projection after loading.
    points = torch.tensor(
        [[0.0, 0.0, 0.2], [0.0, 1.0, -0.1], [1.0, 0.0, 0.4], [1.0, 1.0, 0.1]], dtype=torch.float64
    )
    model.fit(points)
    expected, _ = model.project(points)
    model.device = torch.device("cuda:0")
    # Encode CUDA storage tags without requiring CUDA hardware in the unit test.
    monkeypatch.setattr(torch.serialization, "location_tag", lambda _storage: "cuda:0")
    payload = dill.dumps(model)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "cuda_map.pkl.gz"
    path.write_bytes(payload)

    with pytest.raises(RuntimeError, match="CUDA"):
        dill.loads(payload)

    loaded = gtm_operations.load_gtm_model(str(path))
    actual, _ = loaded.project(points)
    assert loaded.device == torch.device("cpu")
    assert loaded.weights.device.type == "cpu"
    assert torch.allclose(actual, expected)


def test_cuda_available_preserves_checkpoint_device_and_normal_loader(tmp_path, monkeypatch):
    path = tmp_path / "gpu.pkl"
    path.write_bytes(dill.dumps(SimpleNamespace(device="cuda:1", weights=[1, 2])))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gtm_operations,
        "_CPUStorageUnpickler",
        lambda _stream: pytest.fail("CPU remapping must not run on CUDA hosts"),
    )
    loaded = gtm_operations.load_gtm_model(str(path))
    assert loaded.device == "cuda:1"


def test_corrupt_gzip_is_not_reinterpreted_as_plain_pickle(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = tmp_path / "truncated.pkl.gz"
    path.write_bytes(b"\x1f\x8btruncated")
    with pytest.raises((OSError, EOFError)):
        gtm_operations.load_gtm_model(str(path))
