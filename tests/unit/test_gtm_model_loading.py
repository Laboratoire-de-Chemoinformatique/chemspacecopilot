"""Trusted GTM loading must work for actual compressed bytes and CUDA archives."""

from __future__ import annotations

import gzip
import logging
from types import SimpleNamespace

import dill
import pandas as pd
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

    dataset = tmp_path / "molecules.csv"
    pd.DataFrame({"smi": ["CCO", "CCN", "CCC", "CCCl"]}).to_csv(dataset, index=False)
    monkeypatch.setattr(
        gtm_operations,
        "_prepare_gtm_training_data",
        lambda frame, *_args, **_kwargs: SimpleNamespace(df=frame, X=points.numpy()),
    )
    _map, _frame, _descriptors, projected = gtm_operations.data_load_and_prep(
        str(dataset), str(path), descriptor_type="autoencoder"
    )
    assert torch.allclose(torch.from_numpy(projected), expected)


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


class TestDescriptorBackendGuard:
    """A manifold must refuse descriptors from a different latent model.

    chembed and the deepchemography autoencoder both emit 256-wide latents, so
    a trial projection cannot tell them apart: it succeeds and returns
    meaningless coordinates. The recorded tag is the only defence.
    """

    def test_matching_backend_passes(self):
        gtm = SimpleNamespace(descriptor_backend="chembed")
        gtm_operations.validate_gtm_descriptor_backend(gtm, "chembed")

    def test_aliases_resolve_to_the_same_family(self):
        gtm = SimpleNamespace(descriptor_backend="autoencoder")
        gtm_operations.validate_gtm_descriptor_backend(gtm, "latent")

    @pytest.mark.parametrize(
        ("recorded", "supplied"),
        [("autoencoder", "chembed"), ("chembed", "autoencoder")],
    )
    def test_mismatched_backend_is_refused(self, recorded, supplied):
        gtm = SimpleNamespace(descriptor_backend=recorded)
        with pytest.raises(ValueError, match="meaningless coordinates"):
            gtm_operations.validate_gtm_descriptor_backend(gtm, supplied)

    def test_untagged_manifold_warns_but_is_allowed(self, caplog):
        """Manifolds published before the tag existed must keep working."""
        with caplog.at_level(logging.WARNING):
            gtm_operations.validate_gtm_descriptor_backend(SimpleNamespace(), "autoencoder")

        assert "no descriptor_backend" in caplog.text
