"""Trusted GTM loading must work for actual compressed bytes and CUDA archives."""

from __future__ import annotations

import gzip
import logging
from types import SimpleNamespace

import dill
import numpy as np
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


class TestLatentCentersFromGtm:
    """Node centres are the inverse step map-guided generation needs."""

    def _gtm(self, n_nodes=4, n_basis=3, n_dims=5):
        return SimpleNamespace(
            phi=torch.arange(n_nodes * n_basis, dtype=torch.float64).reshape(n_nodes, n_basis),
            weights=torch.ones((n_basis, n_dims), dtype=torch.float64),
        )

    def test_returns_one_center_per_node(self):
        centers = gtm_operations.latent_centers_from_gtm(self._gtm())

        assert centers.shape == (4, 5)

    def test_node_ids_are_one_based(self):
        """Landscape tables number nodes from 1, so these do too."""
        gtm = self._gtm()
        every = gtm_operations.latent_centers_from_gtm(gtm)

        selected = gtm_operations.latent_centers_from_gtm(gtm, node_ids=[1, 3])

        assert selected.shape == (2, 5)
        assert (selected[0] == every[0]).all()
        assert (selected[1] == every[2]).all()

    @pytest.mark.parametrize("bad", [[0], [5], [-1]])
    def test_out_of_range_nodes_are_rejected(self, bad):
        with pytest.raises(ValueError, match="within 1..4"):
            gtm_operations.latent_centers_from_gtm(self._gtm(), node_ids=bad)

    def test_scaler_is_applied_when_the_manifold_was_fitted_on_scaled_data(self):
        """train_latent_gtm fits on scaled vectors and returns the scaler."""

        class _Scaler:
            def inverse_transform(self, x):
                return x * 10.0

        plain = gtm_operations.latent_centers_from_gtm(self._gtm())
        unscaled = gtm_operations.latent_centers_from_gtm(self._gtm(), scaler=_Scaler())

        assert (unscaled == plain * 10.0).all()

    def test_accepts_numpy_attributes(self):
        """A manifold restored from a frozen bundle holds arrays, not tensors."""
        gtm = SimpleNamespace(phi=np.ones((3, 2)), weights=np.ones((2, 4)))

        assert gtm_operations.latent_centers_from_gtm(gtm).shape == (3, 4)
