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

    chembed and the retired deepchemography LSTM both emit 256-wide latents, so
    a trial projection cannot tell them apart: it succeeds and returns
    meaningless coordinates. The recorded tag is the only defence.
    """

    def test_matching_backend_passes(self):
        gtm = SimpleNamespace(descriptor_backend="chembed")
        gtm_operations.validate_gtm_descriptor_backend(gtm, "chembed")

    def test_a_manifold_tagged_with_the_retired_lstm_is_refused(self):
        """Maps fitted on LSTM latents survive in users' caches and session state."""
        gtm = SimpleNamespace(descriptor_backend="autoencoder")
        with pytest.raises(ValueError, match="meaningless coordinates"):
            gtm_operations.validate_gtm_descriptor_backend(gtm, "chembed")

    @pytest.mark.parametrize(
        ("recorded", "supplied"),
        [("autoencoder", "chembed"), ("peptide_latent", "chembed"), ("chembed", "morgan")],
    )
    def test_mismatched_backend_is_refused(self, recorded, supplied):
        gtm = SimpleNamespace(descriptor_backend=recorded)
        with pytest.raises(ValueError, match="meaningless coordinates"):
            gtm_operations.validate_gtm_descriptor_backend(gtm, supplied)

    def test_untagged_manifold_warns_but_is_allowed(self, caplog):
        """Manifolds published before the tag existed must keep working."""
        with caplog.at_level(logging.WARNING):
            gtm_operations.validate_gtm_descriptor_backend(SimpleNamespace(), "chembed")

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

    def test_the_manifolds_own_scaler_is_used_when_none_is_passed(self):
        """A dropped scaler returns z-scored centres that a decoder accepts in
        silence, so the manifold carries its own and the caller cannot lose it."""

        class _Scaler:
            def inverse_transform(self, x):
                return x * 10.0

        gtm = self._gtm()
        gtm.latent_scaler = _Scaler()

        assert (
            gtm_operations.latent_centers_from_gtm(gtm)
            == gtm_operations.latent_centers_from_gtm(self._gtm()) * 10.0
        ).all()

    def test_an_explicit_scaler_overrides_the_stored_one(self):
        class _Stored:
            def inverse_transform(self, x):
                return x * 10.0

        class _Explicit:
            def inverse_transform(self, x):
                return x * 2.0

        gtm = self._gtm()
        gtm.latent_scaler = _Stored()

        centers = gtm_operations.latent_centers_from_gtm(gtm, scaler=_Explicit())

        assert (centers == gtm_operations.latent_centers_from_gtm(self._gtm()) * 2.0).all()


class TestNodeRoundtripFidelity:
    """Latent error does not say whether a decoded centre stayed on its node.

    The decoder is nonlinear, so the only honest reading is the round trip:
    decode a node centre, re-encode what comes back, see where it lands.
    """

    _VECTORS = np.random.default_rng(3).standard_normal((60, 4))
    _CONFIG = {
        "n_basis_functions_sqrt": 2,
        "n_nodes_n_basis_diff": 2,
        "basis_width": 1.0,
        "reg_coeff": 0.1,
    }

    def _fitted(self):
        return gtm_operations.train_latent_gtm(
            self._VECTORS, config=self._CONFIG, descriptor_backend="chembed"
        )

    def _fitted_with_double_standardization(self):
        """What train_latent_gtm produced before it passed standardize=False.

        Built by hand because the function no longer has a way to produce it,
        which is the point of the test that uses this.
        """
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaled = scaler.fit_transform(self._VECTORS).astype(np.float64)
        gtm = gtm_operations.GTM(
            num_basis_functions=4,
            num_nodes=16,
            basis_width=1.0,
            reg_coeff=0.1,
            max_iter=200,
            device="cpu",
            standardize=True,
        )
        gtm.fit(torch.from_numpy(scaled).to(torch.float64))
        gtm_operations._tag_latent_gtm(gtm, scaler, "chembed")
        return gtm, scaler

    class _PerfectEngine:
        """Decodes each centre to its own name and encodes it straight back.

        A lossless engine isolates the manifold: whatever the round trip loses
        here is the 2D bottleneck, not the decoder.
        """

        # Alkanes of increasing length: distinct, and valid, which "C0" is not --
        # RDKit reads that as an unclosed ring bond.
        @staticmethod
        def _name(index):
            return "C" * (index + 1)

        def __init__(self, centers):
            self._by_smiles = {self._name(i): row for i, row in enumerate(centers)}

        def decode(self, z, *, max_len, temp, decode):
            return [self._name(i) for i in range(len(z))]

        def encode(self, smiles_list, *, batch_size):
            return np.array([self._by_smiles[s] for s in smiles_list])

    def test_a_lossless_engine_round_trips_perfectly(self):
        """A node centre is the point its own node is most responsible for, so
        with a lossless decoder every centre must come back to itself. Anything
        less is the manifold, not the engine — which is what makes this metric
        readable on a real one."""
        gtm, scaler, _ = self._fitted()
        centers = gtm_operations.latent_centers_from_gtm(gtm, scaler=scaler)

        result = gtm_operations.node_roundtrip_fidelity(
            gtm, self._PerfectEngine(centers), scaler=scaler
        )

        assert result["nodes"] == gtm.num_nodes
        assert result["decoded"] == gtm.num_nodes
        assert result["same_node"] == 1.0
        assert 0.0 <= result["origin_responsibility_median"] <= 1.0

    def test_double_standardization_would_break_the_round_trip(self):
        """Why train_latent_gtm passes standardize=False.

        chemographykit standardizes by default, re-fitting its own scaler on
        each batch, so node centres end up measured against a frame they were
        not written in and a lossless engine stops round-tripping. This pins
        the failure the flag prevents; it is not reachable through
        train_latent_gtm any more.
        """
        gtm, scaler = self._fitted_with_double_standardization()
        centers = gtm_operations.latent_centers_from_gtm(gtm, scaler=scaler)

        result = gtm_operations.node_roundtrip_fidelity(
            gtm, self._PerfectEngine(centers), scaler=scaler
        )

        assert result["decoded"] == gtm.num_nodes
        assert result["same_node"] < 1.0

    def test_unparseable_decodes_are_excluded_not_counted_as_failures(self):
        gtm, scaler, _ = self._fitted()
        centers = gtm_operations.latent_centers_from_gtm(gtm, scaler=scaler)

        class _HalfBroken(self._PerfectEngine):
            def decode(self, z, *, max_len, temp, decode):
                return [self._name(i) if i % 2 == 0 else "not a molecule(((" for i in range(len(z))]

        result = gtm_operations.node_roundtrip_fidelity(gtm, _HalfBroken(centers), scaler=scaler)

        assert result["decoded"] < result["nodes"]
        assert result["same_node"] == 1.0

    def test_nothing_decoding_is_reported_as_unknown_not_as_zero(self):
        """Zero would read as 'the map is useless'; it means 'no evidence'."""
        gtm, scaler, _ = self._fitted()

        class _Broken(self._PerfectEngine):
            def decode(self, z, *, max_len, temp, decode):
                return ["not a molecule((("] * len(z)

        result = gtm_operations.node_roundtrip_fidelity(gtm, _Broken({}), scaler=scaler)

        assert result["decoded"] == 0
        assert result["same_node"] is None
        assert result["origin_responsibility_median"] is None

    def test_responsibilities_do_not_depend_on_the_rest_of_the_batch(self):
        """A compound's node is a property of the compound.

        chemographykit's standardize re-fits on each call's input, so with it
        on, projecting five compounds alone gave different nodes than
        projecting the same five inside a larger set. That reaches node
        assignment and every activity landscape built on it, not only
        generation.
        """
        gtm, scaler, _ = self._fitted()

        alone = gtm_operations.project_latent_on_gtm(self._VECTORS[:5], gtm, scaler)
        in_batch = gtm_operations.project_latent_on_gtm(self._VECTORS, gtm, scaler)[:5]

        assert (alone.argmax(axis=1) == in_batch.argmax(axis=1)).all()
        np.testing.assert_allclose(alone, in_batch)

    def test_a_manifold_without_a_scaler_is_refused(self):
        """Projecting would need one; a silent skip would return nonsense."""
        gtm = SimpleNamespace(phi=np.ones((4, 2)), weights=np.ones((2, 3)))

        with pytest.raises(ValueError, match="scaler the manifold was fitted with"):
            gtm_operations.node_roundtrip_fidelity(gtm, self._PerfectEngine({}))


class TestTrainLatentGtm:
    """The latent path must actually reach chemographykit's API.

    GTM.fit takes only the data tensor; the iteration cap is a constructor
    argument. Passing n_iterations to fit raised TypeError on every call, so
    this path was unreachable.
    """

    def test_fixed_config_fits(self):
        vectors = np.random.default_rng(0).standard_normal((120, 8))
        config = {
            "n_basis_functions_sqrt": 3,
            "n_nodes_n_basis_diff": 3,
            "basis_width": 1.0,
            "reg_coeff": 100.0,
        }

        gtm, scaler, entropy = gtm_operations.train_latent_gtm(
            vectors, config=config, descriptor_backend="chembed"
        )

        assert gtm.num_nodes == 36
        assert gtm.num_basis_functions == 9
        assert 0.0 <= entropy <= 1.0
        assert scaler is not None

    def test_the_fitted_manifold_carries_its_backend_and_scaler(self):
        """Untagged, the guard warns and passes; scaler-less, centres stay z-scored."""
        vectors = np.random.default_rng(0).standard_normal((60, 4))
        config = {
            "n_basis_functions_sqrt": 2,
            "n_nodes_n_basis_diff": 2,
            "basis_width": 1.0,
            "reg_coeff": 100.0,
        }

        gtm, scaler, _ = gtm_operations.train_latent_gtm(
            vectors, config=config, descriptor_backend="chembed"
        )

        assert gtm.descriptor_backend == "chembed"
        assert gtm.latent_scaler is scaler
        # The guard can now do its job on a manifold this function produced.
        with pytest.raises(ValueError, match="meaningless coordinates"):
            gtm_operations.validate_gtm_descriptor_backend(gtm, "morgan")

    @pytest.mark.parametrize(
        ("supplied", "recorded"),
        [
            ("chembed_embedding", "chembed"),  # an alias resolves to its family
            ("chembed", "chembed"),
            ("peptide_latent", "peptide_latent"),  # not a descriptor, kept verbatim
        ],
    )
    def test_backend_names_are_canonicalised_but_unknown_ones_survive(self, supplied, recorded):
        """Peptide latents have no descriptor alias and must keep their own name,
        so they still fail the check against any small-molecule projection."""
        gtm = SimpleNamespace()

        gtm_operations._tag_latent_gtm(gtm, scaler=None, descriptor_backend=supplied)

        assert gtm.descriptor_backend == recorded

    def test_projection_and_centers_round_trip(self):
        vectors = np.random.default_rng(1).standard_normal((80, 5))
        config = {
            "n_basis_functions_sqrt": 2,
            "n_nodes_n_basis_diff": 2,
            "basis_width": 1.0,
            "reg_coeff": 100.0,
        }
        gtm, scaler, _ = gtm_operations.train_latent_gtm(
            vectors, config=config, descriptor_backend="chembed"
        )

        resps = gtm_operations.project_latent_on_gtm(vectors, gtm, scaler)
        centers = gtm_operations.latent_centers_from_gtm(gtm, scaler=scaler)

        assert resps.shape == (80, gtm.num_nodes)
        # Centres come back in the original latent space, not the scaled one.
        assert centers.shape == (gtm.num_nodes, 5)

    _SMALL = {
        "n_basis_functions_sqrt": 2,
        "n_nodes_n_basis_diff": 2,
        "basis_width": 1.0,
        "reg_coeff": 100.0,
    }

    def test_neighbourhood_preservation_is_scored_in_the_raw_latent_space(self):
        """Peptide latents are no descriptor alias, yet still get a metric.

        Neighbours come from the vectors as given, not the z-scored copy the
        manifold is fitted on: one stretched axis makes the two disagree.
        """
        vectors = np.random.default_rng(0).standard_normal((60, 4)) * [1.0, 1.0, 1.0, 20.0]

        gtm, scaler, score = gtm_operations.train_latent_gtm(
            vectors,
            config=self._SMALL,
            descriptor_backend="peptide_latent",
            objective="neighborhood_preservation",
            neighborhood_k=5,
        )

        coords = (
            gtm_operations.project_latent_on_gtm(vectors, gtm, scaler) @ gtm.nodes.cpu().numpy()
        )
        raw = gtm_operations._knn_indices(vectors, 5, "euclidean")
        assert score == pytest.approx(gtm_operations.neighborhood_preservation(raw, coords))
        with pytest.raises(ValueError, match="Unknown objective"):
            gtm_operations.train_latent_gtm(
                vectors, config=self._SMALL, descriptor_backend="peptide_latent", objective="x"
            )

    def test_the_tool_passes_the_objective_through_and_reports_it(self, monkeypatch, tmp_path):
        from cs_copilot.tools.chemography.gtm import GTMToolkit

        csv = tmp_path / "latents.csv"
        latents = pd.DataFrame(np.random.default_rng(0).standard_normal((40, 4)))
        latents.add_prefix("z_").to_csv(csv, index=False)
        monkeypatch.setattr(gtm_operations.S3, "open", lambda path, mode="r": open(path, mode))
        real, seen = gtm_operations.train_latent_gtm, {}

        def small_fit(X, strategy, **kwargs):
            seen.update(kwargs)
            return real(X, config=self._SMALL, **kwargs)

        monkeypatch.setattr(gtm_operations, "train_latent_gtm", small_fit)

        message = GTMToolkit().train_gtm_on_latent_space(
            str(csv),
            "latents",
            "map",
            SimpleNamespace(session_state={}),
            objective="neighborhood_preservation",
            neighborhood_k=5,
        )

        assert seen["objective"] == "neighborhood_preservation"
        assert seen["neighborhood_metric"] == "euclidean"  # "auto", resolved for peptides
        assert "Neighborhood preservation (k=5, euclidean)" in message


class TestDefaultManifold:
    """The published default map is the chembed manifold.

    It is the only one left: the LSTM manifold went with its engine. Morgan
    sessions are unaffected either way, since ``validate_gtm_descriptor_backend``
    refuses fingerprints against a latent manifold.
    """

    def _agent(self, descriptor):
        return SimpleNamespace(
            session_state={gtm_operations.SESSION_DESCRIPTOR_TYPE_KEY: descriptor}
        )

    def test_chembed_default_resolves_into_the_chembed_cache(self, tmp_path, monkeypatch):
        chembed_cache = tmp_path / "gtm-chembed"
        chembed_cache.mkdir()
        chembed_map = chembed_cache / "chembed.pkl.gz"
        chembed_map.write_text("placeholder")
        monkeypatch.setattr(gtm_operations, "DEFAULT_GTM_CHEMBED_MODEL_PATH", str(chembed_cache))
        monkeypatch.setattr(gtm_operations, "load_gtm_model", lambda _path: object())

        resolved = gtm_operations.resolve_gtm_model_path(
            agent=self._agent("chembed"), use_default=True
        )

        assert resolved == str(chembed_map)
