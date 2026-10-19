#!/usr/bin/env python
# coding: utf-8
"""Tests for the chembed latent engine.

Every test here is offline. chembed's modules are imported lazily inside the
engine, so they are replaced in ``sys.modules`` with fakes and the real 227 MB
checkpoint is never downloaded.
"""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderError, AutoencoderToolkit
from cs_copilot.tools.chemistry.chembed_engine import ChembedEngine

D_MODEL = 8


def _fake_vae(d_bottleneck: int = 1, d_model: int = D_MODEL):
    """Return a stand-in for chembed's SELFIESTransformerVAE."""
    return SimpleNamespace(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        max_len=256,
        vocab=["<START>", "<STOP>", "[C]"],
        model_config={
            "nb_layers_encoder": 6,
            "nb_layers_decoder": 6,
            "dim_feedforward_encoder": 2048,
            "dim_feedforward_decoder": 2048,
            "nhead": 8,
        },
        eval=lambda: None,
    )


@pytest.fixture
def fake_chembed(monkeypatch):
    """Install fake chembed modules and return the recorder for their calls."""
    calls = {}
    vae = _fake_vae()

    checkpoint_utils = ModuleType("chembed.checkpoint_utils")
    checkpoint_utils.load_vae_from_hub = lambda **kw: (calls.update(hub=kw), vae)[1]
    checkpoint_utils.load_vae_from_checkpoint = lambda path, **kw: (
        calls.update(local={"path": path, **kw}),
        vae,
    )[1]

    encode_mod = ModuleType("chembed.encode")

    def _encode_multiple_smiles(smiles_list, model, **kw):
        calls["encode"] = {"smiles": list(smiles_list), **kw}
        return torch.zeros(len(smiles_list), model.d_bottleneck, model.d_model)

    encode_mod.encode_multiple_smiles = _encode_multiple_smiles

    decode_mod = ModuleType("chembed.decode")

    def _decode_zs_to_smiles(zs, model, **kw):
        calls["decode"] = {"shape": tuple(zs.shape), **kw}
        return ["CCO"] * zs.shape[0]

    decode_mod.decode_zs_to_smiles = _decode_zs_to_smiles

    package = ModuleType("chembed")
    for name, module in (
        ("chembed", package),
        ("chembed.checkpoint_utils", checkpoint_utils),
        ("chembed.encode", encode_mod),
        ("chembed.decode", decode_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    package.checkpoint_utils = checkpoint_utils

    calls["vae"] = vae
    return calls


@pytest.fixture
def loaded_engine(fake_chembed):
    """Return an engine with the fake checkpoint already loaded."""
    engine = ChembedEngine()
    engine.load(model_path=None, device="cpu")
    return engine


class TestLoad:
    def test_loads_from_hub_by_default(self, fake_chembed):
        engine = ChembedEngine()
        model, vocab, config = engine.load(model_path=None, device="cpu")

        assert fake_chembed["hub"]["repo_id"] == "3BioCompBio/chembed-default"
        assert fake_chembed["hub"]["filename"] == "model.ckpt"
        assert model is fake_chembed["vae"]
        assert vocab is fake_chembed["vae"].vocab
        assert config is fake_chembed["vae"].model_config
        assert engine.checkpoint_path == "3BioCompBio/chembed-default/model.ckpt"

    def test_toolkit_default_model_path_is_ignored(self, fake_chembed):
        """The toolkit's default path names the deepchemography directory."""
        engine = ChembedEngine()
        engine.load(model_path="/does/not/exist/autoencoder", device="cpu")

        assert "hub" in fake_chembed
        assert "local" not in fake_chembed

    def test_existing_file_loads_as_local_checkpoint(self, fake_chembed, tmp_path):
        checkpoint = tmp_path / "model.ckpt"
        checkpoint.write_bytes(b"")
        engine = ChembedEngine()
        engine.load(model_path=str(checkpoint), device="cpu")

        assert fake_chembed["local"]["path"] == checkpoint
        assert engine.checkpoint_path == str(checkpoint)

    def test_wider_bottleneck_is_rejected(self, fake_chembed, monkeypatch):
        """A bottleneck > 1 would not round-trip through the (n, 1, d) reshape."""
        wide = SimpleNamespace(
            load_vae_from_hub=lambda **kw: _fake_vae(d_bottleneck=2),
            load_vae_from_checkpoint=lambda path, **kw: _fake_vae(d_bottleneck=2),
        )
        monkeypatch.setitem(sys.modules, "chembed.checkpoint_utils", wide)
        monkeypatch.setattr(sys.modules["chembed"], "checkpoint_utils", wide)
        with pytest.raises(AutoencoderError, match="d_bottleneck=2"):
            ChembedEngine().load(model_path=None, device="cpu")

    def test_missing_package_raises_autoencoder_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chembed", None)
        monkeypatch.setitem(sys.modules, "chembed.checkpoint_utils", None)
        with pytest.raises(AutoencoderError, match="chembed"):
            ChembedEngine().load(model_path=None, device="cpu")


class TestEncode:
    def test_flattens_bottleneck_axis(self, loaded_engine):
        result = loaded_engine.encode(["CCO", "c1ccccc1"], batch_size=32)

        assert result.shape == (2, D_MODEL)
        assert result.dtype == np.float32

    def test_replaces_unknown_tokens_rather_than_failing_the_batch(
        self, loaded_engine, fake_chembed
    ):
        loaded_engine.encode(["CCO"], batch_size=32)

        assert fake_chembed["encode"]["replace_if_not_in_vocab"] is True
        assert fake_chembed["encode"]["use_sigma"] is False

    def test_unparseable_smiles_keeps_the_row(self, loaded_engine, fake_chembed):
        """GTM pairs rows with a dataframe, so the row count must not change."""
        result = loaded_engine.encode(["CCO", "not-a-smiles", "c1ccccc1"], batch_size=32)

        assert result.shape[0] == 3
        assert fake_chembed["encode"]["smiles"] == ["CCO", "C", "c1ccccc1"]

    def test_smiles_are_not_pre_standardized(self, loaded_engine, fake_chembed):
        """chembed standardizes internally; doing it twice risks vocab misses."""
        loaded_engine.encode(["CC(=O)OC1=CC=CC=C1C(=O)O"], batch_size=32)

        assert fake_chembed["encode"]["smiles"] == ["CC(=O)OC1=CC=CC=C1C(=O)O"]


class TestDecode:
    def test_reshapes_flat_latents_for_chembed(self, loaded_engine, fake_chembed):
        loaded_engine.decode(
            np.zeros((4, D_MODEL), dtype=np.float32), max_len=100, temp=1.0, decode="greedy"
        )

        assert fake_chembed["decode"]["shape"] == (4, 1, D_MODEL)

    def test_accepts_torch_input(self, loaded_engine, fake_chembed):
        out = loaded_engine.decode(torch.zeros(3, D_MODEL), max_len=100, temp=1.0, decode="greedy")

        assert fake_chembed["decode"]["shape"] == (3, 1, D_MODEL)
        assert out == ["CCO"] * 3

    def test_ignored_decode_parameters_do_not_raise(self, loaded_engine):
        out = loaded_engine.decode(
            np.zeros((2, D_MODEL), dtype=np.float32), max_len=40, temp=0.7, decode="sample"
        )

        assert out == ["CCO", "CCO"]


class TestMetadata:
    def test_latent_dim_is_the_flattened_width(self, loaded_engine):
        assert loaded_engine.latent_dim() == D_MODEL

    def test_info_declares_what_chembed_cannot_do(self, loaded_engine):
        info = loaded_engine.info()

        assert info["backend"] == "chembed"
        assert info["supports_temperature"] is False
        assert info["supports_stochastic_decode"] is False
        assert info["latent_dimension"] == D_MODEL

    def test_info_keeps_the_deepchemography_keys(self, loaded_engine):
        """Callers formatting model info should not have to special-case engines."""
        shared = {
            "vocabulary_size",
            "latent_dimension",
            "encoder_type",
            "encoder_layers",
            "encoder_hidden_size",
            "decoder_type",
            "decoder_layers",
            "decoder_hidden_size",
            "batch_normalization",
            "bidirectional_encoder",
        }

        assert shared <= set(loaded_engine.info())

    def test_decode_settings_record_requested_and_applied(self, loaded_engine):
        settings = loaded_engine.decode_settings(temp=0.7, decode="sample", max_len=40)

        assert settings["temperature"] is None
        assert settings["decode_mode"] == "greedy"
        assert settings["max_length"] == 256
        assert settings["requested_temperature"] == 0.7
        assert settings["requested_decode_mode"] == "sample"
        assert settings["requested_max_length"] == 40
        assert "temperature" in settings["ignored_parameters"]


class TestToolkitIntegration:
    """The engine must not disturb the Agno tool surface.

    ``execution_contracts`` names every tool after ``type(owner).__name__``, so
    the toolkit class must stay ``AutoencoderToolkit`` whichever engine is used.
    """

    def _toolkit(self, engine=None):
        toolkit = AutoencoderToolkit.__new__(AutoencoderToolkit)
        toolkit.engine = engine
        toolkit.device = torch.device("cpu")
        toolkit.model_path = "test-only-no-download"
        return toolkit

    def test_registered_tool_names_are_engine_independent(self, fake_chembed, monkeypatch):
        with_engine = AutoencoderToolkit(model_path="unused", device="cpu", engine=ChembedEngine())
        defaulted = AutoencoderToolkit(model_path="unused", device="cpu")

        assert type(with_engine).__name__ == type(defaulted).__name__ == "AutoencoderToolkit"
        assert set(with_engine.functions) == set(defaulted.functions)
        assert {
            "encode_smiles",
            "decode_latent",
            "sample_molecules",
            "interpolate_molecules",
            "reconstruct_smiles",
            "get_latent_dimension",
            "validate_model_loaded",
            "explore_latent_neighborhood",
            "get_model_info",
        } <= set(with_engine.functions)

    def test_toolkit_records_the_loaded_checkpoint(self, fake_chembed):
        toolkit = AutoencoderToolkit(model_path="unused", device="cpu", engine=ChembedEngine())

        assert toolkit.model_path == "3BioCompBio/chembed-default/model.ckpt"
        assert toolkit.validate_model_loaded() is True
        assert toolkit.get_latent_dimension() == D_MODEL
        assert toolkit.get_model_info()["backend"] == "chembed"

    def test_generation_audit_reports_the_applied_settings(self, fake_chembed):
        toolkit = AutoencoderToolkit(model_path="unused", device="cpu", engine=ChembedEngine())

        toolkit.sample_from_latent(n_samples=2, temp=0.7, decode="sample", audit_path=None)
        settings = toolkit.last_generation_audit["settings"]

        assert settings["backend"] == "chembed"
        assert settings["temperature"] is None
        assert settings["requested_temperature"] == 0.7
        assert settings["decode_mode"] == "greedy"
        assert toolkit.last_generation_audit["raw_outputs_available"] is True


class TestDecodeDeviceHandling:
    """Prior sampling passes a tensor on the model's device, not a numpy array."""

    def test_device_tensor_is_not_routed_through_numpy(self, loaded_engine, fake_chembed):
        """np.asarray raises on a CUDA tensor, so tensors must bypass it."""

        class _TensorRefusingNumpy(torch.Tensor):
            @staticmethod
            def __new__(cls, data):
                return torch.Tensor._make_subclass(cls, data)

            def __array__(self, *args, **kwargs):
                raise TypeError("can't convert cuda:0 device type tensor to numpy")

        z = _TensorRefusingNumpy(torch.zeros(5, D_MODEL))
        loaded_engine.decode(z, max_len=100, temp=1.0, decode="greedy")

        assert fake_chembed["decode"]["shape"] == (5, 1, D_MODEL)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_cuda_tensor_decodes(self, loaded_engine, fake_chembed):
        loaded_engine.decode(
            torch.zeros(3, D_MODEL, device="cuda"), max_len=100, temp=1.0, decode="greedy"
        )

        assert fake_chembed["decode"]["shape"] == (3, 1, D_MODEL)


class TestNoiseScale:
    """Noise tuned for one latent space does not transfer to another.

    Measured over aspirin, lidocaine and a large peptidomimetic at 60 samples
    each: at 0.1 chembed returns the parent essentially every time.
    """

    def test_engine_default_leaves_the_parent(self, loaded_engine):
        """0.1, tuned for the retired LSTM, returned the parent nearly every time."""
        assert loaded_engine.default_noise_scale() > 0.1

    def test_toolkit_uses_the_engine_default_when_none_given(self, fake_chembed, monkeypatch):
        from cs_copilot.tools.chemistry.chembed_engine import DEFAULT_NOISE_SCALE

        toolkit = AutoencoderToolkit(model_path="unused", device="cpu", engine=ChembedEngine())
        assert toolkit._default_noise_scale() == DEFAULT_NOISE_SCALE

    def test_explicit_noise_scale_is_respected(self, fake_chembed, monkeypatch):
        toolkit = AutoencoderToolkit(model_path="unused", device="cpu", engine=ChembedEngine())
        captured = {}

        def _capture(z, **kwargs):
            captured["z"] = z
            return []

        monkeypatch.setattr(toolkit, "decode_latent", _capture)
        monkeypatch.setattr(toolkit, "encode_smiles_array", lambda s: np.zeros((1, D_MODEL), "f"))
        toolkit.explore_latent_neighborhood("CCO", noise_scale=0.0, n_neighbors=4)

        # Zero noise must leave every neighbour exactly on the parent latent.
        assert captured["z"].shape == (4, D_MODEL)
        assert np.allclose(captured["z"], 0.0)


class TestDescriptorEnginePinning:
    """Descriptor families pin their engine; only generation follows the env.

    The retired LSTM's families are refused rather than aliased to chembed:
    _compute_descriptors reuses an existing column as a cache without recording
    which model wrote it, so an alias would mix LSTM and chembed vectors in one
    autoencoder_embedding column.
    """

    @pytest.mark.parametrize("retired", ["autoencoder", "ae", "latent", "embedding"])
    def test_retired_descriptor_families_are_refused(self, retired):
        from cs_copilot.tools.chemistry.descriptors import MolecularDescriptorEncoder

        with pytest.raises(ValueError, match="retired"):
            MolecularDescriptorEncoder(retired)

    def test_chembed_keeps_its_own_column(self):
        from cs_copilot.tools.chemistry.descriptors import MolecularDescriptorEncoder

        assert MolecularDescriptorEncoder("chembed").column_name() == "chembed_embedding"
        assert MolecularDescriptorEncoder("morgan").column_name() == "morgan_fingerprint"

    def test_generation_default_is_chembed(self, monkeypatch):
        from cs_copilot.tools.chemistry.autoencoder_toolkit import resolve_autoencoder_backend

        monkeypatch.delenv("AUTOENCODER_BACKEND", raising=False)

        assert resolve_autoencoder_backend(None) == "chembed"

    def test_a_retired_engine_in_the_env_fails_loudly(self, monkeypatch):
        """Silently generating on chembed would attribute results to the wrong model."""
        from cs_copilot.tools.chemistry.autoencoder_toolkit import (
            AutoencoderError,
            resolve_autoencoder_backend,
        )

        monkeypatch.setenv("AUTOENCODER_BACKEND", "deepchemography")

        with pytest.raises(AutoencoderError, match="retired"):
            resolve_autoencoder_backend(None)


class TestEngineNamingAndPeptideDependency:
    """chembed replaces the small-molecule LSTM, not the deepchemography package.

    The package provides two unrelated models: LSTMAutoencoder for small
    molecules, which chembed supersedes, and the peptide WAE, which has no
    chembed counterpart. A cleanup that reads "chembed is the autoencoder now,
    drop deepchemography" would delete peptide generation.
    """

    def test_retiring_the_engine_names_the_tag_that_still_has_it(self):
        from cs_copilot.tools.chemistry.autoencoder_toolkit import (
            AutoencoderError,
            resolve_autoencoder_backend,
        )

        with pytest.raises(AutoencoderError, match="v0.4.0"):
            resolve_autoencoder_backend("deepchemography_lstm")

    def test_chembed_is_still_accepted_by_name(self):
        from cs_copilot.tools.chemistry.autoencoder_toolkit import resolve_autoencoder_backend

        assert resolve_autoencoder_backend("chembed") == "chembed"

    def test_peptide_designer_imports_the_package_independently(self):
        """Peptide generation must not route through the small-molecule engine."""
        source = (
            Path(__file__).parents[2] / "src/cs_copilot/tools/chemistry/peptide_designer_toolkit.py"
        ).read_text()

        # Its own import of the package, not a call into the molecule engine.
        assert "from deepchemography.peptides import" in source
        assert "resolve_autoencoder_backend" not in source

    def test_the_dependency_is_still_declared(self):
        """Retiring the LSTM engine must not remove the package peptides need."""
        pyproject = (Path(__file__).parents[2] / "pyproject.toml").read_text()

        assert "deepchemography" in pyproject
