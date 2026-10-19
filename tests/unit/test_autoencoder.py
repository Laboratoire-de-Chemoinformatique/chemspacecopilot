#!/usr/bin/env python
# coding: utf-8
"""Tests for the autoencoder toolkit."""

from types import SimpleNamespace

import pytest
import torch

from cs_copilot.tools.chemistry.autoencoder_toolkit import (
    DEFAULT_SAMPLE_COUNT_CPU,
    DEFAULT_SAMPLE_COUNT_GPU,
    AutoencoderToolkit,
)
from cs_copilot.tools.io.session_memory import load_candidate_set_artifact


class StubEngine:
    """Minimal :class:`AutoencoderEngine` so tests need no checkpoint."""

    name = "chembed"

    def load(self, *, model_path, device):
        return SimpleNamespace(eval=lambda: None), {}, SimpleNamespace(d_z=4)

    def encode(self, smiles_list, *, batch_size):
        raise NotImplementedError

    def decode(self, z, *, max_len, temp, decode):
        return ["CCO"] * len(z)

    def latent_dim(self):
        return 4

    def info(self):
        return {"backend": self.name, "latent_dimension": 4}

    def decode_settings(self, *, temp, decode, max_len):
        return {
            "temperature": temp,
            "decode_mode": decode,
            "max_length": max_len,
            "backend": self.name,
        }

    def default_noise_scale(self):
        return 0.5


def stub_toolkit(device: str = "cpu") -> AutoencoderToolkit:
    """Build a toolkit on :class:`StubEngine`, loading nothing."""
    return AutoencoderToolkit(model_path=None, device=device, engine=StubEngine())


def test_sample_molecules_registers_autoencoder_candidate_set(monkeypatch, tmp_path):
    """Direct Autoencoder sampling records generated provenance and artifact-backed set."""
    monkeypatch.chdir(tmp_path)
    toolkit = stub_toolkit()
    monkeypatch.setattr(toolkit, "sample_from_latent", lambda **_kwargs: ["CCO", "CCN"])
    agent = SimpleNamespace(session_state={})
    session_state = {}

    summary = toolkit.sample_molecules(agent=agent, session_state=session_state)

    memory = session_state["session_objects"]
    assert summary["registered_candidate_set_id"] == "cset_001"
    assert summary["artifact_path"].endswith(
        "02_analog_generation/candidate_sets/cset_001/candidates.json"
    )
    assert session_state["sampled_molecules"]["candidate_set_id"] == "cset_001"
    assert session_state["sampled_molecules"]["preview"] == [
        {"smiles": "CCO"},
        {"smiles": "CCN"},
    ]
    assert memory["candidate_sets"]["cset_001"]["origin_agent"] == "autoencoder_toolkit"
    assert memory["candidate_sets"]["cset_001"]["generation_engine"] == "autoencoder"
    assert memory["candidate_sets"]["cset_001"]["artifact_path"] == summary["artifact_path"]
    assert memory["compounds"]["cmp_001"]["origin_agent"] == "autoencoder_toolkit"
    assert memory["compounds"]["cmp_001"]["candidate_set_id"] == "cset_001"
    artifact = load_candidate_set_artifact(session_state, "sampled_molecules")
    assert artifact["candidates"] == ["CCO", "CCN"]


def test_generation_audit_names_the_engine_that_ran(monkeypatch):
    """Every audit names its backend, so no artifact is engine-ambiguous."""
    toolkit = stub_toolkit()
    monkeypatch.setattr(toolkit, "validate_model_loaded", lambda: True)

    toolkit.sample_from_latent(n_samples=2, temp=0.7, decode="sample")

    settings = toolkit.last_generation_audit["settings"]
    assert settings["backend"] == "chembed"
    assert settings["temperature"] == 0.7
    assert settings["decode_mode"] == "sample"


class TestSampleCountDefault:
    """The prior-sampling default depends on the device.

    chembed decodes autoregressively, so 5000 molecules take roughly 35 s on a
    GPU and about 17 minutes on CPU. A CPU host therefore gets a smaller default
    rather than a call that looks like a hang.
    """

    def _toolkit(self, device: str) -> AutoencoderToolkit:
        toolkit = AutoencoderToolkit.__new__(AutoencoderToolkit)
        toolkit.device = torch.device(device)
        toolkit.engine = StubEngine()
        return toolkit

    def test_gpu_default(self):
        assert self._toolkit("cuda")._resolve_sample_count(None) == DEFAULT_SAMPLE_COUNT_GPU

    def test_cpu_default_is_smaller(self):
        assert self._toolkit("cpu")._resolve_sample_count(None) == DEFAULT_SAMPLE_COUNT_CPU
        assert DEFAULT_SAMPLE_COUNT_CPU < DEFAULT_SAMPLE_COUNT_GPU

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_explicit_count_wins_on_every_device(self, device):
        """An explicit count must not depend on the host."""
        assert self._toolkit(device)._resolve_sample_count(42) == 42

    def test_summary_records_where_the_count_came_from(self, tmp_path, monkeypatch):
        """Two runs of the same call can differ, so the record must say why."""
        toolkit = stub_toolkit()
        monkeypatch.setattr(toolkit, "validate_model_loaded", lambda: True)
        monkeypatch.setattr(toolkit, "sample_from_latent", lambda **kwargs: ["CCO"])

        state: dict = {}
        summary = toolkit.sample_molecules(
            session_state=state, audit_path=str(tmp_path / "audit.json")
        )
        audit = summary["generation_audit"]

        assert audit["n_samples_requested"] is None
        assert audit["n_samples_resolved"] == DEFAULT_SAMPLE_COUNT_CPU
        assert audit["n_samples_source"] == "default_cpu"

        summary = toolkit.sample_molecules(
            n_samples=7, session_state=state, audit_path=str(tmp_path / "audit2.json")
        )

        assert summary["generation_audit"]["n_samples_source"] == "caller"
        assert summary["generation_audit"]["n_samples_resolved"] == 7
