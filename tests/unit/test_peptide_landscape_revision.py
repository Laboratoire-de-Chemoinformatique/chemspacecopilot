"""Validate branch-derived node sampling and prospective raw-output provenance."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.numpy import save_file

from cs_copilot.tools.chemistry import peptide_designer_toolkit as toolkit_module
from cs_copilot.tools.chemistry.peptide_landscape_store import (
    PeptideLandscapeError,
    latent_centers_for_nodes,
    load_local_peptide_landscape,
    sample_latents_from_nodes,
    select_active_landscape_nodes,
)


def bundle_files(root):
    root.mkdir(exist_ok=True)
    (root / "landscape.json").write_text(
        json.dumps(
            {
                "landscape_id": "unit_test_landscape",
                "latent_dim": 2,
                "peptide_alphabet": list("ACDEFGHIKLMNPQRSTVWY"),
            }
        )
    )
    (root / "sampler.json").write_text(json.dumps({"activity_threshold": 0.5}))
    pd.DataFrame(
        {
            "node_id": [1, 2],
            "x": [1, 1],
            "y": [1, 2],
            "organism": ["Escherichia coli"] * 2,
            "density": [10.0, 10.0],
            "activity_mean": [0.8, 0.7],
            "activity_class": ["active_enriched"] * 2,
            "uncertainty": [0.1, 0.1],
            "n_observations": [5.0, 5.0],
        }
    ).to_parquet(root / "nodes.parquet")
    save_file(
        {
            "gtm.phi": np.eye(2),
            "gtm.weights": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "scaler.mean": np.array([10.0, 20.0]),
            "scaler.scale": np.array([2.0, 3.0]),
        },
        str(root / "landscape.safetensors"),
    )
    return root


def test_one_based_centers_and_inverse_scaler_are_exact(tmp_path):
    bundle = load_local_peptide_landscape(str(bundle_files(tmp_path)))
    np.testing.assert_equal(latent_centers_for_nodes(bundle, [2]), [[3.0, 4.0]])
    selected = bundle.nodes.iloc[[1]]
    latent, assignments = sample_latents_from_nodes(
        bundle,
        selected,
        n_samples=2,
        local_noise_scale=0,
        random_state=42,
    )
    np.testing.assert_equal(latent, [[16.0, 32.0], [16.0, 32.0]])
    assert assignments["node_id"].tolist() == [2, 2]
    assert assignments["node_selection_probability"].tolist() == [1.0, 1.0]
    with pytest.raises(PeptideLandscapeError, match="outside"):
        latent_centers_for_nodes(bundle, [0])


def test_activity_filters_and_sampling_seed_are_effective(tmp_path):
    bundle = load_local_peptide_landscape(str(bundle_files(tmp_path)))
    selected = select_active_landscape_nodes(bundle, "E. coli", top_n=2, min_activity=0.75)
    assert selected["node_id"].tolist() == [1]
    kwargs = {"n_samples": 5, "local_noise_scale": 0.2, "random_state": 11}
    first, _ = sample_latents_from_nodes(bundle, selected, **kwargs)
    second, _ = sample_latents_from_nodes(bundle, selected, **kwargs)
    np.testing.assert_equal(first, second)
    with pytest.raises(PeptideLandscapeError, match="No active"):
        select_active_landscape_nodes(bundle, "Ecoli", min_activity=0.99)


def test_local_loader_rejects_incompatible_scaler(tmp_path):
    root = bundle_files(tmp_path)
    save_file(
        {
            "gtm.phi": np.eye(2),
            "gtm.weights": np.eye(2),
            "scaler.mean": np.zeros(2),
            "scaler.scale": np.array([0.0, 1.0]),
        },
        str(root / "landscape.safetensors"),
    )
    with pytest.raises(PeptideLandscapeError, match="Invalid GTM/scaler"):
        load_local_peptide_landscape(str(root))


def test_tool_keeps_raw_denominator_duplicate_invalid_and_node_provenance(monkeypatch, tmp_path):
    cls = toolkit_module.PeptideDesignerToolkit
    monkeypatch.setattr(cls, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(cls, "_load_model", lambda self: None)
    toolkit = cls(model_path=str(tmp_path), device="cpu")
    assert "sample_peptides_from_landscape" in toolkit.functions
    monkeypatch.setattr(toolkit, "get_latent_dimension", lambda: 2)

    def decode(*args, **kwargs):
        torch.rand(1)
        return ["A C D", "ACD", "B", "A E"]

    monkeypatch.setattr(toolkit, "decode_latent", decode)
    captured = {}

    def save(state, **kwargs):
        captured.update(kwargs)
        return {
            "artifact_path": str(tmp_path / "candidates.json"),
            "artifact_rel_path": str(tmp_path / "candidates.json"),
            "peptide_candidate_set_id": "pep_cset_001",
        }

    monkeypatch.setattr(toolkit_module, "_save_peptide_design_artifact", save)
    monkeypatch.setattr(toolkit_module.S3, "open", lambda p, mode: Path(p).open(mode))
    monkeypatch.setattr(toolkit_module.S3, "path", lambda p: str(p))
    state = {"peptide_landscape_bundle": {"bundle_path": str(bundle_files(tmp_path / "bundle"))}}
    rng_state = torch.get_rng_state().clone()
    output = toolkit.sample_peptides_from_landscape(
        n_candidates=2,
        oversample_factor=2,
        random_state=17,
        session_state=state,
    )
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert output["count_attempted"] == 4 and output["count_returned"] == 2
    assert output["validity_fraction"] == 0.75
    assert output["exact_sequence_uniqueness_fraction"] == pytest.approx(2 / 3)
    metadata = captured["metadata"]
    assert len(metadata["raw_outputs"]) == 4
    assert metadata["torch_seed"] == metadata["random_state"] == 17
    assert len(metadata["selected_node_probabilities"]) == 2
    assert sum(
        r["node_selection_probability"] for r in metadata["selected_node_probabilities"]
    ) == pytest.approx(1)
    assert all("node_assignment" in r for r in metadata["raw_outputs"])
    assert "active_prob" in pd.read_csv(output["activity_landscape_path"]).columns
    assert "landscape_sampled_peptides" in state
