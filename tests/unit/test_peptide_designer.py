#!/usr/bin/env python
# coding: utf-8
"""Tests for the Peptide Designer public facade."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

import cs_copilot.tools as tools
import cs_copilot.tools.chemistry as chemistry
import cs_copilot.tools.chemistry.peptide_designer_toolkit as peptide_designer_module
from cs_copilot.agents.registry import list_available_agent_types
from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.peptide_designer_toolkit import (
    LLMPeptideDesignEngine,
    PeptideDesignerError,
    PeptideDesignerToolkit,
)


def test_peptide_designer_public_registry_name_replaces_peptide_wae():
    """The public agent type should be Peptide Designer, not Peptide WAE."""
    agent_types = list_available_agent_types()

    assert "peptide_designer" in agent_types
    assert "peptide_wae" not in agent_types


def test_peptide_designer_toolkit_public_name(monkeypatch, tmp_path):
    """The public toolkit name should be peptide_designer."""
    monkeypatch.setattr(PeptideDesignerToolkit, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(PeptideDesignerToolkit, "_load_model", lambda self: None)

    toolkit = PeptideDesignerToolkit(model_path=str(tmp_path), device="cpu")

    assert toolkit.name == "peptide_designer"
    assert "sample_peptides" in toolkit.functions
    assert "design_peptides" in toolkit.functions


def test_peptide_designer_public_exports_replace_peptide_wae():
    """Only the Peptide Designer class should be exported from public tool packages."""
    assert tools.PeptideDesignerToolkit is PeptideDesignerToolkit
    assert chemistry.PeptideDesignerToolkit is PeptideDesignerToolkit
    assert issubclass(PeptideDesignerError, Exception)
    assert not hasattr(tools, "PeptideWAEToolkit")
    assert not hasattr(chemistry, "PeptideWAEToolkit")


def _toolkit_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(PeptideDesignerToolkit, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(PeptideDesignerToolkit, "_load_model", lambda self: None)
    return PeptideDesignerToolkit(model_path=str(tmp_path), device="cpu")


def _write_synthetic_peptide_landscape(tmp_path):
    from safetensors.numpy import save_file

    root = tmp_path / "landscapes" / "dbaasp_amp_v1"
    (root / "runtime").mkdir(parents=True)
    (root / "runtime" / "gtm.pkl.gz").write_bytes(b"not-used-in-tests")

    manifest = {
        "schema_version": "1.0.0",
        "required_loader_version": "0.1.0",
        "landscape_id": "dbaasp_amp_v1",
        "dataset_repo": "axelrolov/peptide_designer_data",
        "compatible_decoder_repo": "axelrolov/wae_peptides",
        "latent_dim": 3,
        "condition_dim": 2,
        "max_sequence_length": 25,
        "peptide_alphabet": ["A", "C", "D", "E", "F"],
        "raw_source_data_redistributed": False,
        "activity_endpoint": {
            "organisms": ["Escherichia coli", "Bacillus subtilis"],
            "plotted_organisms": ["Escherichia coli"],
            "type": "binary antimicrobial activity",
            "units": "active/inactive class",
        },
    }
    (root / "landscape.json").write_text(json.dumps(manifest))
    (root / "sampler.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "target_landscape": "dbaasp_amp_v1",
                "activity_threshold": 0.5,
                "objective_weights": {
                    "activity": 1.0,
                    "density_penalty": 0.1,
                    "uncertainty_penalty": 0.1,
                },
            }
        )
    )
    nodes = pd.DataFrame(
        [
            {
                "x": 1,
                "y": 1,
                "node_id": 1,
                "organism": "Escherichia coli",
                "density": 2.0,
                "activity_mean": 0.95,
                "activity_class": "active_enriched",
                "uncertainty": 0.05,
                "n_observations": 5.0,
            },
            {
                "x": 1,
                "y": 2,
                "node_id": 2,
                "organism": "Escherichia coli",
                "density": 1.0,
                "activity_mean": 0.90,
                "activity_class": "active_enriched",
                "uncertainty": 0.10,
                "n_observations": 3.0,
            },
            {
                "x": 2,
                "y": 1,
                "node_id": 3,
                "organism": "Escherichia coli",
                "density": 3.0,
                "activity_mean": 0.20,
                "activity_class": "inactive_enriched",
                "uncertainty": 0.15,
                "n_observations": 7.0,
            },
            {
                "x": 2,
                "y": 2,
                "node_id": 4,
                "organism": "Bacillus subtilis",
                "density": 1.5,
                "activity_mean": 0.85,
                "activity_class": "active_enriched",
                "uncertainty": 0.08,
                "n_observations": 4.0,
            },
        ]
    )
    nodes.to_parquet(root / "nodes.parquet", index=False)
    save_file(
        {
            "gtm.phi": np.array(
                [
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [0.5, 0.5, 1.0],
                ],
                dtype=np.float32,
            ),
            "gtm.weights": np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "scaler.mean": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "scaler.scale": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        },
        str(root / "landscape.safetensors"),
    )
    return root


def test_list_design_engines_reports_wae_and_llm(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    result = toolkit.list_design_engines()

    assert result["default_engine"] == "wae"
    assert {engine["name"] for engine in result["engines"]} == {"wae", "llm"}


def test_wae_design_filters_normalizes_and_deduplicates_candidates(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)
    monkeypatch.setattr(
        toolkit,
        "sample_peptides",
        lambda **kwargs: ["A C D", "ACD", "B", ""],
    )

    result = toolkit.design_peptides(
        goal="Generate peptides",
        engine="wae",
        n_candidates=4,
        return_format="list",
    )

    assert [candidate["sequence"] for candidate in result] == ["A C D"]
    assert result[0]["engine"] == "wae"
    assert result[0]["properties"]["length"] == 3


def test_wae_analog_and_interpolation_wrappers_dispatch(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)
    calls = []

    def fake_neighborhood(**kwargs):
        calls.append(("neighborhood", kwargs))
        return ["A C E"]

    def fake_interpolate(**kwargs):
        calls.append(("interpolate", kwargs))
        return ["A C D", "A C E"]

    monkeypatch.setattr(toolkit, "explore_latent_neighborhood", fake_neighborhood)
    monkeypatch.setattr(toolkit, "interpolate_peptides", fake_interpolate)

    analogs = toolkit.generate_peptide_analogs(
        seed_sequence="A C D",
        n_analogs=1,
        return_format="list",
    )
    interpolation = toolkit.design_peptide_interpolation(
        sequence1="A C D",
        sequence2="A C E",
        n_steps=2,
        return_format="list",
    )

    assert analogs[0]["sequence"] == "A C E"
    assert interpolation[0]["sequence"] == "A C D"
    assert calls[0][0] == "neighborhood"
    assert calls[0][1]["base_sequence"] == "A C D"
    assert calls[1][0] == "interpolate"
    assert calls[1][1]["seq2"] == "A C E"


def test_llm_engine_parses_structured_peptide_proposals(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    class _FakeAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, prompt, stream=False):
            return SimpleNamespace(
                content={
                    "candidates": [
                        {"sequence": "ACD", "rationale": "compact valid", "score": 0.8},
                        {"sequence": "B", "rationale": "invalid", "score": 0.1},
                    ]
                }
            )

    monkeypatch.setattr(peptide_designer_module, "Agent", _FakeAgent)
    agent = SimpleNamespace(model=object(), session_state={})

    result = toolkit.design_peptides(
        goal="Design short peptides",
        engine="llm",
        n_candidates=2,
        return_format="list",
        agent=agent,
    )

    assert [candidate["sequence"] for candidate in result] == ["A C D"]
    assert result[0]["engine"] == "llm"
    assert result[0]["rationale"] == "compact valid"


def test_llm_engine_requires_agent_model(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    try:
        toolkit.design_peptides(
            goal="Design peptides",
            engine="llm",
            n_candidates=1,
            return_format="list",
        )
    except PeptideDesignerError as exc:
        assert "requires an agent with a model" in str(exc)
    else:
        raise AssertionError("Expected PeptideDesignerError")


def test_summary_mode_stores_artifact_pointer_without_inline_candidates(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)
    monkeypatch.setattr(
        toolkit,
        "sample_peptides",
        lambda **kwargs: ["A C D", "A C E", "A C F"],
    )
    agent = SimpleNamespace(session_state={})
    shared_state = {}

    summary = toolkit.design_peptides(
        goal="Generate peptides",
        engine="wae",
        n_candidates=3,
        session_key="test_peptides",
        agent=agent,
        session_state=shared_state,
    )

    pointer = shared_state["test_peptides"]
    assert summary["session_key"] == "test_peptides"
    assert summary["count_returned"] == 3
    assert pointer["peptide_candidate_set_id"] == summary["peptide_candidate_set_id"]
    assert "candidates" not in pointer
    assert pointer["artifact_path"].endswith(".json")
    assert shared_state["session_objects"]["current"]["analysis"] == "ana_001"

    loaded = toolkit.load_peptide_design_candidates(
        "test_peptides",
        session_state=shared_state,
    )
    assert loaded["count"] == 3
    assert {candidate["sequence"] for candidate in loaded["candidates"]} == {
        "A C D",
        "A C E",
        "A C F",
    }


def test_validate_and_rank_peptide_design_candidates(monkeypatch, tmp_path):
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    validated = toolkit.validate_design_candidates(["ACD", "A C E", "B"])
    ranked = toolkit.rank_design_candidates(validated, seed_sequence="A C D")

    assert ranked[0]["sequence"] == "A C D"
    assert ranked[0]["properties"]["seed_sequence_similarity"] == 1.0
    assert ranked[-1]["valid"] is False


def test_llm_design_engine_parses_json_string_response():
    engine = LLMPeptideDesignEngine(model=object())

    result = engine._parse_response('{"candidates": [{"sequence": "A C D", "score": 0.7}]}')

    assert result.candidates[0].sequence == "A C D"
    assert result.candidates[0].score == 0.7


def test_peptide_landscape_lists_organisms_from_cached_bundle(monkeypatch, tmp_path):
    _write_synthetic_peptide_landscape(tmp_path)
    monkeypatch.setenv("PEPTIDE_DESIGNER_DATA_PATH", str(tmp_path))
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    result = toolkit.list_peptide_landscape_organisms()

    assert result["landscape_id"] == "dbaasp_amp_v1"
    assert result["raw_source_data_redistributed"] is False
    assert result["organisms"] == ["Bacillus subtilis", "Escherichia coli"]
    assert result["plotted_organisms"] == ["Escherichia coli"]


def test_sample_peptides_from_landscape_persists_analysis_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _write_synthetic_peptide_landscape(tmp_path)
    monkeypatch.setenv("PEPTIDE_DESIGNER_DATA_PATH", str(tmp_path))
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)
    decoded = iter(["A C D", "A C E", "A C F", "A C D", "A C E"] * 4)

    def fake_decode(latent_vectors, **kwargs):
        return [next(decoded) for _ in latent_vectors]

    monkeypatch.setattr(toolkit, "decode_latent", fake_decode)
    shared_state = {}

    summary = toolkit.sample_peptides_from_landscape(
        organism="E. coli",
        n_candidates=3,
        top_n_nodes=2,
        local_noise_scale=0.0,
        random_state=7,
        session_state=shared_state,
    )

    pointer = shared_state["landscape_sampled_peptides"]
    assert summary["count_returned"] == 3
    assert summary["selected_node_ids"] == [1, 2]
    assert pointer["raw_source_data_redistributed"] is False
    assert pointer["identity_summary"]["n_unique_sequences"] == 3
    assert pointer["diversity_summary"]["unique_node_count"] >= 1
    assert pointer["seq2logo_png"].endswith(".png")
    assert shared_state["session_objects"]["current"]["analysis"] == "ana_001"

    with S3.open(pointer["generated_peptides_csv_rel_path"], "r") as handle:
        generated = pd.read_csv(handle)
    assert set(generated["sequence"]) == {"A C D", "A C E", "A C F"}
    assert set(generated["properties_node_id"]).issubset({1, 2})


def test_sample_peptides_from_node_coordinates_returns_inline_candidates(monkeypatch, tmp_path):
    _write_synthetic_peptide_landscape(tmp_path)
    monkeypatch.setenv("PEPTIDE_DESIGNER_DATA_PATH", str(tmp_path))
    toolkit = _toolkit_without_model(monkeypatch, tmp_path)

    monkeypatch.setattr(
        toolkit,
        "decode_latent",
        lambda latent_vectors, **kwargs: ["A C D" for _ in latent_vectors],
    )

    result = toolkit.sample_peptides_from_node_coordinates(
        coordinates=[[1, 2]],
        organism="Escherichia coli",
        n_candidates=1,
        local_noise_scale=0.0,
        return_format="list",
    )

    assert result[0]["sequence"] == "A C D"
    assert result[0]["engine"] == "wae_landscape"
    assert result[0]["properties"]["node_id"] == 2
    assert result[0]["properties"]["landscape_id"] == "dbaasp_amp_v1"


def test_peptide_designer_prompt_prefers_hf_aggregate_landscapes():
    from cs_copilot.agents.prompts import PEPTIDE_DESIGNER_INSTRUCTIONS

    joined = "\n".join(PEPTIDE_DESIGNER_INSTRUCTIONS)
    assert "Do NOT request raw DBAASP records" in joined
    assert "sample_peptides_from_landscape" in joined
    assert "Logomaker sequence-logo" in joined
