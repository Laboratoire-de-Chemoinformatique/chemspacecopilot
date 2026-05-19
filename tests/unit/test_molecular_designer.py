#!/usr/bin/env python
# coding: utf-8
"""Tests for the Molecular Designer engine facade."""

from types import SimpleNamespace

from cs_copilot.agents.registry import list_available_agent_types
from cs_copilot.tools.chemistry.molecular_designer_toolkit import (
    LLMDesignEngine,
    MolecularCandidate,
    MolecularDesignerToolkit,
    MolecularDesignResult,
)
from cs_copilot.tools.io.session_memory import load_candidate_set_artifact, resolve_candidate_set


class _FakeAutoencoderToolkit:
    def __init__(self):
        self.calls = []

    def sample_molecules(self, **kwargs):
        self.calls.append(("sample_molecules", kwargs))
        return ["CCO", "CCO", "not-a-smiles", "c1ccccc1"]

    def explore_latent_neighborhood(self, **kwargs):
        self.calls.append(("explore_latent_neighborhood", kwargs))
        return ["CCO", "CCN"]

    def interpolate_molecules(self, **kwargs):
        self.calls.append(("interpolate_molecules", kwargs))
        return ["CCO", "CCN"]


def test_molecular_designer_public_registry_name_replaces_autoencoder():
    """The public agent type should be Molecular Designer, not Autoencoder."""
    agent_types = list_available_agent_types()

    assert "molecular_designer" in agent_types
    assert "autoencoder" not in agent_types


def test_autoencoder_engine_design_filters_and_deduplicates_candidates():
    """Facade autoencoder dispatch should return valid canonical candidates only."""
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())

    result = toolkit.design_molecules(
        goal="Generate small molecules",
        engine="autoencoder",
        n_candidates=4,
        return_format="list",
    )

    assert {candidate["smiles"] for candidate in result} == {"CCO", "c1ccccc1"}
    assert all(candidate["valid"] for candidate in result)
    assert all(candidate["engine"] == "autoencoder" for candidate in result)


def test_design_molecules_summary_saves_full_result_as_artifact(monkeypatch, tmp_path):
    """Summary mode should keep full candidate lists out of the LLM-visible session state."""
    monkeypatch.chdir(tmp_path)
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())
    agent = SimpleNamespace(session_state={})
    shared_state = {}

    summary = toolkit.design_molecules(
        goal="Generate small molecules",
        engine="autoencoder",
        n_candidates=4,
        session_key="test_designs",
        agent=agent,
        session_state=shared_state,
    )

    assert summary["session_key"] == "test_designs"
    assert summary["count_returned"] == 2
    assert len(summary["preview"]) == 2
    assert summary["artifact_path"].endswith(
        "02_analog_generation/candidate_sets/cset_001/candidates.json"
    )
    assert shared_state["test_designs"]["candidate_set_id"] == "cset_001"
    assert shared_state["test_designs"]["artifact_path"] == summary["artifact_path"]
    assert shared_state["test_designs"]["count"] == 2
    assert shared_state["session_objects"]["current"]["compound"] == "cmp_001"
    assert shared_state["session_objects"]["current"]["candidate_set"] == "cset_001"
    assert shared_state["session_objects"]["current"]["generated_compounds"] == "cset_001"
    assert shared_state["session_objects"]["compounds"]["cmp_001"]["smiles"] in {
        "CCO",
        "c1ccccc1",
    }
    assert shared_state["session_objects"]["compounds"]["cmp_001"]["origin_agent"] == (
        "molecular_designer"
    )
    assert shared_state["session_objects"]["compounds"]["cmp_001"]["generation_engine"] == (
        "autoencoder"
    )
    assert {item["smiles"] for item in shared_state["session_objects"]["compounds"].values()} == {
        "CCO",
        "c1ccccc1",
    }
    candidate_set = shared_state["session_objects"]["candidate_sets"]["cset_001"]
    assert candidate_set["origin_agent"] == "molecular_designer"
    assert candidate_set["generation_engine"] == "autoencoder"
    assert candidate_set["compound_ids"] == ["cmp_001", "cmp_002"]
    assert candidate_set["artifact_path"] == summary["artifact_path"]
    assert summary["registered_candidate_set_id"] == "cset_001"
    assert set(agent.session_state["session_objects"]["compounds"]) == {"cmp_001", "cmp_002"}
    artifact = load_candidate_set_artifact(shared_state, "test_designs")
    assert {candidate["smiles"] for candidate in artifact["candidates"]} == {"CCO", "c1ccccc1"}


def test_interpolation_uses_autoencoder_interpolation_not_analog_dispatch():
    """Interpolation has a seed SMILES but should not be treated as analog generation."""
    fake_autoencoder = _FakeAutoencoderToolkit()
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=fake_autoencoder)

    result = toolkit.interpolate_molecules(
        smiles1="CCO",
        smiles2="CCN",
        n_steps=2,
        return_format="list",
    )

    assert result
    assert fake_autoencoder.calls[0][0] == "interpolate_molecules"


def test_llm_engine_results_are_filtered_by_facade(monkeypatch):
    """LLM-proposed invalid SMILES should not survive default design output."""
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())

    def fake_design(self, request):
        return MolecularDesignResult(
            engine="llm",
            candidates=[
                MolecularCandidate(
                    smiles="CCO",
                    original_smiles="CCO",
                    engine="llm",
                    valid=True,
                    score=0.8,
                ),
                MolecularCandidate(
                    smiles=None,
                    original_smiles="not-a-smiles",
                    engine="llm",
                    valid=False,
                    error="invalid",
                ),
            ],
        )

    monkeypatch.setattr(LLMDesignEngine, "design", fake_design)
    agent = SimpleNamespace(model=object(), session_state={})

    result = toolkit.design_molecules(
        goal="Design alcohol-like molecules",
        engine="llm",
        n_candidates=2,
        return_format="list",
        agent=agent,
    )

    assert [candidate["smiles"] for candidate in result] == ["CCO"]
    assert result[0]["engine"] == "llm"


def test_validate_and_rank_design_candidates_scores_seed_similarity():
    """Candidate ranking should prioritize seed-similar valid molecules."""
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())

    validated = toolkit.validate_design_candidates(["CCO", "c1ccccc1", "bad"])
    ranked = toolkit.rank_design_candidates(validated, seed_smiles="CCO")

    assert ranked[0]["smiles"] == "CCO"
    assert ranked[0]["properties"]["seed_tanimoto"] == 1.0
    assert ranked[-1]["valid"] is False


def test_register_design_candidates_tool_is_exposed():
    """The facade should expose explicit persistence for validated/ranked outputs."""
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())

    assert "register_design_candidates" in toolkit.functions


def test_register_design_candidates_persists_ranked_autoencoder_candidate_set(
    monkeypatch, tmp_path
):
    """Validated/ranked low-level autoencoder candidates become an artifact-backed set."""
    monkeypatch.chdir(tmp_path)
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())
    agent = SimpleNamespace(session_state={})
    shared_state = {}

    validated = toolkit.validate_design_candidates(["CCO", "CCN", "bad"], engine="autoencoder")
    ranked = toolkit.rank_design_candidates(validated, seed_smiles="CCO")
    summary = toolkit.register_design_candidates(
        ranked,
        engine="autoencoder",
        generation_mode="analog",
        seed_smiles="CCO",
        goal="Register low-level autoencoder analogs.",
        session_key="autoencoder_candidates",
        agent=agent,
        session_state=shared_state,
    )

    memory = shared_state["session_objects"]
    candidate_set = memory["candidate_sets"]["cset_001"]

    assert summary["status"] == "registered"
    assert summary["registered_candidate_set_id"] == "cset_001"
    assert summary["artifact_path"].endswith(
        "02_analog_generation/candidate_sets/cset_001/candidates.json"
    )
    assert summary["registered_compound_ids"] == ["cmp_001", "cmp_002"]
    assert candidate_set["generation_engine"] == "autoencoder"
    assert candidate_set["generation_mode"] == "analog"
    assert candidate_set["compound_ids"] == ["cmp_001", "cmp_002"]
    assert shared_state["autoencoder_candidates"]["candidate_set_id"] == "cset_001"
    assert memory["current"]["candidate_set"] == "cset_001"
    assert memory["current"]["generated_compounds"] == "cset_001"
    assert [
        shared_state["session_objects"]["compounds"][cid]["smiles"]
        for cid in candidate_set["compound_ids"]
    ] == [
        "CCO",
        "CCN",
    ]
    assert (
        agent.session_state["session_objects"]["candidate_sets"]["cset_001"]["generation_engine"]
        == "autoencoder"
    )


def test_autoencoder_registration_resolves_over_older_llm_candidate_set(monkeypatch, tmp_path):
    """Explicit autoencoder registration prevents fallback to older LLM candidates."""
    monkeypatch.chdir(tmp_path)
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())
    shared_state = {}

    toolkit.register_design_candidates(
        [{"smiles": "CCC", "valid": True}],
        engine="llm",
        generation_mode="design",
        goal="Older LLM candidates.",
        session_key="llm_candidates",
        session_state=shared_state,
    )
    toolkit.register_design_candidates(
        [
            {"smiles": "CCO", "valid": True, "ranking_score": 0.9},
            {"smiles": "CCN", "valid": True, "ranking_score": 0.8},
        ],
        engine="autoencoder",
        generation_mode="analog",
        goal="New autoencoder candidates.",
        session_key="autoencoder_candidates",
        session_state=shared_state,
    )

    resolved_autoencoder = resolve_candidate_set(shared_state, "autoencoder candidates")
    resolved_llm = resolve_candidate_set(shared_state, "LLM candidates")

    assert resolved_autoencoder["status"] == "resolved"
    assert resolved_autoencoder["candidate_set"]["id"] == "cset_002"
    assert resolved_autoencoder["candidate_set"]["generation_engine"] == "autoencoder"
    assert [compound["smiles"] for compound in resolved_autoencoder["compounds"]] == [
        "CCO",
        "CCN",
    ]
    assert resolved_llm["candidate_set"]["id"] == "cset_001"
    assert [compound["smiles"] for compound in resolved_llm["compounds"]] == ["CCC"]
