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


def test_design_molecules_summary_persists_full_result_in_session_state():
    """Summary mode should keep full candidate lists out of the LLM-visible response."""
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_FakeAutoencoderToolkit())
    agent = SimpleNamespace(session_state={})

    summary = toolkit.design_molecules(
        goal="Generate small molecules",
        engine="autoencoder",
        n_candidates=4,
        session_key="test_designs",
        agent=agent,
    )

    assert summary["session_key"] == "test_designs"
    assert summary["count_returned"] == 2
    assert len(summary["preview"]) == 2
    assert {candidate["smiles"] for candidate in agent.session_state["test_designs"]} == {
        "CCO",
        "c1ccccc1",
    }


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
