#!/usr/bin/env python
# coding: utf-8
"""Tests for the Peptide Designer public facade."""

from types import SimpleNamespace

import cs_copilot.tools as tools
import cs_copilot.tools.chemistry as chemistry
import cs_copilot.tools.chemistry.peptide_designer_toolkit as peptide_designer_module
from cs_copilot.agents.registry import list_available_agent_types
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
