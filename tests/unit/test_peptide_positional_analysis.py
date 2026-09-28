"""Positional analysis uses declared denominators and real saved plot artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from cs_copilot.tools.chemistry import peptide_designer_toolkit as toolkit_module
from cs_copilot.tools.chemistry.peptide_analysis import (
    analyze_sequences,
    positional_identity,
    render_frequency_logo,
)


def test_identity_distinguishes_permutations_and_variable_lengths():
    assert positional_identity("A C D", "ACD") == 1
    assert positional_identity("AC", "CA") == 0
    assert positional_identity("ACD", "AC") == pytest.approx(2 / 3)
    metrics, pairs, frequency = analyze_sequences(["ACD", "A C D", "AC"])
    assert metrics["exact_sequence_uniqueness"] == pytest.approx(2 / 3)
    assert metrics["pairwise_comparison_count"] == 3
    assert len(pairs) == 3
    assert frequency["n_observed"].tolist() == [3, 3, 2]
    assert frequency.loc[2, "D"] == 1
    np.testing.assert_allclose(frequency.drop(columns=["position", "n_observed"]).sum(axis=1), 1)


def test_single_sequence_and_short_tail_denominators_are_explicit():
    metrics, pairs, frequency = analyze_sequences(["AC"])
    assert metrics["mean_pairwise_similarity"] is None
    assert pairs.empty
    assert frequency["n_observed"].tolist() == [1, 1]
    _, _, frequency = analyze_sequences(["ACD", "AX".replace("X", "E")])
    assert frequency.loc[1, "C"] == frequency.loc[1, "E"] == 0.5
    assert frequency.loc[2, "D"] == 1.0
    with pytest.raises(ValueError, match="20 standard"):
        analyze_sequences(["ACB"])


def test_logo_outputs_real_png_and_vector_svg():
    _, _, frequency = analyze_sequences(["ACD", "AED", "AC"])
    outputs = render_frequency_logo(frequency)
    assert outputs["png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in outputs["svg"]
    assert b"N-terminal" in outputs["svg"]
    assert len(outputs["png"]) > 1000


def test_analysis_tool_persists_checksummed_artifacts(monkeypatch, tmp_path):
    cls = toolkit_module.PeptideDesignerToolkit
    monkeypatch.setattr(cls, "_ensure_model_exists", lambda self: None)
    monkeypatch.setattr(cls, "_load_model", lambda self: None)
    toolkit = cls(model_path=str(tmp_path), device="cpu")
    source = tmp_path / "candidate.json"
    source.write_text(
        json.dumps(
            {
                "peptide_candidate_set_id": "p1",
                "metadata": {},
                "candidates": [
                    {"sequence": "A C D", "valid": True},
                    {"sequence": "A C", "valid": True},
                ],
            }
        )
    )
    monkeypatch.setattr(toolkit_module.S3, "open", lambda path, mode: Path(path).open(mode))
    monkeypatch.setattr(toolkit_module.S3, "path", lambda path: str(path))
    monkeypatch.setattr(
        toolkit_module,
        "_peptide_design_artifact_rel_path",
        lambda *args, **kwargs: str(tmp_path / "analysis.json"),
    )
    state = {"landscape_sampled_peptides": {"artifact_path": str(source)}}
    result = toolkit.analyze_peptide_candidates(session_state=state)
    assert "analyze_peptide_candidates" in toolkit.functions
    assert result["mean_pairwise_similarity"] == pytest.approx(2 / 3)
    artifacts = result["artifacts"]
    metrics = json.loads(Path(artifacts["metrics_json"]).read_text())
    assert metrics["candidate_artifact_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads(Path(artifacts["manifest_json"]).read_text())
    assert len(manifest["artifacts"]) == 5
    for record in manifest["artifacts"]:
        assert hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() == record["sha256"]
    assert state["peptide_sequence_analysis"] == result
