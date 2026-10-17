"""Positional analysis uses declared denominators and real saved plot artifacts."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cs_copilot.tools.chemistry import peptide_designer_toolkit as toolkit_module
from cs_copilot.tools.chemistry.peptide_analysis import (
    align_sequences,
    analyze_sequences,
    positional_identity,
    render_frequency_logo,
)

RAGGED = ["GIGKFLHSAKKFGKAFVGEIMNS", "GIGAVLKVLTTGLPALISWIKRKRQQ", "FLPIIAKLLGGLL"]


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


def test_pyfamsa_pads_ragged_sequences_into_equal_length_columns():
    pytest.importorskip("pyfamsa")
    aligned, summary = align_sequences(RAGGED)
    assert summary["method"] == "pyfamsa"
    assert summary["guide_tree"] == "upgma"
    assert len({len(sequence) for sequence in aligned}) == 1
    assert summary["gap_count"] > 0
    assert [sequence.replace("-", "") for sequence in aligned] == RAGGED


def test_alignment_survives_sequences_that_share_no_residue():
    """PyFAMSA's UPGMA tree segfaults here, which no except clause can catch."""
    pytest.importorskip("pyfamsa")
    script = (
        "from cs_copilot.tools.chemistry.peptide_analysis import analyze_sequences\n"
        "for sequences in (['AC', 'DE'], ['CI', 'E', 'DEH', 'ICDI']):\n"
        "    metrics, _, _ = analyze_sequences(sequences, align=True)\n"
        "    print(metrics['alignment']['method'], metrics['alignment']['guide_tree'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.split() == ["pyfamsa", "sl", "pyfamsa", "sl"]


def test_alignment_disables_nondeterministic_refinement(monkeypatch):
    pyfamsa = pytest.importorskip("pyfamsa")
    calls = []
    real_aligner = pyfamsa.Aligner

    def recording_aligner(**kwargs):
        calls.append(kwargs)
        return real_aligner(**kwargs)

    monkeypatch.setattr(pyfamsa, "Aligner", recording_aligner)
    _, summary = align_sequences(RAGGED)
    assert [call["refine"] for call in calls] == [False]
    assert summary["refine"] is False


def test_alignment_degrades_without_two_sequences():
    aligned, summary = align_sequences(["ACD"])
    assert aligned == ["ACD"]
    assert summary["method"] == "n_terminal"
    assert "Fewer than two" in summary["reason"]


def test_aligned_similarity_compares_the_logo_columns():
    pytest.importorskip("pyfamsa")
    extended = ["WKLAKLAKK", "KLAKLAKK"]
    aligned, _, _ = analyze_sequences(extended, align=True)
    raw, _, _ = analyze_sequences(extended)
    assert aligned["mean_pairwise_similarity"] == pytest.approx(8 / 9)
    assert raw["mean_pairwise_similarity"] < 0.5
    assert "PyFAMSA" in aligned["similarity_definition"]
    assert "N-terminal" in raw["similarity_definition"]


def test_low_support_counts_distinct_sequences_per_position():
    metrics, _, _ = analyze_sequences(["ACDEF"] * 5)
    assert metrics["low_support_positions"] == [1, 2, 3, 4, 5]
    metrics, _, _ = analyze_sequences(["ACDEF", "ACDEG", "ACDEH", "ACDEI", "ACDEKL"])
    assert metrics["low_support_positions"] == [6]


def test_aligned_analysis_records_provenance_and_renders_with_logomaker():
    pytest.importorskip("pyfamsa")
    pytest.importorskip("logomaker")
    metrics, _, frequency = analyze_sequences(RAGGED, align=True)
    assert metrics["alignment"]["method"] == "pyfamsa"
    assert metrics["aligned_sequences"] is not None
    assert "PyFAMSA" in metrics["logo_definition"]
    assert metrics["low_support_positions"] == frequency["position"].tolist()
    np.testing.assert_allclose(frequency.drop(columns=["position", "n_observed"]).sum(axis=1), 1)
    outputs = render_frequency_logo(
        frequency, aligned=True, title="Peptide Sequence Logo for GTM Node 509"
    )
    assert outputs["png"].startswith(b"\x89PNG\r\n\x1a\n")
    assert b"<svg" in outputs["svg"]
    assert len(outputs["png"]) > 1000
    # The title and axis label must reach the figure; a wrong label is how a logo ends up
    # claiming an alignment it does not show.
    assert b"Alignment position" in outputs["svg"]
    assert b"no multiple-sequence alignment" not in outputs["svg"]
    assert b"GTM Node 509" in outputs["svg"]
    # Both engines colour a residue the same way (G is green in Logomaker's chemistry scheme).
    assert b"#008000" in outputs["svg"]
    assert b"#008000" in render_frequency_logo(frequency)["svg"]


def test_unaligned_analysis_keeps_its_own_provenance():
    metrics, _, _ = analyze_sequences(RAGGED)
    assert metrics["alignment"]["method"] == "n_terminal"
    assert metrics["alignment"]["reason"] == "Alignment not requested."
    assert metrics["aligned_sequences"] is None
    assert "not multiple-sequence alignment" in metrics["logo_definition"]


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
    assert result["alignment"]["reason"] == "Alignment not requested."
    assert result["low_support_positions"] == [1, 2, 3]
