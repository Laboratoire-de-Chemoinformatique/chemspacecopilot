"""Real denominator checks with deterministic fake backend outputs, no models or network."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cs_copilot.generation_audit import capture_generation_audit
from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit
from cs_copilot.tools.chemistry.molecular_designer_toolkit import MolecularDesignerToolkit

_SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_generation.py"
_spec = importlib.util.spec_from_file_location("summarize_generation", _SCRIPT)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)


def _backend():
    toolkit = AutoencoderToolkit.__new__(AutoencoderToolkit)
    toolkit.device = torch.device("cpu")
    toolkit.model_path = "test-only-no-download"
    toolkit.config = SimpleNamespace(d_z=2)
    toolkit.validate_model_loaded = lambda: True
    toolkit.model = SimpleNamespace(
        eval=lambda: None, sample=lambda **_kwargs: ["OCC", "CCO", "bad", "CCN"]
    )
    return toolkit


def test_backend_capture_precedes_standardization_and_dedup(tmp_path):
    toolkit = _backend()
    result = toolkit.sample_molecules(
        n_samples=4, return_format="list", audit_path=str(tmp_path / "raw.json")
    )
    assert result == ["CCO", "CCN"]
    audit = json.loads((tmp_path / "raw.json").read_text())
    assert audit["raw_outputs"] == ["OCC", "CCO", "bad", "CCN"]
    assert audit["observed_output_count"] == 4
    assert audit["backend_attempt_count"] is None
    assert audit["raw_outputs_available"] is True


def test_facade_keeps_original_backend_strings_and_compact_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=_backend())
    summary = toolkit.design_molecules("test", n_candidates=4, session_state={})
    assert summary["count_returned"] == 2
    assert summary["generation_audit"]["observed_output_count"] == 4
    assert "raw_outputs" not in summary["metadata"]["generation_audit"]
    from cs_copilot.storage import S3

    with S3.open(summary["generation_audit"]["artifact_path"]) as handle:
        audit = json.load(handle)
    assert audit["raw_outputs"] == ["OCC", "CCO", "bad", "CCN"]


def test_raw_rates_exact_membership_and_parent_properties():
    audit = capture_generation_audit(
        ["OCC", "CCO", "bad", "CCN"], source="test", requested_count=10, raw_outputs_available=True
    )
    result = metrics.summarize(audit, parent_smiles="CCO", training_smiles=["OCC"])
    assert result["requested_count"] == 10
    assert result["observed_record_count"] == 4
    assert result["raw_output_validity"] == 0.75
    assert result["raw_output_uniqueness"] == pytest.approx(2 / 3)
    assert result["novel_to_training_fraction"] == 0.5
    assert result["parent_reconstruction_count"] == 2
    assert result["records"][0]["parent_tanimoto"] == 1.0
    assert result["records"][0]["molecular_weight"] == pytest.approx(46.069)
    assert result["records"][1]["duplicate_of_earlier_valid"] is True


def test_archived_filtered_outputs_cannot_establish_generation_rates():
    result = metrics.summarize(
        {"count_attempted": 200, "candidates": [{"smiles": "CCO", "valid": True}]}
    )
    assert result["raw_output_validity"] is None
    assert result["raw_output_uniqueness"] is None
    assert result["backend_attempt_count"] is None
    assert result["novel_to_training_fraction"] is None
    assert result["observed_record_count"] == 1


def test_unknown_backend_capture_does_not_invent_raw_rates():
    audit = capture_generation_audit(
        ["CCO"],
        source="third_party_returned_candidates",
        requested_count=200,
        raw_outputs_available=False,
    )
    result = metrics.summarize(audit)
    assert result["raw_output_validity"] is None


def test_empty_outputs_have_undefined_rates():
    result = metrics.summarize(
        capture_generation_audit([], source="test", requested_count=5, raw_outputs_available=True)
    )
    assert result["raw_output_validity"] is None
    assert result["raw_output_uniqueness"] is None


@pytest.mark.parametrize("training", [[], ["bad"]])
def test_invalid_or_empty_training_corpus_fails_closed(training):
    with pytest.raises(ValueError):
        metrics.summarize(["CCO"], training_smiles=training)


def test_cli_writes_complete_rows_and_hashes(tmp_path):
    source = tmp_path / "raw.json"
    source.write_text(
        json.dumps(
            capture_generation_audit(
                ["CCO", "CCO", "bad"], source="test", requested_count=3, raw_outputs_available=True
            )
        )
    )
    training = tmp_path / "train.smi"
    training.write_text("OCC ethanol\n")
    prefix = tmp_path / "report"
    metrics.main([str(source), "--output-prefix", str(prefix), "--training-corpus", str(training)])
    result = json.loads(prefix.with_suffix(".json").read_text())
    assert len(result["records"]) == 3
    assert result["novel_to_training_fraction"] == 0
    assert len(result["input_sha256"]) == len(result["training_corpus_sha256"]) == 64
    assert len(prefix.with_suffix(".csv").read_text().splitlines()) == 4


@pytest.mark.parametrize("mode", ["analog", "interpolate"])
def test_facade_captures_neighborhood_and_interpolation_outputs(tmp_path, mode):
    import numpy as np

    backend = _backend()
    backend.encode_smiles_array = lambda _smiles: np.zeros((1, 2))
    toolkit = MolecularDesignerToolkit(autoencoder_toolkit=backend)
    path = tmp_path / f"{mode}.json"
    result = toolkit.design_molecules(
        "test",
        n_candidates=4,
        seed_smiles="CCO",
        generation_mode=mode,
        constraints={"smiles2": "CCN"},
        return_format="list",
        audit_path=str(path),
    )
    audit = json.loads(path.read_text())
    assert audit["raw_outputs"] == ["OCC", "CCO", "bad", "CCN"]
    assert audit["raw_outputs_available"] is True
    assert audit["requested_count"] == 4
    assert len(result) == 2


def test_llm_proposals_captured_before_validation_and_dedup(monkeypatch):
    from cs_copilot.tools.chemistry import molecular_designer_toolkit as module

    monkeypatch.setattr(
        module,
        "Agent",
        lambda **_kwargs: SimpleNamespace(
            run=lambda *_args, **_kwargs2: SimpleNamespace(
                content={
                    "candidates": [
                        {"smiles": "OCC"},
                        {"smiles": "CCO"},
                        {"smiles": "bad"},
                    ]
                }
            )
        ),
    )
    result = module.LLMDesignEngine(model=object()).design(
        module.MolecularDesignRequest(goal="test", n_candidates=5)
    )
    assert result.generation_audit["raw_outputs"] == ["OCC", "CCO", "bad"]
    assert result.generation_audit["observed_output_count"] == 3
    assert result.generation_audit["backend_attempt_count"] is None
    assert len(result.candidates) == 2


def test_duplicate_and_invalid_training_or_output_records_are_not_hidden():
    audit = capture_generation_audit(
        [None, "", "bad", "CCO", "OCC"],
        source="test",
        requested_count=5,
        raw_outputs_available=True,
    )
    result = metrics.summarize(audit, training_smiles=["CCO", "OCC"])
    assert result["observed_record_count"] == 5
    assert result["raw_output_validity"] == 0.4
    assert result["raw_output_uniqueness"] == 0.5
    assert result["training_unique_standardized_count"] == 1
    assert result["novel_to_training_fraction"] == 0


def test_incomplete_audit_rejected():
    audit = capture_generation_audit(
        ["CCO"], source="test", requested_count=5, raw_outputs_available=True
    )
    audit["observed_output_count"] = 5
    with pytest.raises(ValueError, match="every observed output"):
        metrics.summarize(audit)


@pytest.mark.parametrize("payload", [None, 42, "not a candidates document"])
def test_invalid_payload_type_is_rejected_cleanly(payload):
    with pytest.raises(ValueError, match="JSON object/list"):
        metrics.summarize(payload)


@pytest.mark.parametrize("overwrite", ["input", "training"])
def test_cli_rejects_overwriting_input_or_training(tmp_path, overwrite):
    source = tmp_path / "input.json"
    source.write_text('["CCO"]')
    training = tmp_path / "training.csv"
    training.write_text("smiles\nCCO\n")
    target = source if overwrite == "input" else training
    with pytest.raises(SystemExit) as error:
        metrics.main(
            [
                str(source),
                "--output-prefix",
                str(target.with_suffix("")),
                "--training-corpus",
                str(training),
            ]
        )
    assert error.value.code == 2
    assert source.read_text() == '["CCO"]'
    assert training.read_text() == "smiles\nCCO\n"
