"""Prospective frozen inputs preserve curated scientific thresholds."""

import importlib.util
import json
import types
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "revision_fixtures", Path(__file__).parents[2] / "scripts/prepare_revision_fixtures.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_curated_view_preserves_intermediate_gap_and_continuous_values():
    frame = pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN", "CCC"],
            "pIC50": [7.0, 5.0, 5.5],
            "activity_class": ["active", "inactive", "intermediate"],
            "activity_binary": [1, 0, None],
            "standard_type": ["IC50"] * 3,
            "standard_units": ["nM"] * 3,
            "standard_value": [100, 10000, 3162],
        }
    )
    view = module.curated_view(frame)
    assert view["activity_comment"].iloc[:2].tolist() == ["active", "inactive"]
    assert pd.isna(view["activity_comment"].iloc[2])
    assert view["pIC50"].tolist() == frame["pIC50"].tolist()
    assert view["median_IC50_nM"].tolist() == [100, 10000, 3162]
    assert "standard_type" not in view and "standard_value" not in view
    assert "canonical_smiles" in frame and "activity_comment" not in frame


def test_missing_curated_labels_cannot_silently_be_inferred():
    with pytest.raises(ValueError, match="Missing curated fields"):
        module.curated_view(pd.DataFrame({"canonical_smiles": ["CCO"]}))


def _fake_build_env(tmp_path, monkeypatch, validate, *, backend='chembed', candidate_smiles=None):
    """Stand up a minimal build() environment; `validate` replaces the backend guard."""
    import numpy as np

    curated = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN"],
            "pIC50": [7.0, 5.0],
            "activity_class": ["active", "inactive"],
            "activity_binary": [1, 0],
            "standard_type": ["IC50"] * 2,
            "standard_units": ["nM"] * 2,
            "standard_value": [100, 10000],
        }
    ).to_csv(curated, index=False)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    model_file = tmp_path / "manifold.pkl.gz"
    model_file.write_bytes(b"not-a-real-pickle")
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {"canonical_smiles": candidate_smiles or ["CCOC", "CCNC"]}
    ).to_csv(candidates, index=False)

    gtm = types.SimpleNamespace(num_nodes=2)
    if backend is not None:
        gtm.descriptor_backend = backend
    projected = pd.DataFrame({"smi": ["CCO", "CCN"], "pIC50": [7.0, 5.0]})
    responsibilities = np.array([[0.6, 0.4], [0.3, 0.7]])
    descriptors = np.zeros((2, 256))
    calls = {}

    def fake_data_load_and_prep(dataset, model_path, descriptor_type=None, **kwargs):
        calls["descriptor_type"] = descriptor_type
        return gtm, projected.copy(), descriptors, responsibilities

    import cs_copilot.tools.chemography.gtm_operations as real_ops

    monkeypatch.setattr(real_ops, "data_load_and_prep", fake_data_load_and_prep)
    monkeypatch.setattr(real_ops, "validate_gtm_descriptor_backend", validate)
    monkeypatch.setattr(
        real_ops, "calculate_latent_coords", lambda r, **k: pd.DataFrame({"x": [0.0, 1.0]})
    )
    monkeypatch.setattr(real_ops, "get_density_matrix", lambda r: r)
    monkeypatch.setattr(real_ops, "density_to_table", lambda d, **k: pd.DataFrame({"node": [0, 1]}))
    monkeypatch.setattr(real_ops, "get_reg_density_matrix", lambda r, a: (r, r))
    monkeypatch.setattr(
        real_ops, "reg_density_to_table", lambda d, a, **k: pd.DataFrame({"node": [0, 1]})
    )

    args = types.SimpleNamespace(
        curated_csv=curated,
        provenance=provenance,
        gtm_model=model_file,
        candidates_csv=candidates,
        output_dir=tmp_path / "out",
    )
    return args, gtm, calls


def test_builder_projects_chembed_and_records_it_everywhere(tmp_path, monkeypatch):
    """Every recorded descriptor family and engine must say chembed, by value.

    Asserted positively rather than as the absence of the string "autoencoder":
    the codebase's own name for the retired engine is "deepchemography"
    (descriptors.py _DESCRIPTOR_ENGINES), so an absence check on one spelling
    passes while the other is stamped on a frozen artifact.
    """
    calls = {}

    def fake_validate(model, descriptor_type):
        calls["validated"] = descriptor_type

    args, gtm, load_calls = _fake_build_env(tmp_path, monkeypatch, fake_validate)
    manifest = module.build(args)

    assert load_calls["descriptor_type"] == "chembed"
    assert calls["validated"] == "chembed", "builder never checked the manifold backend"

    analysis = json.loads((tmp_path / "out" / "seh_analysis.json").read_text())["session_state"]
    assert analysis["default_descriptor"] == "chembed"
    maps = analysis["session_objects"]["maps"]
    assert len(maps) == 1, sorted(maps)
    assert next(iter(maps.values()))["descriptor_type"] == "chembed"

    candidates_state = json.loads((tmp_path / "out" / "seh_candidates.json").read_text())[
        "session_state"
    ]
    candidate_sets = candidates_state["session_objects"]["candidate_sets"]
    assert len(candidate_sets) == 1, sorted(candidate_sets)
    candidate_set = next(iter(candidate_sets.values()))
    assert candidate_set["engine"] == "chembed"
    assert "chembed" in candidate_set["label"]
    assert "autoencoder" not in json.dumps(manifest)
    assert "deepchemography" not in json.dumps(manifest)


def test_builder_writes_no_projection_when_the_manifold_backend_is_wrong(tmp_path, monkeypatch):
    """The guard must stop the build, not annotate a build that already happened.

    Placing the call after the artifacts are written would leave every fixture on
    disk, digested and ready to wire into the study, before anything checked that
    the manifold matched the latents projected onto it.
    """

    def refusing_validate(model, descriptor_type):
        raise ValueError("GTM manifold was fitted on 'autoencoder' descriptors")

    args, _gtm, _calls = _fake_build_env(tmp_path, monkeypatch, refusing_validate)

    with pytest.raises(ValueError, match="fitted on 'autoencoder'"):
        module.build(args)

    out = tmp_path / "out"
    assert not (out / "seh_projection.csv").exists(), "projection written despite a refused manifold"
    assert not (out / "seh_analysis.json").exists(), "analysis fixture written despite refusal"
    assert not (out / "manifest.json").exists(), "manifest written despite refusal"


def test_builder_refuses_a_manifold_that_records_no_descriptor_backend(tmp_path, monkeypatch):
    """An untagged manifold must not be frozen on an assumption.

    validate_gtm_descriptor_backend deliberately lets an untagged manifold
    through with a warning, because older published manifolds carry no tag. That
    is the right call for the library and the wrong one for a builder that
    freezes manuscript fixtures: the warning goes to stderr while this script's
    stdout is a JSON manifest, so it is easy to miss entirely.
    """

    def permissive_validate(model, descriptor_type):
        return None  # mimics the library's warn-and-continue on an untagged manifold

    args, _gtm, _calls = _fake_build_env(
        tmp_path, monkeypatch, permissive_validate, backend=None
    )

    with pytest.raises(ValueError, match="descriptor_backend"):
        module.build(args)

    assert not (tmp_path / "out" / "manifest.json").exists()


def test_builder_refuses_an_unparseable_candidate_smiles(tmp_path, monkeypatch):
    """A partly-invalid candidate set must not be frozen silently.

    The builder previously rejected only a wholly empty list, so a single
    unparseable SMILES landed in candidates.json and the frozen candidate set --
    and if it sat at row 0 it became Case 3's fixed planning target.
    """

    def fake_validate(model, descriptor_type):
        return None

    args, _gtm, _calls = _fake_build_env(
        tmp_path, monkeypatch, fake_validate, candidate_smiles=["CCOC", "C1CC"]
    )

    with pytest.raises(ValueError, match="candidate SMILES"):
        module.build(args)


def test_builder_refuses_a_candidate_csv_whose_rows_were_silently_dropped(tmp_path, monkeypatch):
    """A row pandas discards must not shrink the frozen set in silence.

    read_csv treats a whitespace-only line as blank and drops it, so a candidate
    file with such a row yields fewer candidates than it has rows and no error.
    The frozen set would simply be short, which no SMILES-parse check can catch
    because the bad value never reaches the frame.
    """

    def fake_validate(model, descriptor_type):
        return None

    args, _gtm, _calls = _fake_build_env(tmp_path, monkeypatch, fake_validate)
    args.candidates_csv.write_text("canonical_smiles\nCCOC\n   \nCCNC\n")

    with pytest.raises(ValueError, match="row"):
        module.build(args)
