"""Reference membership must not be promoted to checkpoint training novelty."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_reference_corpus", Path(__file__).parents[2] / "scripts/compare_reference_corpus.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_same_identity_policy_and_distinct_denominators():
    result = module.compare(["CCO", "OCC", "CC(=O)O"], ["OCC", "CCO", "CCC"])
    assert result["candidate_unique_count"] == 2
    assert result["reference_rows"] == 3
    assert result["candidate_present_count"] == 1
    assert result["candidate_absent_count"] == 1
    assert result["candidate_absent_fraction"] == 0.5
    assert result["novel_to_training_count"] is None
    assert result["training_membership_confirmed"] is False
    ethanol = next(row for row in result["candidates"] if row["canonical_smiles"] == "CCO")
    assert ethanol["matching_reference_rows"] == 2
    assert ethanol["first_matching_reference_row"] == 1


def test_salt_and_stereo_standardization_matches():
    result = module.compare(["C[C@H](O)C(=O)O"], ["[Na+].C[C@@H](O)C(=O)[O-]"])
    assert result["candidate_present_count"] == 1


def test_invalid_reference_rows_leave_nonmembership_unknown():
    result = module.compare(["CCO", "CCC"], ["OCC", "not a molecule"])
    assert result["reference_invalid_rows"] == 1
    assert result["candidate_absent_count"] is None
    assert result["candidate_absent_fraction"] is None
    assert {
        row["canonical_smiles"]: row["present_in_reference"] for row in result["candidates"]
    } == {"CCO": True, "CCC": None}


@pytest.mark.parametrize(
    "candidates,reference", [([], ["CCO"]), (["invalid"], ["CCO"]), (["CCO"], [])]
)
def test_invalid_input_fails(candidates, reference):
    with pytest.raises(ValueError):
        module.compare(candidates, reference)
