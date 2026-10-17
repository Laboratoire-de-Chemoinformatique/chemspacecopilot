"""The chembed candidate set reproduces its pre-declared retrosynthesis targets."""

import csv
import hashlib
from pathlib import Path

import pytest

CANDIDATES = (
    Path(__file__).parents[2]
    / "reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv"
)
TARGETS = (
    Path(__file__).parents[2]
    / "reports/manuscript_reliability_chembed/generation_chembed/retrosynthesis_targets.csv"
)
SELECTION_SEED = 42


def declared_targets(rows, count):
    """Rank non-parent candidates by the pre-declared rule and take the first `count`."""
    eligible = sorted(
        (row for row in rows if row["is_parent"].strip().lower() not in {"true", "1"}),
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SEED}|{row['canonical_smiles']}".encode()
        ).hexdigest(),
    )
    if count > len(eligible):
        raise ValueError(f"Only {len(eligible)} eligible candidates for {count} targets")
    return [row["canonical_smiles"] for row in eligible[:count]]


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_declared_targets_are_reproducible_from_the_rule():
    rows = _rows(CANDIDATES)
    recorded = [row["smiles"] for row in _rows(TARGETS)]
    assert declared_targets(rows, len(recorded)) == recorded


def test_candidate_smiles_are_all_present_and_parseable():
    from rdkit import Chem, rdBase

    rows = _rows(CANDIDATES)
    assert rows, "candidate set is empty"
    with rdBase.BlockLogs():
        bad = [
            row["canonical_smiles"]
            for row in rows
            if not row["canonical_smiles"].strip()
            or Chem.MolFromSmiles(row["canonical_smiles"]) is None
        ]
    assert bad == [], f"{len(bad)} unparseable candidate SMILES, first: {bad[:3]}"


def test_an_all_parent_candidate_set_cannot_yield_targets():
    rows = [{"canonical_smiles": "CCO", "is_parent": "True"}]
    with pytest.raises(ValueError, match="Only 0 eligible candidates"):
        declared_targets(rows, 10)
