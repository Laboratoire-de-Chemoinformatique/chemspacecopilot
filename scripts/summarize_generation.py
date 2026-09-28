#!/usr/bin/env python
"""Summarize saved molecular outputs without invoking models or network services.

Example:
  python scripts/summarize_generation.py raw_audit.json --output-prefix results/generation \
      --parent-smiles CCO --training-corpus training.smi

Training files must be the actual generator training corpus (one SMILES per line,
optionally followed by an identifier, or CSV with --training-smiles-column).
Archived candidates may already be filtered: their generation rates remain null.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import QED, Descriptors, rdFingerprintGenerator

from cs_copilot.tools.chemistry.standardize import standardize_smiles

AUDIT_SCHEMA = "chemspacecopilot.generation_audit.v1"
PROPERTY_FUNCTIONS = {
    "molecular_weight": Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotatable_bonds": Descriptors.NumRotatableBonds,
    "qed": QED.qed,
}


def _canonical(smiles: Any) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return standardize_smiles(smiles)


def _smiles(item: Any) -> Any:
    if isinstance(item, dict):
        for key in ("original_smiles", "smiles", "canonical_smiles", "smi"):
            if item.get(key) is not None:
                return item[key]
        return None
    return item


def summarize(
    payload: Any, *, parent_smiles: str | None = None, training_smiles: list[str] | None = None
) -> dict[str, Any]:
    """Compute observed-record metrics; never infer raw denominators from requests."""
    if not isinstance(payload, (dict, list)):
        raise ValueError("Expected a raw generation audit or candidates JSON object/list")
    is_audit = isinstance(payload, dict) and payload.get("schema") == AUDIT_SCHEMA
    if is_audit:
        records = payload.get("raw_outputs")
        if not isinstance(records, list) or payload.get("observed_output_count") != len(records):
            raise ValueError("Audit must contain every observed output and its matching count")
        raw_available = payload.get("raw_outputs_available") is True
        parent_smiles = parent_smiles or payload.get("seed_smiles")
        stage = payload.get("capture_stage")
    else:
        records = payload if isinstance(payload, list) else payload.get("candidates")
        if not isinstance(records, list):
            raise ValueError("Expected a raw generation audit or a candidates JSON list/object")
        raw_available = False
        stage = "archived_candidates_filtering_unknown"
        if isinstance(payload, dict):
            parent_smiles = parent_smiles or payload.get("metadata", {}).get("seed_smiles")

    parent = _canonical(parent_smiles) if parent_smiles else None
    if parent_smiles and parent is None:
        raise ValueError("Parent SMILES is invalid")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    parent_fp = fp.GetFingerprint(Chem.MolFromSmiles(parent)) if parent else None
    training_set = None
    if training_smiles is not None:
        if not training_smiles:
            raise ValueError("Training corpus is empty; training novelty cannot be evaluated")
        training_set = set()
        for index, smiles in enumerate(training_smiles, start=1):
            canonical = _canonical(smiles)
            if canonical is None:
                raise ValueError(
                    f"Invalid training SMILES at record {index}; provide a curated corpus"
                )
            training_set.add(canonical)

    rows = []
    seen: set[str] = set()
    for index, item in enumerate(records, start=1):
        raw = _smiles(item)
        canonical = _canonical(raw)
        parsed = Chem.MolFromSmiles(raw) if isinstance(raw, str) and raw.strip() else None
        row = {
            "record_index": index,
            "raw_smiles": raw,
            "rdkit_valid": parsed is not None and parsed.GetNumAtoms() > 0,
            "standardized_valid": canonical is not None,
            "canonical_smiles": canonical,
            "duplicate_of_earlier_valid": canonical in seen if canonical else False,
            "is_parent": canonical == parent if canonical and parent else None,
            "novel_to_training": (
                canonical not in training_set if canonical and training_set is not None else None
            ),
            "parent_tanimoto": None,
        }
        if canonical:
            seen.add(canonical)
            mol = Chem.MolFromSmiles(canonical)
            row.update({name: float(fn(mol)) for name, fn in PROPERTY_FUNCTIONS.items()})
            if parent_fp is not None:
                row["parent_tanimoto"] = float(
                    DataStructs.TanimotoSimilarity(parent_fp, fp.GetFingerprint(mol))
                )
        rows.append(row)

    unique = [
        row for row in rows if row["standardized_valid"] and not row["duplicate_of_earlier_valid"]
    ]
    n_valid = sum(row["rdkit_valid"] for row in rows)
    n_standardized = sum(row["standardized_valid"] for row in rows)
    n_novel = sum(row["novel_to_training"] for row in unique) if training_set is not None else None
    distributions = {}
    for key in [*PROPERTY_FUNCTIONS, "parent_tanimoto"]:
        values = [row[key] for row in unique if row.get(key) is not None]
        distributions[key] = (
            {
                "n": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }
            if values
            else None
        )
    return {
        "schema": "chemspacecopilot.generation_metrics.v1",
        "rdkit_version": rdBase.rdkitVersion,
        "capture_stage": stage,
        "raw_output_denominator_available": raw_available,
        "backend_attempt_count": payload.get("backend_attempt_count") if is_audit else None,
        "requested_count": payload.get("requested_count") if is_audit else None,
        "observed_record_count": len(rows),
        "rdkit_valid_count": n_valid,
        "standardized_valid_count": n_standardized,
        "unique_standardized_count": len(unique),
        "raw_output_validity": n_valid / len(rows) if raw_available and rows else None,
        "raw_output_uniqueness": (
            len(unique) / n_standardized if raw_available and n_standardized else None
        ),
        "novel_to_training_count": n_novel,
        "novel_to_training_fraction": (
            n_novel / len(unique) if n_novel is not None and unique else None
        ),
        "training_unique_standardized_count": (
            len(training_set) if training_set is not None else None
        ),
        "parent_smiles": parent,
        "parent_reconstruction_count": (
            sum(row["is_parent"] is True for row in rows) if parent else None
        ),
        "unique_candidate_distributions": distributions,
        "definitions": {
            "validity": "RDKit parse with at least one atom / observed unfiltered backend outputs",
            "uniqueness": "unique standardized valid molecules / standardized valid observed outputs",
            "novelty": "exact standardized nonmembership in the supplied generator training corpus",
            "standardization": "repository cleanup, largest fragment, uncharge, canonical tautomer, remove stereochemistry",
            "similarity": "Morgan radius=2, 2048-bit fingerprint Tanimoto to standardized parent",
            "limitations": "Internal backend decoding attempts are unknown; archived filtered outputs cannot establish generation rates. Novelty measures only supplied corpus membership, not biological activity or patentability.",
        },
        "records": rows,
    }


def _training_records(path: Path, smiles_column: str) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if smiles_column not in (reader.fieldnames or []):
                raise ValueError(f"Training CSV has no {smiles_column!r} column")
            return [row[smiles_column] for row in reader]
    return [line.strip().split()[0] for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--parent-smiles")
    parser.add_argument("--training-corpus", type=Path)
    parser.add_argument("--training-smiles-column", default="smiles")
    args = parser.parse_args(argv)
    outputs = [args.output_prefix.with_suffix(".json"), args.output_prefix.with_suffix(".csv")]
    protected = {args.input.resolve()}
    if args.training_corpus:
        protected.add(args.training_corpus.resolve())
    if any(path.resolve() in protected for path in outputs):
        parser.error("Output paths must not overwrite the input or training corpus")
    payload = json.loads(args.input.read_text())
    training = (
        _training_records(args.training_corpus, args.training_smiles_column)
        if args.training_corpus
        else None
    )
    report = summarize(payload, parent_smiles=args.parent_smiles, training_smiles=training)
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    report["training_corpus_sha256"] = (
        hashlib.sha256(args.training_corpus.read_bytes()).hexdigest()
        if args.training_corpus
        else None
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with args.output_prefix.with_suffix(".csv").open("w", newline="") as handle:
        keys = list(dict.fromkeys(key for row in report["records"] for key in row))
        writer = csv.DictWriter(handle, fieldnames=keys or ["record_index", "raw_smiles"])
        writer.writeheader()
        writer.writerows(report["records"])


if __name__ == "__main__":
    main()
