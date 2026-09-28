#!/usr/bin/env python
"""Compare generated identities with a named reference, without claiming training novelty.

Every reference row receives the same structure standardization as the candidates.
This streaming comparison records row counts, not a deduplicated corpus size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from rdkit import rdBase

from cs_copilot.tools.chemistry.standardize import standardize_smiles


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def csv_smiles(path, column):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing SMILES column {column!r} in {path}")
        for row in reader:
            yield row[column]


def compare(candidates, reference, *, workers=1, progress_every=0):
    if workers < 1:
        raise ValueError("workers must be positive")
    identities = set()
    for index, smiles in enumerate(candidates, 1):
        identity = standardize_smiles(smiles)
        if not identity:
            raise ValueError(f"Invalid candidate at row {index}")
        identities.add(identity)
    if not identities:
        raise ValueError("No candidates supplied")
    counts, first_match = Counter(), {}
    n_rows = n_invalid = 0
    pool = multiprocessing.get_context("spawn").Pool(workers) if workers > 1 else None
    try:
        canonical_rows = (
            pool.imap(standardize_smiles, reference, chunksize=500)
            if pool
            else map(standardize_smiles, reference)
        )
        for n_rows, identity in enumerate(canonical_rows, 1):
            if not identity:
                n_invalid += 1
            elif identity in identities:
                counts[identity] += 1
                first_match.setdefault(identity, n_rows)
            if progress_every and n_rows % progress_every == 0:
                print(
                    f"Compared {n_rows} reference rows; {len(counts)} candidate matches", flush=True
                )
    finally:
        if pool:
            pool.close()
            pool.join()
    if not n_rows:
        raise ValueError("Reference corpus is empty")
    complete = n_invalid == 0
    return {
        "schema": "chemspacecopilot.reference_membership.v1",
        "reference_rows": n_rows,
        "reference_standardized_valid_rows": n_rows - n_invalid,
        "reference_invalid_rows": n_invalid,
        "reference_fully_standardized": complete,
        "candidate_unique_count": len(identities),
        "candidate_present_count": len(counts),
        "candidate_absent_count": len(identities) - len(counts) if complete else None,
        "candidate_absent_fraction": (
            (len(identities) - len(counts)) / len(identities) if complete else None
        ),
        "training_membership_confirmed": False,
        "novel_to_training_count": None,
        "definitions": {
            "identity": "Repository cleanup, largest fragment, uncharge, canonical tautomer, remove stereochemistry, canonical SMILES; applied identically to every candidate and reference row.",
            "reference_denominator": "Input rows; duplicate reference structures are not deduplicated or reported as unique.",
            "missing_rows": "Absent counts remain unavailable if any reference row fails standardization.",
            "scope": "Exact membership in the named reference only. The supplied reference is not verified as the exact checkpoint training corpus; this is not a training-novelty estimate.",
        },
        "candidates": [
            {
                "canonical_smiles": identity,
                "present_in_reference": (
                    bool(counts[identity]) if complete or counts[identity] else None
                ),
                "matching_reference_rows": counts[identity],
                "first_matching_reference_row": first_match.get(identity),
            }
            for identity in sorted(identities)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--candidate-column", default="canonical_smiles")
    parser.add_argument("--reference-column", default="SMILES")
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--reference-url", required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve() in {
        args.candidates.resolve(),
        args.reference.resolve(),
    }:
        parser.error("Output must be a new file and must not overwrite inputs")
    before = {"candidates": sha256(args.candidates), "reference": sha256(args.reference)}
    if before["reference"] != args.expected_reference_sha256:
        parser.error("Reference checksum does not match the predeclared source")
    started = time.perf_counter()
    report = compare(
        csv_smiles(args.candidates, args.candidate_column),
        csv_smiles(args.reference, args.reference_column),
        workers=args.workers,
        progress_every=25000,
    )
    if before != {"candidates": sha256(args.candidates), "reference": sha256(args.reference)}:
        raise ValueError("Inputs changed during the comparison")
    from cs_copilot.tools.chemistry import standardize

    report.update(
        {
            "reference_name": args.reference_name,
            "reference_url": args.reference_url,
            "inputs": {
                "candidates": {"path": str(args.candidates), "sha256": before["candidates"]},
                "reference": {"path": str(args.reference), "sha256": before["reference"]},
            },
            "rdkit_version": rdBase.rdkitVersion,
            "implementation_sha256": {
                "scripts/compare_reference_corpus.py": sha256(__file__),
                "src/cs_copilot/tools/chemistry/standardize.py": sha256(standardize.__file__),
            },
            "workers": args.workers,
            "wall_time_seconds": time.perf_counter() - started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
