#!/usr/bin/env python
"""Freeze a prospective human sEH IC50 dataset with raw-page and filter provenance.

Run with PYTHONPATH=src. Network access is limited to the public ChEMBL API.
Use --reuse-raw to reprocess a completed, checksum-verified local snapshot.
This does not reconstruct the original manuscript dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse

from rdkit import rdBase

from cs_copilot.tools.chemistry.standardize import standardize_smiles

API = "https://www.ebi.ac.uk/chembl/api/data/"
TARGET = "CHEMBL2409"
EXCLUDED_ASSAYS = {
    "CHEMBL4415272": "Phosphatase activity of bifunctional EPHX2, not epoxide hydrolase activity"
}
POLICY = {
    "purpose": "prospective_measurement_not_historical_reconstruction",
    "target_chembl_id": TARGET,
    "target_organism": "Homo sapiens",
    "target_type": "SINGLE PROTEIN",
    "target_accession": "P34913",
    "assay_confidence_score": 9,
    "assay_types": ["B", "F"],
    "excluded_assays_after_domain_review": EXCLUDED_ASSAYS,
    "assay_organism_filter": "none; human identity follows confidence-9 target assignment",
    "standard_type": "IC50",
    "standard_units": "nM",
    "standard_relation": "=",
    "standard_value": "finite and greater than zero",
    "data_validity_comment": "null or empty only",
    "potential_duplicate": "unflagged only (0, false, null or empty)",
    "chemical_identity": "RDKit cleanup, largest fragment, uncharge, canonical tautomer, remove stereochemistry",
    "duplicate_measurements": "collapse same standardized structure + assay + exact numeric IC50; retain all source IDs",
    "aggregation": "median pIC50 = median(9 - log10(IC50 in nM)); independent assays retained",
    "classification": "active pIC50 > 6 (IC50 < 1000 nM); inactive pIC50 <= 5 (IC50 >= 10000 nM); otherwise intermediate",
    "conflicts": "retain all compounds; flag multiple observed classes and active/inactive disagreement separately",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identity(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def get_bytes(url: str) -> bytes:
    """Retry transient read failures; retain only successful response bodies."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read(32 * 1024 * 1024 + 1)
            if len(data) > 32 * 1024 * 1024:
                raise ValueError("API page exceeds the 32 MiB limit")
            return data
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def _api_url(url: str) -> str:
    resolved = urljoin(API, url)
    parsed = urlparse(resolved)
    if parsed.scheme != "https" or parsed.netloc != "www.ebi.ac.uk":
        raise ValueError("Pagination left the public ChEMBL API origin")
    if not parsed.path.startswith("/chembl/api/data/"):
        raise ValueError("Pagination left the public ChEMBL API path")
    return resolved


def download_snapshot(output: Path, expected_release: str, fetch=get_bytes) -> dict[str, Any]:
    """Save original JSON response bytes and all page URLs before processing."""
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "raw_manifest.json"
    if manifest_path.exists():
        raise ValueError("Raw manifest already exists; use --reuse-raw or a new output directory")
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "chemspacecopilot.seh_snapshot.v1",
        "complete": False,
        "expected_release": expected_release,
        "api_base": API,
        "target_chembl_id": TARGET,
        "artifacts": [],
        "collections": {},
    }

    def save(url: str, name: str) -> dict[str, Any]:
        url = _api_url(url)
        body = fetch(url)
        value = json.loads(body)
        path = raw / name
        path.write_bytes(body)
        manifest["artifacts"].append(
            {
                **identity(path, output),
                "request_url": url,
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(manifest_path, manifest)
        return value

    before = save(API + "status.json", "status_before.json")
    if before.get("chembl_db_version") != expected_release:
        raise ValueError(f"Expected {expected_release}, got {before.get('chembl_db_version')}")
    save(API + f"target/{TARGET}.json", "target.json")
    for endpoint, key in [("assay", "assays"), ("activity", "activities")]:
        url = API + endpoint + ".json?" + urlencode({"target_chembl_id": TARGET, "limit": 1000})
        pages, seen, rows, total = [], set(), 0, None
        while url:
            url = _api_url(url)
            if url in seen:
                raise ValueError("Repeated pagination URL")
            seen.add(url)
            name = f"{endpoint}_{len(pages):04d}.json"
            value = save(url, name)
            meta = value["page_meta"]
            if total is None:
                total = meta["total_count"]
            if meta["total_count"] != total or meta["offset"] != rows:
                raise ValueError("Inconsistent pagination count or offset")
            pages.append(str((raw / name).relative_to(output)))
            rows += len(value[key])
            url = meta.get("next")
        if rows != total:
            raise ValueError(f"Incomplete {endpoint} pagination: {rows} of {total}")
        manifest["collections"][key] = {"paths": pages, "row_count": rows}
        write_json(manifest_path, manifest)
    after = save(API + "status.json", "status_after.json")
    if after.get("chembl_db_version") != expected_release:
        raise ValueError("ChEMBL release changed during extraction")
    manifest["complete"] = True
    write_json(manifest_path, manifest)
    return manifest


def read_snapshot(output: Path) -> tuple[dict, dict, list[dict], list[dict]]:
    manifest = json.loads((output / "raw_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Incomplete raw snapshot")
    root = output.resolve()
    saved = set()
    for item in manifest["artifacts"]:
        path = (output / item["path"]).resolve()
        if not path.is_relative_to(root) or identity(path, root) != {
            k: item[k] for k in ("path", "bytes", "sha256")
        }:
            raise ValueError(f"Raw checksum/path mismatch: {item['path']}")
        saved.add(item["path"])
    for phase in ["before", "after"]:
        path = f"raw/status_{phase}.json"
        if (
            path not in saved
            or json.loads((output / path).read_text())["chembl_db_version"]
            != manifest["expected_release"]
        ):
            raise ValueError("Unverified ChEMBL release")
    if "raw/target.json" not in saved:
        raise ValueError("Unverified target metadata")
    target = json.loads((output / "raw/target.json").read_text())
    values = []
    for key in ["assays", "activities"]:
        collection = manifest["collections"][key]
        records = []
        for path in collection["paths"]:
            if path not in saved:
                raise ValueError("Unverified collection page")
            records.extend(json.loads((output / path).read_text())[key])
        if len(records) != collection["row_count"]:
            raise ValueError("Raw collection count mismatch")
        values.append(records)
    return manifest, target, values[0], values[1]


def _number(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def activity_class(pic50: float) -> str:
    if pic50 > 6:
        return "active"
    return "inactive" if pic50 <= 5 else "intermediate"


def prepare_dataset(target: dict, assays: list[dict], activities: list[dict]) -> dict[str, Any]:
    """Apply predeclared filters and retain record-level exclusions and identities."""
    if (
        target.get("target_chembl_id") != TARGET
        or target.get("organism") != "Homo sapiens"
        or target.get("target_type") != "SINGLE PROTEIN"
        or [c.get("accession") for c in target.get("target_components", [])] != ["P34913"]
    ):
        raise ValueError("Target metadata does not identify human single-protein sEH/P34913")
    assay_by_id = {a["assay_chembl_id"]: a for a in assays}
    if len(assay_by_id) != len(assays):
        raise ValueError("Repeated assay identifiers in raw pages")
    ids = [a.get("activity_id") for a in activities]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or repeated activity identifiers in raw pages")
    excluded, accepted = [], []
    for row in sorted(activities, key=lambda a: a["activity_id"]):
        assay = assay_by_id.get(row.get("assay_chembl_id"), {})
        value = _number(row.get("standard_value"))
        checks = [
            (
                row.get("target_chembl_id") == TARGET
                and row.get("target_organism") == "Homo sapiens",
                "other_target_or_organism",
            ),
            (assay.get("target_chembl_id") == TARGET, "missing_or_other_target_assay"),
            (_number(assay.get("confidence_score")) == 9, "assay_confidence_not_9"),
            (assay.get("assay_type") in {"B", "F"}, "assay_type_not_B_or_F"),
            (row.get("assay_chembl_id") not in EXCLUDED_ASSAYS, "phosphatase_domain_assay"),
            (row.get("standard_type") == "IC50", "endpoint_not_IC50"),
            (row.get("standard_units") == "nM", "units_not_nM"),
            (row.get("standard_relation") == "=", "relation_not_equal"),
            (value is not None and value > 0, "value_not_finite_positive"),
            (row.get("data_validity_comment") in (None, ""), "data_validity_comment"),
            (row.get("potential_duplicate") in (None, "", 0, "0", False), "potential_duplicate"),
        ]
        reason = next((reason for passed, reason in checks if not passed), None)
        smiles = None
        if reason is None:
            smiles = standardize_smiles(row.get("canonical_smiles"))
            if not smiles:
                reason = "invalid_or_empty_standardized_structure"
        if reason:
            excluded.append(
                {
                    "activity_id": row["activity_id"],
                    "assay_chembl_id": row.get("assay_chembl_id"),
                    "reason": reason,
                }
            )
        else:
            accepted.append(
                {
                    **row,
                    "standardized_smiles": smiles,
                    "ic50_nm": value,
                    "pIC50": 9 - math.log10(value),
                }
            )
    grouped = defaultdict(list)
    for row in accepted:
        grouped[(row["standardized_smiles"], row["assay_chembl_id"], row["ic50_nm"])].append(row)
    measurements = []
    for (smiles, assay_id, value), rows in sorted(grouped.items()):
        measurements.append(
            {
                "canonical_smiles": smiles,
                "assay_chembl_id": assay_id,
                "assay_type": assay_by_id[assay_id]["assay_type"],
                "ic50_nm": value,
                "pIC50": 9 - math.log10(value),
                "activity_ids": sorted(r["activity_id"] for r in rows),
                "molecule_chembl_ids": sorted({r["molecule_chembl_id"] for r in rows}),
                "n_source_records": len(rows),
            }
        )
    by_structure = defaultdict(list)
    for row in measurements:
        by_structure[row["canonical_smiles"]].append(row)
    compounds = []
    for smiles, rows in sorted(by_structure.items()):
        values = [r["pIC50"] for r in rows]
        median = statistics.median(values)
        label = activity_class(median)
        classes = {activity_class(v) for v in values}
        compounds.append(
            {
                "canonical_smiles": smiles,
                "target_chembl_id": TARGET,
                "standard_type": "IC50",
                "standard_units": "nM",
                "standard_value": 10 ** (9 - median),
                "pIC50": median,
                "activity_class": label,
                "activity_binary": {"active": 1, "inactive": 0}.get(label),
                "n_measurements": len(rows),
                "n_source_records": sum(r["n_source_records"] for r in rows),
                "pIC50_min": min(values),
                "pIC50_max": max(values),
                "pIC50_range": max(values) - min(values),
                "observed_classes": sorted(classes),
                "activity_class_conflict": len(classes) > 1,
                "active_inactive_conflict": {"active", "inactive"}.issubset(classes),
                "assay_chembl_ids": sorted({r["assay_chembl_id"] for r in rows}),
                "molecule_chembl_ids": sorted({i for r in rows for i in r["molecule_chembl_ids"]}),
                "activity_ids": sorted(i for r in rows for i in r["activity_ids"]),
            }
        )
    return {
        "compounds": compounds,
        "measurements": measurements,
        "excluded": excluded,
        "counts": {
            "raw_assays": len(assays),
            "raw_activities": len(activities),
            "retained_assays": len({r["assay_chembl_id"] for r in measurements}),
            "excluded_by_first_failed_filter": dict(
                sorted(Counter(r["reason"] for r in excluded).items())
            ),
            "accepted_source_records": len(accepted),
            "exact_within_assay_duplicate_measurements_collapsed": len(accepted)
            - len(measurements),
            "unique_measurements": len(measurements),
            "unique_standardized_compounds": len(compounds),
            "class_counts": dict(sorted(Counter(r["activity_class"] for r in compounds).items())),
            "compounds_with_multiple_observed_classes": sum(
                r["activity_class_conflict"] for r in compounds
            ),
            "compounds_with_active_inactive_conflict": sum(
                r["active_inactive_conflict"] for r in compounds
            ),
        },
    }


def write_csv(path: Path, records: list[dict], fields: list[str] | None = None) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or (list(records[0]) if records else []))
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                    for k, v in row.items()
                    if k in writer.fieldnames
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/reviewer_revision/seh"))
    parser.add_argument("--expected-release", default="ChEMBL_37")
    parser.add_argument("--reuse-raw", action="store_true")
    args = parser.parse_args()
    if not args.reuse_raw:
        download_snapshot(args.output_dir, args.expected_release)
    manifest, target, assays, activities = read_snapshot(args.output_dir)
    if manifest["expected_release"] != args.expected_release:
        raise ValueError("Saved snapshot differs from requested ChEMBL release")
    result = prepare_dataset(target, assays, activities)
    if not result["compounds"]:
        raise ValueError("No eligible compounds; inspect saved raw pages and filters")
    for row in result["compounds"]:
        row["chembl_release"] = manifest["expected_release"]
    files = []
    for name, records in [
        ("raw_activities.csv", activities),
        ("clean_measurements.csv", result["measurements"]),
        ("clean_compounds.csv", result["compounds"]),
        ("excluded_activities.csv", result["excluded"]),
    ]:
        path = args.output_dir / name
        write_csv(path, records, sorted({key for row in records for key in row}))
        files.append(identity(path, args.output_dir))
    import cs_copilot.tools.chemistry.standardize as standardization_module

    provenance = {
        "schema": "chemspacecopilot.seh_preparation.v1",
        "policy": POLICY,
        "domain_review_excluded_assays": [
            a for a in assays if a["assay_chembl_id"] in EXCLUDED_ASSAYS
        ],
        "counts": result["counts"],
        "chembl_release": manifest["expected_release"],
        "raw_manifest": identity(args.output_dir / "raw_manifest.json", args.output_dir),
        "outputs": files,
        "software": {"python": platform.python_version(), "rdkit": rdBase.rdkitVersion},
        "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "standardization_source_sha256": hashlib.sha256(
            Path(standardization_module.__file__).read_bytes()
        ).hexdigest(),
    }
    write_json(args.output_dir / "provenance.json", provenance)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
