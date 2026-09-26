#!/usr/bin/env python3
"""Summarize saved SynPlanner plan.json files without running synthesis searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def summarize_plan(path: Path) -> dict[str, Any]:
    """Retain one execution, including no-route, error, and incomplete outcomes."""
    content = path.read_bytes()
    plan = json.loads(content)
    if not isinstance(plan, dict) or not isinstance(plan.get("smiles"), str):
        raise ValueError(f"{path}: expected a SynPlanner plan with target smiles")
    molecule = Chem.MolFromSmiles(plan["smiles"])
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(f"{path}: invalid target SMILES")
    routes = plan.get("routes")
    attempts = plan.get("attempts")
    if not isinstance(routes, list) or any(not isinstance(r, dict) for r in routes):
        raise ValueError(f"{path}: routes must be a list of objects")
    if not isinstance(attempts, list) or any(not isinstance(a, dict) for a in attempts):
        raise ValueError(f"{path}: attempts must be a list of objects")

    errors = [str(a["error"]) for a in attempts if a.get("error")]
    if plan.get("error"):
        errors.append(str(plan["error"]))
    has_no_route = any(a.get("stop_reason") == "no_routes" for a in attempts)
    has_errors = bool(errors) or any(a.get("stop_reason") == "error" for a in attempts)
    if routes:
        outcome = "route_found"
    elif has_no_route and has_errors:
        outcome = "no_route_with_errors"
    elif has_no_route:
        outcome = "no_route"
    elif has_errors:
        outcome = "error"
    else:
        outcome = "unavailable"

    times = [_number(a.get("search_time")) for a in attempts]
    times = [value if value is not None and value >= 0 else None for value in times]
    known_times = [value for value in times if value is not None]
    complete_timing = bool(attempts) and len(known_times) == len(attempts)

    # Preserve backend ordering; the meaning/direction of its score is not inferred.
    first_route = routes[0] if routes else {}
    step_count = first_route.get("num_steps")
    if isinstance(step_count, bool) or not isinstance(step_count, int) or step_count < 0:
        step_count = None
    steps = first_route.get("steps")
    length_check = "unavailable"
    if step_count is not None and isinstance(steps, list):
        length_check = "consistent" if step_count == len(steps) else "inconsistent"
    if step_count == 0:
        length_check = "zero_steps_requires_review"

    return {
        "source_path": str(path.resolve()),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "target_smiles": plan["smiles"],
        "canonical_target_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
        "outcome": outcome,
        "route_count": len(routes),
        "first_route_num_steps": step_count,
        "first_route_score": _number(first_route.get("score")),
        "route_length_check": length_check,
        "attempt_count": len(attempts),
        "error_attempt_count": sum(
            bool(a.get("error")) or a.get("stop_reason") == "error" for a in attempts
        ),
        "search_time_complete": complete_timing,
        "search_seconds": sum(known_times) if complete_timing else None,
        "observed_search_seconds": sum(known_times) if known_times else None,
        "successful_attempt": plan.get("successful_attempt"),
        "attempts_json": json.dumps(attempts, sort_keys=True),
        "errors_json": json.dumps(errors),
        "visualizations_json": json.dumps(plan.get("visualizations", []), sort_keys=True),
    }


def write_summary(paths: list[Path], output_dir: Path) -> dict[str, Any]:
    """Write per-execution rows; do not silently count repeats as new compounds."""
    if not paths:
        raise ValueError("Provide at least one saved plan")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("The same input file was supplied more than once")
    # Validate every input before creating output files.
    rows = [summarize_plan(path) for path in resolved]
    targets = {row["canonical_target_smiles"] for row in rows}
    found = sum(row["outcome"] == "route_found" for row in rows)
    summary = {
        "schema_version": "1.0",
        "rdkit_version": rdBase.rdkitVersion,
        "canonicalization": "RDKit canonical isomeric SMILES; no salt or tautomer normalization",
        "execution_count": len(rows),
        "distinct_target_count": len(targets),
        "repeated_target_executions": len(rows) - len(targets),
        "outcomes": {
            name: sum(row["outcome"] == name for row in rows)
            for name in ("route_found", "no_route", "no_route_with_errors", "error", "unavailable")
        },
        "route_found_fraction_of_supplied_executions": found / len(rows),
        "complete_timing_count": sum(row["search_time_complete"] for row in rows),
        "inputs": [{"path": r["source_path"], "sha256": r["source_sha256"]} for r in rows],
        "interpretation": (
            "Backend-reported predicted routes, not verified synthetic feasibility. "
            "This is a summary of supplied files; establish the full target denominator "
            "from a predeclared target list and retain failures that produced no plan file. "
            "Search seconds exclude setup, rendering, and LLM overhead. First-route "
            "scores and lengths are reported as stored and require scientific review."
        ),
    }
    if any((output_dir / name).resolve() in resolved for name in ("targets.csv", "summary.json")):
        raise ValueError("Output would overwrite an input artifact")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", nargs="+", type=Path, help="Saved per-target plan.json files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = write_summary(args.plans, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
