#!/usr/bin/env python3
"""Export all declared retrosynthesis targets, including failures without plan files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

from _study_common import TERMINAL, file_identity, write_json
from summarize_retrosynthesis import summarize_plan


def summarize_study(study_dir: Path, output_dir: Path) -> dict:
    study_dir = study_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir == study_dir or study_dir in output_dir.parents:
        raise ValueError("Keep the exported summary outside the source study directory")
    manifest_path = study_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    spec = manifest["specification"]
    if spec.get("study") != "retrosynthesis":
        raise ValueError("Expected a retrosynthesis study manifest")
    jobs = spec["jobs"]
    ids = [job["id"] for job in jobs]
    if not jobs or len(set(ids)) != len(ids):
        raise ValueError("Study jobs must be nonempty and have unique IDs")
    rows, inputs = [], [file_identity(manifest_path)]
    seed_controls = {}
    for job in jobs:
        if Path(job["id"]).name != job["id"] or job["id"] in (".", ".."):
            raise ValueError("Unsafe job identifier")
        case_dir = study_dir / job["id"]
        execution_path = case_dir / "execution.json"
        record = json.loads(execution_path.read_text()) if execution_path.is_file() else {}
        if record and (record.get("id") != job["id"] or record.get("seed") != job["seed"]):
            raise ValueError(f"{job['id']}: execution disagrees with declaration")
        status = record.get("status", "not_started")
        if status not in TERMINAL | {"not_started", "running"}:
            raise ValueError(f"{job['id']}: unknown execution status")
        result = record.get("result", {})
        seed_controls[job["id"]] = result.get("seed_control")
        if execution_path.is_file():
            inputs.append(file_identity(execution_path))
        row = {
            "target_id": job["target_id"],
            "job_id": job["id"],
            "smiles": job["smiles"],
            "seed": job["seed"],
            "execution_status": status,
            "outcome": status,
            "route_count": None,
            "first_route_num_steps": None,
            "first_route_score": None,
            "route_length_check": "unavailable",
            "backend_reported_search_seconds": None,
            "worker_wall_seconds": record.get("wall_seconds"),
            "attempt_count": None,
            "iterations": None,
            "error": result.get("error") or record.get("error"),
            "plan_sha256": None,
        }
        plan_path = case_dir / "plan.json"
        if plan_path.is_file():
            identity = file_identity(plan_path)
            declared = result.get("artifacts", {}).get("plan.json")
            if status == "completed" and not declared:
                raise ValueError(f"{job['id']}: completed plan has no declared artifact hash")
            if declared and declared.get("sha256") != identity["sha256"]:
                raise ValueError(f"{job['id']}: plan no longer matches execution artifact hash")
            plan = summarize_plan(plan_path)
            if plan["canonical_target_smiles"] != job["canonical_smiles"]:
                raise ValueError(f"{job['id']}: plan target disagrees with declaration")
            if status == "completed" and result.get("scientific_outcome") != plan["outcome"]:
                raise ValueError(f"{job['id']}: ledger and plan outcomes disagree")
            inputs.append(identity)
            # A partial plan from a failed/interrupted worker is not promoted to success.
            row.update(
                outcome=plan["outcome"] if status == "completed" else status,
                route_count=plan["route_count"],
                first_route_num_steps=plan["first_route_num_steps"],
                first_route_score=plan["first_route_score"],
                route_length_check=plan["route_length_check"],
                backend_reported_search_seconds=plan["search_seconds"],
                attempt_count=plan["attempt_count"],
                iterations=sum(a.get("iterations", 0) for a in json.loads(plan["attempts_json"])),
                error=row["error"] or "; ".join(json.loads(plan["errors_json"])) or None,
                plan_sha256=identity["sha256"],
            )
        elif status == "completed":
            raise ValueError(f"{job['id']}: completed execution is missing its plan")
        rows.append(row)
    terminal = all(row["execution_status"] in TERMINAL for row in rows)
    outcomes = dict(Counter(row["outcome"] for row in rows))
    times = [row["worker_wall_seconds"] for row in rows if row["worker_wall_seconds"] is not None]
    summary = {
        "schema": "chemspacecopilot.retrosynthesis_study_summary.v1",
        "finalized": terminal,
        "declared_target_count": len(jobs),
        "attempted_target_count": sum(row["execution_status"] != "not_started" for row in rows),
        "outcomes": outcomes,
        "route_found_count": outcomes.get("route_found", 0),
        "route_found_fraction_of_declared_targets": outcomes.get("route_found", 0) / len(jobs),
        "worker_wall_seconds": {
            "observed_count": len(times),
            "total": sum(times) if times else None,
            "median": statistics.median(times) if times else None,
            "min": min(times) if times else None,
            "max": max(times) if times else None,
        },
        "manifest_created_at": manifest["created_at"],
        "configuration": spec["configuration"],
        "seed_controls": seed_controls,
        "software": spec["software"],
        "assets": spec["assets"],
        "implementation_files": spec["implementation_files"],
        "launcher": spec["launcher"],
        "ledger_code": spec["ledger_code"],
        "target_source": spec["target_source"],
        "case_timeout_seconds": spec["case_timeout_seconds"],
        "inputs": inputs,
        "interpretation": (
            "Every predeclared target is retained, including failures before a plan was written. "
            "Only completed executions with verified plan hashes count as route_found. "
            "Predicted routes are not experimentally verified synthetic feasibility. "
            "Backend-reported search time may omit its final iteration; worker wall time includes "
            "loading, search and rendering. No-route describes only these fixed budgets."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "targets.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = summarize_study(args.study_dir, args.output_dir)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
