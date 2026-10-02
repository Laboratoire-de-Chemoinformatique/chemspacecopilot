"""Combine per-batch reliability bundles into the manuscript result tables.

This reads only retained run records. It performs no inference, no scientific
computation and no network access, and it never reruns or substitutes a run.

Two conventions are load-bearing:

* Missing telemetry is *unavailable*, not zero. Records whose
  ``token_metrics_status``/``telemetry_status`` is not ``complete`` are excluded
  from the corresponding efficiency statistics and reported as coverage instead.
* Task fulfilment (``task_success``) is kept distinct from completing without a
  tool error (``execution_status``) and from scientific outcomes such as finding
  a route. A truthful completed search that finds no route can still fulfil an
  orchestration task.

Architecture differences from this design are paired but small-sample; they are
reported as exploratory tradeoffs with intervals, never as a proven advantage.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ARMS = ("team", "single_agent")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict:
    """Wilson score interval; more honest than normal approximation at small n."""
    if total == 0:
        return {"low": None, "high": None}
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total) / denom
    return {"low": round(max(0.0, centre - margin), 4), "high": round(min(1.0, centre + margin), 4)}


def describe(values: list[float]) -> dict:
    """Median/IQR summary; n is the count actually observed, not the count planned."""
    if not values:
        return {"n": 0, "median": None, "iqr_low": None, "iqr_high": None, "min": None, "max": None}
    ordered = sorted(values)
    if len(ordered) >= 4:
        quartiles = statistics.quantiles(ordered, n=4, method="inclusive")
        iqr_low, iqr_high = round(quartiles[0], 3), round(quartiles[2], 3)
    else:
        iqr_low = iqr_high = None
    return {
        "n": len(ordered),
        "median": round(statistics.median(ordered), 3),
        "iqr_low": iqr_low,
        "iqr_high": iqr_high,
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


def load_records(study_dir: Path, plan: dict) -> list[dict]:
    """Attach global variant/repetition indices from the plan to each local record."""
    by_batch = {batch["batch_id"]: batch for batch in plan["batches"]}
    records: list[dict] = []
    runs_root = study_dir / "runs"
    for batch_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        batch = by_batch.get(batch_dir.name)
        if batch is None:
            raise SystemExit(f"run directory has no matching plan batch: {batch_dir.name}")
        for bundle in sorted(batch_dir.glob("*/reliability/runs.jsonl")):
            for line in bundle.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                record["batch_id"] = batch["batch_id"]
                record["global_prompt_variant"] = batch["global_prompt_variant"]
                record["global_repetition"] = batch["global_repetition"]
                record["arm_order"] = batch["arm_order"]
                record["source_bundle"] = str(bundle)
                records.append(record)
    return records


def efficiency(records: list[dict], field: str, status_field: str) -> dict:
    """Summarize a metric over only those records whose telemetry is complete."""
    usable = [r for r in records if r.get(status_field) == "complete" and r.get(field) is not None]
    summary = describe([float(r[field]) for r in usable])
    summary["coverage"] = f"{len(usable)}/{len(records)}"
    return summary


def summarize(records: list[dict]) -> dict:
    total = len(records)
    successes = sum(1 for r in records if r.get("task_success"))
    completed = sum(1 for r in records if r.get("execution_status") == "success")
    # tool-call counts are only meaningful where telemetry was actually captured
    with_tools = [r for r in records if r.get("telemetry_status") == "complete"]
    tool_calls = sum(r.get("tool_call_count") or 0 for r in with_tools)
    failed_calls = sum(r.get("failed_tool_call_count") or 0 for r in with_tools)
    return {
        "executions": total,
        "task_success": successes,
        "task_success_rate": round(successes / total, 4) if total else None,
        "task_success_ci95": wilson_interval(successes, total),
        "completed_without_agent_exception": completed,
        "completion_rate": round(completed / total, 4) if total else None,
        "tool_calls_observed": tool_calls,
        "failed_tool_calls_observed": failed_calls,
        "failed_tool_call_fraction": (round(failed_calls / tool_calls, 4) if tool_calls else None),
        "tool_telemetry_coverage": f"{len(with_tools)}/{total}",
        "wall_time_seconds": efficiency(records, "wall_time_seconds", "telemetry_status"),
        "total_tokens": efficiency(records, "total_tokens", "token_metrics_status"),
        "cache_read_tokens": efficiency(records, "cache_read_tokens", "token_metrics_status"),
        "failure_categories": dict(
            Counter(c for r in records for c in (r.get("failure_categories") or [])).most_common()
        ),
    }


def repeatability(records: list[dict]) -> dict:
    """Agreement of task_success across repetitions of an identical cell."""
    cells: dict[tuple, list[bool]] = defaultdict(list)
    for r in records:
        key = (r.get("case_name"), r.get("system_under_test"), r.get("global_prompt_variant"))
        cells[key].append(bool(r.get("task_success")))
    unanimous = sum(1 for outcomes in cells.values() if len(set(outcomes)) == 1)
    return {
        "cells": len(cells),
        "unanimous_cells": unanimous,
        "unanimous_fraction": round(unanimous / len(cells), 4) if cells else None,
        "detail": {
            "|".join(str(part) for part in key): {
                "outcomes": outcomes,
                "successes": sum(outcomes),
                "repetitions": len(outcomes),
            }
            for key, outcomes in sorted(cells.items(), key=lambda kv: str(kv[0]))
        },
    }


def paired_architecture(records: list[dict]) -> dict:
    """Pair arms on identical (case, variant, repetition) cells."""
    paired: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        key = (r.get("case_name"), r.get("global_prompt_variant"), r.get("global_repetition"))
        paired[key][r.get("system_under_test")] = r
    complete = {k: v for k, v in paired.items() if all(arm in v for arm in ARMS)}

    both, team_only, single_only, neither = 0, 0, 0, 0
    deltas_wall: list[float] = []
    deltas_tokens: list[float] = []
    for pair in complete.values():
        team_ok = bool(pair["team"].get("task_success"))
        single_ok = bool(pair["single_agent"].get("task_success"))
        if team_ok and single_ok:
            both += 1
        elif team_ok:
            team_only += 1
        elif single_ok:
            single_only += 1
        else:
            neither += 1
        if all(p.get("telemetry_status") == "complete" for p in pair.values()):
            deltas_wall.append(
                float(pair["team"]["wall_time_seconds"]) - float(pair["single_agent"]["wall_time_seconds"])
            )
        if all(p.get("token_metrics_status") == "complete" for p in pair.values()):
            deltas_tokens.append(
                float(pair["team"]["total_tokens"]) - float(pair["single_agent"]["total_tokens"])
            )

    discordant = team_only + single_only
    return {
        "paired_cells": len(complete),
        "incomplete_pairs": len(paired) - len(complete),
        "both_succeeded": both,
        "team_only": team_only,
        "single_agent_only": single_only,
        "neither": neither,
        "discordant_pairs": discordant,
        "exact_mcnemar_note": (
            "With {} discordant pairs a two-sided exact binomial test has no useful power; "
            "the direction is reported descriptively, not as a significant difference."
        ).format(discordant),
        "team_minus_single_wall_seconds": describe(deltas_wall),
        "team_minus_single_total_tokens": describe(deltas_tokens),
        "interpretation_guard": (
            "Paired but exploratory. Delegation overhead and failure propagation are "
            "measured tradeoffs at this sample size, not evidence of superiority."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, help="Directory holding study_plan.json and runs/")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    plan = json.loads((args.study_dir / "study_plan.json").read_text(encoding="utf-8"))
    records = load_records(args.study_dir, plan)
    if not records:
        raise SystemExit("no retained run records found")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_tier: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_tier[r.get("tier") or "unknown"].append(r)

    frozen = by_tier.get("frozen", [])
    per_case_arm = {}
    for case in sorted({r.get("case_name") for r in records}):
        for arm in ARMS:
            subset = [r for r in records if r.get("case_name") == case and r.get("system_under_test") == arm]
            if subset:
                per_case_arm[f"{case}|{arm}"] = summarize(subset)

    report = {
        "purpose": "Retained-run summary of the predeclared reliability and architecture study",
        "plan_commit_at_planning": plan.get("git_commit_at_planning"),
        "planned_total_executions": plan.get("total_executions"),
        "retained_executions": len(records),
        "provider": {
            "model_id": records[0].get("model_id"),
            "provider": records[0].get("model_provider"),
        },
        "overall": summarize(records),
        "by_tier": {tier: summarize(rows) for tier, rows in sorted(by_tier.items())},
        # Pooled across tiers, so it is NOT the matched comparison: the live tier
        # runs the team arm only. Use by_tier_and_arm or architecture_paired.
        "by_arm_pooled_across_tiers": {
            arm: summarize([r for r in records if r.get("system_under_test") == arm])
            for arm in ARMS
            if any(r.get("system_under_test") == arm for r in records)
        },
        "by_tier_and_arm": {
            f"{tier}|{arm}": summarize(subset)
            for tier, rows in sorted(by_tier.items())
            for arm in ARMS
            if (subset := [r for r in rows if r.get("system_under_test") == arm])
        },
        "by_case_and_arm": per_case_arm,
        "repeatability": repeatability(frozen or records),
        "architecture_paired": paired_architecture(frozen or records),
        "limitations": [
            "Task fulfilment is judged by automated artifact validators; this is not "
            "independent chemical or factual expert review.",
            "Missing telemetry is reported as coverage and excluded from efficiency "
            "statistics rather than counted as zero.",
            "Token counts are provider/Agno usage records, not a cost estimate; "
            "cache-hit counts are retained separately.",
            "No provider seed exists; repeated runs are not claimed deterministic.",
            "Counts include delegation overhead for the team arm by construction.",
        ],
    }

    (args.output_dir / "reliability_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = [
            "batch_id", "tier", "case_name", "system_under_test", "global_prompt_variant",
            "global_repetition", "arm_order", "task_success", "execution_status",
            "wall_time_seconds", "tool_call_count", "failed_tool_call_count",
            "total_tokens", "input_tokens", "output_tokens", "reasoning_tokens",
            "cache_read_tokens", "telemetry_status", "token_metrics_status", "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    print(json.dumps({
        "retained_executions": report["retained_executions"],
        "planned": report["planned_total_executions"],
        "overall_task_success": report["overall"]["task_success_rate"],
        "by_tier_and_arm": {
            k: v["task_success_rate"] for k, v in report["by_tier_and_arm"].items()
        },
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
