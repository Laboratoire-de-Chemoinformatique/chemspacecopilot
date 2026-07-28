"""Statistics, manifests, and publication-facing reliability reports."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .models import RELIABILITY_SCHEMA_VERSION

SYSTEM_COMPARISON_SCHEMA_VERSION = "1.0"
NON_EVALUABLE_EXECUTION_STATUSES = frozenset({"fixture_error", "prerequisite_error"})


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> Tuple[float, float]:
    """Calculate a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    spread = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
    return max(0.0, (centre - spread) / denominator), min(1.0, (centre + spread) / denominator)


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _distribution(values: Iterable[float | int | None]) -> Dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    return {
        "median": statistics.median(clean) if clean else None,
        "q1": _percentile(clean, 0.25),
        "q3": _percentile(clean, 0.75),
    }


def _repeatability(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("case_name") or "unknown"),
            int(record.get("prompt_variant") or 0),
        )
        groups[key].append(record)

    group_results = []
    for (case_name, prompt_variant), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        successes = sum(bool(record.get("task_success")) for record in group)
        sequences = [
            tuple(
                str(call.get("tool_name") or "")
                for call in (record.get("tool_calls") or [])
                if isinstance(call, dict)
            )
            for record in group
        ]
        most_common_sequence = Counter(sequences).most_common(1)[0][1]
        group_results.append(
            {
                "case_name": case_name,
                "prompt_variant": prompt_variant,
                "repetitions": len(group),
                "task_outcome_agreement": max(successes, len(group) - successes) / len(group),
                "exact_tool_sequence_agreement": most_common_sequence / len(group),
            }
        )

    return {
        "groups_with_repeats": len(group_results),
        "task_outcome_agreement": _distribution(
            item["task_outcome_agreement"] for item in group_results
        ),
        "exact_tool_sequence_agreement": _distribution(
            item["exact_tool_sequence_agreement"] for item in group_results
        ),
        "groups": group_results,
    }


def summarize_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    successful = sum(bool(record.get("task_success")) for record in records)
    low, high = wilson_interval(successful, total)
    total_tools = sum(int(record.get("tool_call_count") or 0) for record in records)
    failed_tools = sum(int(record.get("failed_tool_call_count") or 0) for record in records)
    cost_values = [
        float(record["estimated_cost"])
        for record in records
        if record.get("estimated_cost") is not None
    ]
    categories = Counter(
        category for record in records for category in (record.get("failure_categories") or [])
    )

    by_case: Dict[str, Dict[str, Any]] = {}
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("case_name") or "unknown")].append(record)
    for case_name, case_records in sorted(grouped.items()):
        case_total = len(case_records)
        case_success = sum(bool(record.get("task_success")) for record in case_records)
        case_low, case_high = wilson_interval(case_success, case_total)
        by_case[case_name] = {
            "runs": case_total,
            "successful": case_success,
            "success_rate": case_success / case_total if case_total else 0,
            "wilson_95": [case_low, case_high],
        }

    return {
        "schema_version": RELIABILITY_SCHEMA_VERSION,
        "runs": total,
        "successful": successful,
        "success_rate": successful / total if total else 0,
        "wilson_95": [low, high],
        "tool_calls": total_tools,
        "failed_tool_calls": failed_tools,
        "failed_tool_calls_per_100": failed_tools / total_tools * 100 if total_tools else 0,
        "incorrect_tool_selection_runs": categories.get("incorrect_tool_selection", 0),
        "incorrect_tool_selection_runs_per_100": (
            categories.get("incorrect_tool_selection", 0) / total * 100 if total else 0
        ),
        "execution_statuses": dict(
            Counter(str(record.get("execution_status") or "unknown") for record in records)
        ),
        "wall_time_seconds_total": sum(
            float(record.get("wall_time_seconds") or 0) for record in records
        ),
        "total_tokens_sum": sum(int(record.get("total_tokens") or 0) for record in records),
        "estimated_cost_total": sum(cost_values) if cost_values else None,
        "wall_time_seconds": _distribution(record.get("wall_time_seconds") for record in records),
        "total_tokens": _distribution(record.get("total_tokens") for record in records),
        "tool_calls_per_run": _distribution(record.get("tool_call_count") for record in records),
        "estimated_cost": _distribution(record.get("estimated_cost") for record in records),
        "failure_categories": dict(categories),
        "by_case": by_case,
        "repeatability": _repeatability(records),
    }


def _comparison_key(record: Mapping[str, Any]) -> Tuple[str, str, int, int]:
    return (
        str(record.get("tier") or "both"),
        str(record.get("case_name") or "unknown"),
        int(record.get("prompt_variant") or 0),
        int(record.get("repetition") or 0),
    )


def _serialized_comparison_key(key: Tuple[str, str, int, int]) -> Dict[str, Any]:
    tier, case_name, prompt_variant, repetition = key
    return {
        "tier": tier,
        "case_name": case_name,
        "prompt_variant": prompt_variant,
        "repetition": repetition,
    }


def _index_comparison_records(
    records: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> Dict[Tuple[str, str, int, int], Mapping[str, Any]]:
    indexed: Dict[Tuple[str, str, int, int], Mapping[str, Any]] = {}
    for record in records:
        key = _comparison_key(record)
        if key in indexed:
            raise ValueError(
                "Duplicate comparison record for "
                f"arm={arm!r}, tier={key[0]!r}, case={key[1]!r}, "
                f"prompt_variant={key[2]}, repetition={key[3]}"
            )
        indexed[key] = record
    return indexed


def _is_architecture_evaluable(record: Mapping[str, Any]) -> bool:
    """Return whether a run actually exercised an agentic architecture."""
    return str(record.get("execution_status") or "unknown") not in NON_EVALUABLE_EXECUTION_STATUSES


def _excluded_comparison_runs(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Describe runs blocked by inputs common to both architecture arms."""
    excluded = []
    for record in records:
        if _is_architecture_evaluable(record):
            continue
        excluded.append(
            {
                **_serialized_comparison_key(_comparison_key(record)),
                "execution_status": str(record.get("execution_status") or "unknown"),
            }
        )
    return excluded


def _median(summary: Mapping[str, Any], key: str) -> float | None:
    distribution = summary.get(key)
    if not isinstance(distribution, Mapping):
        return None
    value = distribution.get("median")
    return float(value) if value is not None else None


def _comparison_deltas(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, float | None]:
    metric_names = (
        "success_rate",
        "wall_time_seconds_median",
        "total_tokens_median",
        "tool_calls_per_run_median",
        "failed_tool_calls_per_100",
        "incorrect_tool_selection_runs_per_100",
        "estimated_cost_median",
    )
    if not left.get("runs") or not right.get("runs"):
        return dict.fromkeys(metric_names)

    def subtract(left_value: Any, right_value: Any) -> float | None:
        if left_value is None or right_value is None:
            return None
        return float(left_value) - float(right_value)

    return {
        "success_rate": subtract(left.get("success_rate"), right.get("success_rate")),
        "wall_time_seconds_median": subtract(
            _median(left, "wall_time_seconds"),
            _median(right, "wall_time_seconds"),
        ),
        "total_tokens_median": subtract(
            _median(left, "total_tokens"),
            _median(right, "total_tokens"),
        ),
        "tool_calls_per_run_median": subtract(
            _median(left, "tool_calls_per_run"),
            _median(right, "tool_calls_per_run"),
        ),
        "failed_tool_calls_per_100": subtract(
            left.get("failed_tool_calls_per_100"),
            right.get("failed_tool_calls_per_100"),
        ),
        "incorrect_tool_selection_runs_per_100": subtract(
            left.get("incorrect_tool_selection_runs_per_100"),
            right.get("incorrect_tool_selection_runs_per_100"),
        ),
        "estimated_cost_median": subtract(
            _median(left, "estimated_cost"),
            _median(right, "estimated_cost"),
        ),
    }


def build_system_comparison(
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    robustness_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a paired, publication-facing comparison of agentic architectures."""
    arms = list(records_by_arm)
    if len(arms) != 2:
        raise ValueError("System comparison requires exactly two arms")
    if set(arms) == {"team", "single_agent"}:
        # Keep publication tables and deltas stable even when execution order is
        # alternated to reduce temporal service bias.
        arms = ["team", "single_agent"]

    evaluable_records = {
        arm: [record for record in records_by_arm[arm] if _is_architecture_evaluable(record)]
        for arm in arms
    }
    excluded_runs = {arm: _excluded_comparison_runs(list(records_by_arm[arm])) for arm in arms}
    indexed = {arm: _index_comparison_records(evaluable_records[arm], arm=arm) for arm in arms}
    left_arm, right_arm = arms
    left_keys = set(indexed[left_arm])
    right_keys = set(indexed[right_arm])
    paired_keys = sorted(left_keys & right_keys)
    unmatched = {
        left_arm: [_serialized_comparison_key(key) for key in sorted(left_keys - right_keys)],
        right_arm: [_serialized_comparison_key(key) for key in sorted(right_keys - left_keys)],
    }
    warnings = []
    if unmatched[left_arm] or unmatched[right_arm]:
        warnings.append(
            "Some runs could not be paired by tier, case, prompt variant, and repetition; "
            "aggregate arm summaries include them, but paired outcomes do not."
        )
    if any(excluded_runs.values()):
        warnings.append(
            "Runs blocked by missing fixtures or prerequisites were not evaluated and are "
            "excluded from architecture success, efficiency, and pairing metrics."
        )

    per_arm = {arm: summarize_records(evaluable_records[arm]) for arm in arms}
    paired_per_arm = {
        arm: summarize_records([indexed[arm][key] for key in paired_keys]) for arm in arms
    }
    case_names = sorted(
        {
            str(record.get("case_name") or "unknown")
            for records in evaluable_records.values()
            for record in records
        }
    )
    per_case: Dict[str, Any] = {}
    for case_name in case_names:
        case_paired_keys = [key for key in paired_keys if key[1] == case_name]
        arm_summaries = {
            arm: summarize_records(
                [
                    record
                    for record in evaluable_records[arm]
                    if str(record.get("case_name") or "unknown") == case_name
                ]
            )
            for arm in arms
        }
        paired_arm_summaries = {
            arm: summarize_records([indexed[arm][key] for key in case_paired_keys]) for arm in arms
        }
        per_case[case_name] = {
            "per_arm": arm_summaries,
            "paired_runs": len(case_paired_keys),
            "delta": {
                "left_arm": left_arm,
                "right_arm": right_arm,
                **_comparison_deltas(
                    paired_arm_summaries[left_arm],
                    paired_arm_summaries[right_arm],
                ),
            },
        }

    paired_outcomes = {
        "both_success": 0,
        f"{left_arm}_only_success": 0,
        f"{right_arm}_only_success": 0,
        "both_failed": 0,
    }
    for key in paired_keys:
        left_success = bool(indexed[left_arm][key].get("task_success"))
        right_success = bool(indexed[right_arm][key].get("task_success"))
        if left_success and right_success:
            paired_outcomes["both_success"] += 1
        elif left_success:
            paired_outcomes[f"{left_arm}_only_success"] += 1
        elif right_success:
            paired_outcomes[f"{right_arm}_only_success"] += 1
        else:
            paired_outcomes["both_failed"] += 1

    secondary: Dict[str, Any] = {"per_arm": {}, "per_test": {}}
    if robustness_summaries:
        secondary["per_arm"] = {
            arm: {
                "tests": summary.get("total_tests", 0),
                "passed": summary.get("passed", 0),
                "pass_rate": summary.get("pass_rate", 0),
                "average_robustness_score": summary.get("average_robustness_score", 0),
                "overall_rating": summary.get("overall_rating", "N/A"),
            }
            for arm, summary in robustness_summaries.items()
        }
        test_names = sorted(
            {
                test_name
                for summary in robustness_summaries.values()
                for test_name in (summary.get("results") or {})
            }
        )
        secondary["per_test"] = {
            test_name: {
                arm: (
                    (robustness_summaries[arm].get("results") or {}).get(test_name, {}) or {}
                ).get("robustness_score")
                for arm in arms
            }
            for test_name in test_names
        }

    return {
        "schema_version": SYSTEM_COMPARISON_SCHEMA_VERSION,
        "arms": arms,
        "pairing": {
            "key_fields": ["tier", "case_name", "prompt_variant", "repetition"],
            "paired_runs": len(paired_keys),
            "unmatched_runs": unmatched,
            "excluded_runs": excluded_runs,
            "outcomes": paired_outcomes,
        },
        "per_arm": per_arm,
        "paired_per_arm": paired_per_arm,
        "delta": {
            "left_arm": left_arm,
            "right_arm": right_arm,
            **_comparison_deltas(
                paired_per_arm[left_arm],
                paired_per_arm[right_arm],
            ),
        },
        "per_case": per_case,
        "secondary_robustness": secondary,
        "warnings": warnings,
    }


def _git_value(project_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip()


def _package_versions(names: Sequence[str]) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _hardware_details() -> Dict[str, Any]:
    details: Dict[str, Any] = {"cpu_count": os.cpu_count()}
    try:
        import psutil

        details["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        details["memory_bytes"] = None
    try:
        import torch

        details["cuda_available"] = torch.cuda.is_available()
        details["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except Exception:
        details["cuda_available"] = None
        details["cuda_devices"] = []
    return details


def build_environment_manifest(
    *,
    project_root: Path,
    model_provider: str,
    model_id: str,
    config_path: Path,
    pricing: Mapping[str, Any] | None = None,
    inference_settings: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    status = _git_value(project_root, "status", "--short")
    env_names = (
        "USE_S3",
        "MODEL_PROVIDER",
        "MODEL_ID",
        "MODEL_MAX_TOKENS",
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "SYNPLANNER_MODEL_PATH",
        "PEPTIDE_DESIGNER_MODEL_PATH",
    )
    config_bytes = config_path.read_bytes() if config_path.exists() else b""
    return {
        "schema_version": RELIABILITY_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_value(project_root, "rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hardware": _hardware_details(),
        "model_provider": model_provider,
        "model_id": model_id,
        "inference_settings": dict(inference_settings or {}),
        "pricing": dict(pricing or {}),
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "environment_variables_present": {name: bool(os.environ.get(name)) for name in env_names},
        "packages": _package_versions(
            (
                "agno",
                "chemographykit",
                "chembl-webresource-client",
                "numpy",
                "pandas",
                "rdkit",
                "torch",
                "optuna",
            )
        ),
    }


def _markdown_report(summary: Mapping[str, Any]) -> str:
    interval = summary.get("wilson_95") or [0, 0]
    lines = [
        "# Manuscript Reliability Benchmark",
        "",
        "## Overall results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Runs | {summary.get('runs', 0)} |",
        f"| Successful | {summary.get('successful', 0)} |",
        f"| Success rate | {summary.get('success_rate', 0):.1%} |",
        f"| Wilson 95% CI | {interval[0]:.1%}–{interval[1]:.1%} |",
        f"| Tool calls | {summary.get('tool_calls', 0)} |",
        f"| Failed tool calls | {summary.get('failed_tool_calls', 0)} |",
        ("| Failed tool calls per 100 | " f"{summary.get('failed_tool_calls_per_100', 0):.2f} |"),
        (
            "| Runs with incorrect tool selection per 100 | "
            f"{summary.get('incorrect_tool_selection_runs_per_100', 0):.2f} |"
        ),
        f"| Total wall time (s) | {summary.get('wall_time_seconds_total', 0):.3f} |",
        f"| Total tokens | {summary.get('total_tokens_sum', 0)} |",
        "",
        "## Results by case",
        "",
        "| Case | Runs | Successful | Rate | Wilson 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    if summary.get("estimated_cost_total") is not None:
        results_header = lines.index("## Results by case")
        lines.insert(
            results_header - 1,
            f"| Total estimated cost | {summary['estimated_cost_total']:.6f} |",
        )
    for case_name, item in (summary.get("by_case") or {}).items():
        case_interval = item.get("wilson_95") or [0, 0]
        lines.append(
            f"| {case_name} | {item['runs']} | {item['successful']} | "
            f"{item['success_rate']:.1%} | "
            f"{case_interval[0]:.1%}–{case_interval[1]:.1%} |"
        )

    lines.extend(["", "## Runtime and usage", ""])
    for label, key in (
        ("Wall time (s)", "wall_time_seconds"),
        ("Total tokens", "total_tokens"),
        ("Tool calls per run", "tool_calls_per_run"),
        ("Estimated cost", "estimated_cost"),
    ):
        stats = summary.get(key) or {}
        median = stats.get("median")
        if median is None:
            continue
        lines.append(
            f"- {label}: median {median:.3f}; IQR "
            f"{stats.get('q1', 0):.3f}–{stats.get('q3', 0):.3f}"
        )

    repeatability = summary.get("repeatability") or {}
    if repeatability.get("groups_with_repeats"):
        lines.extend(["", "## Repeatability across identical prompts", ""])
        for label, key in (
            ("Task-outcome agreement", "task_outcome_agreement"),
            ("Exact tool-sequence agreement", "exact_tool_sequence_agreement"),
        ):
            stats = repeatability.get(key) or {}
            if stats.get("median") is not None:
                lines.append(
                    f"- {label}: median {stats['median']:.1%}; IQR "
                    f"{stats['q1']:.1%}–{stats['q3']:.1%}"
                )

    categories = summary.get("failure_categories") or {}
    if categories:
        lines.extend(["", "## Failure categories", ""])
        for category, count in sorted(categories.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {category}: {count}")
    return "\n".join(lines) + "\n"


def _format_distribution(summary: Mapping[str, Any], key: str, digits: int = 2) -> str:
    distribution = summary.get(key)
    if not isinstance(distribution, Mapping) or distribution.get("median") is None:
        return "n/a"
    template = f"{{:.{digits}f}}"
    return (
        f"{template.format(float(distribution['median']))} "
        f"({template.format(float(distribution['q1']))}–"
        f"{template.format(float(distribution['q3']))})"
    )


def _format_delta(value: Any, *, percent: bool = False, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    if percent:
        return f"{numeric:+.1%}"
    return f"{numeric:+.{digits}f}"


def _markdown_system_comparison(comparison: Mapping[str, Any]) -> str:
    arms = list(comparison.get("arms") or [])
    left_arm, right_arm = arms
    per_arm = comparison.get("per_arm") or {}
    pairing = comparison.get("pairing") or {}
    lines = [
        "# Multi-agent vs Single-agent Ablation",
        "",
        "Both arms use the same model configuration, prompts, fixtures, scientific "
        "tools, objective validators, and execution harness. The independent variable "
        "is the agentic structure: coordinator plus specialists versus one flat "
        "tool-calling agent.",
        "",
        "Objective task success is the primary outcome. Prompt-robustness similarity "
        "is reported separately as a secondary descriptive metric.",
        "",
        "Runs blocked before either architecture executes because a common fixture or "
        "prerequisite is unavailable are listed as not evaluated and excluded from "
        "success and efficiency metrics.",
        "",
        "## Overall objective results",
        "",
        "| Arm | Runs | Successful | Success rate (Wilson 95% CI) | "
        "Wall time median (IQR), s | Tokens median (IQR) | "
        "Tool calls median (IQR) | Failed calls / 100 | "
        "Incorrect selection runs / 100 | Cost median (IQR) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        summary = per_arm[arm]
        interval = summary.get("wilson_95") or [0, 0]
        lines.append(
            f"| {arm} | {summary.get('runs', 0)} | {summary.get('successful', 0)} | "
            f"{summary.get('success_rate', 0):.1%} "
            f"({interval[0]:.1%}–{interval[1]:.1%}) | "
            f"{_format_distribution(summary, 'wall_time_seconds')} | "
            f"{_format_distribution(summary, 'total_tokens', digits=0)} | "
            f"{_format_distribution(summary, 'tool_calls_per_run')} | "
            f"{summary.get('failed_tool_calls_per_100', 0):.2f} | "
            f"{summary.get('incorrect_tool_selection_runs_per_100', 0):.2f} | "
            f"{_format_distribution(summary, 'estimated_cost', digits=6)} |"
        )

    lines.extend(
        [
            "",
            "## Paired outcomes",
            "",
            (
                f"Runs are paired by tier, case, prompt variant, and repetition. "
                f"Matched pairs: {pairing.get('paired_runs', 0)}."
            ),
            "",
            "| Outcome | Pairs |",
            "|---|---:|",
        ]
    )
    for outcome, count in (pairing.get("outcomes") or {}).items():
        lines.append(f"| {outcome.replace('_', ' ')} | {count} |")

    excluded_runs = pairing.get("excluded_runs") or {}
    if any(excluded_runs.values()):
        lines.extend(
            [
                "",
                "## Not evaluated",
                "",
                "| Arm | Case | Status |",
                "|---|---|---|",
            ]
        )
        for arm in arms:
            for excluded in excluded_runs.get(arm) or []:
                lines.append(
                    f"| {arm} | {excluded.get('case_name', 'unknown')} | "
                    f"{excluded.get('execution_status', 'unknown')} |"
                )

    lines.extend(
        [
            "",
            f"## Paired overall deltas ({left_arm} − {right_arm})",
            "",
            "| Metric | Delta |",
            "|---|---:|",
        ]
    )
    delta = comparison.get("delta") or {}
    delta_rows = (
        ("Success rate", "success_rate", True, 2),
        ("Median wall time (s)", "wall_time_seconds_median", False, 2),
        ("Median total tokens", "total_tokens_median", False, 0),
        ("Median tool calls", "tool_calls_per_run_median", False, 2),
        ("Failed tool calls per 100", "failed_tool_calls_per_100", False, 2),
        (
            "Incorrect-selection runs per 100",
            "incorrect_tool_selection_runs_per_100",
            False,
            2,
        ),
        ("Median estimated cost", "estimated_cost_median", False, 6),
    )
    for label, key, percent, digits in delta_rows:
        lines.append(
            f"| {label} | {_format_delta(delta.get(key), percent=percent, digits=digits)} |"
        )

    lines.extend(
        [
            "",
            "## Results by representative case",
            "",
            "| Case | Arm | Runs | Success rate (Wilson 95% CI) | "
            "Wall time median (IQR), s | Tokens median (IQR) | Tool calls median (IQR) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case_name, case_result in (comparison.get("per_case") or {}).items():
        for arm in arms:
            summary = case_result["per_arm"][arm]
            interval = summary.get("wilson_95") or [0, 0]
            lines.append(
                f"| {case_name} | {arm} | {summary.get('runs', 0)} | "
                f"{summary.get('success_rate', 0):.1%} "
                f"({interval[0]:.1%}–{interval[1]:.1%}) | "
                f"{_format_distribution(summary, 'wall_time_seconds')} | "
                f"{_format_distribution(summary, 'total_tokens', digits=0)} | "
                f"{_format_distribution(summary, 'tool_calls_per_run')} |"
            )

    secondary = comparison.get("secondary_robustness") or {}
    if secondary.get("per_arm"):
        lines.extend(
            [
                "",
                "## Secondary prompt-robustness results",
                "",
                "| Arm | Tests | Passed | Pass rate | Average robustness score | Rating |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for arm in arms:
            summary = secondary["per_arm"][arm]
            lines.append(
                f"| {arm} | {summary.get('tests', 0)} | {summary.get('passed', 0)} | "
                f"{summary.get('pass_rate', 0):.1%} | "
                f"{summary.get('average_robustness_score', 0):.3f} | "
                f"{summary.get('overall_rating', 'N/A')} |"
            )

    warnings = list(comparison.get("warnings") or [])
    if warnings:
        lines.extend(["", "## Pairing warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "These results are descriptive. Interpret any architectural advantage "
            "together with confidence intervals, efficiency costs, and observed failure modes.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _write_human_review(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "review_id",
        "review_packet_path",
        "artifact_grounded_factuality_0_2",
        "task_fulfillment_0_2",
        "uncertainty_handling_0_2",
        "unsupported_claims_0_2",
        "reviewer_id",
        "notes",
    ]
    review_entries = []
    blinded_dir = path.parent / "human_review_packets"
    blinded_dir.mkdir(exist_ok=True)
    run_root = path.parent.parent
    for record in records:
        identity = (
            f"{record.get('benchmark_run_id')}:{record.get('run_id')}:" f"{record.get('case_name')}"
        )
        review_id = hashlib.sha256(identity.encode()).hexdigest()[:12]
        review_packet = None
        response_path = record.get("response_path")
        if response_path:
            source = run_root / str(response_path)
            if source.is_file():
                response_text = source.read_text(encoding="utf-8")
                artifact_references = sorted(
                    {
                        str(value)
                        for value in (record.get("generated_files") or {}).values()
                        if value
                    }
                )
                packet_lines = [
                    "# Blinded Review Packet",
                    "",
                    "## User prompt",
                    "",
                    str(record.get("prompt") or ""),
                    "",
                    "## System response",
                    "",
                    response_text,
                    "",
                    "## Artifact references",
                    "",
                ]
                packet_lines.extend(
                    f"- artifact_{index:03d}: {reference}"
                    for index, reference in enumerate(artifact_references, start=1)
                )
                destination = blinded_dir / f"{review_id}.md"
                destination.write_text("\n".join(packet_lines) + "\n", encoding="utf-8")
                review_packet = str(destination.relative_to(path.parent))
        review_entries.append(
            {
                "review_id": review_id,
                "review_packet_path": review_packet,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in sorted(review_entries, key=lambda item: item["review_id"]):
            writer.writerow(entry)


def save_reliability_bundle(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    *,
    environment_manifest: Mapping[str, Any],
) -> Dict[str, Any]:
    """Write normalized JSONL, summary, report, and blinded review sheet."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records_list = [dict(record) for record in records]
    summary = summarize_records(records_list)
    _write_jsonl(output_dir / "runs.jsonl", records_list)
    _write_jsonl(
        output_dir / "tool_calls.jsonl",
        (
            {
                "run_id": record.get("run_id"),
                "case_name": record.get("case_name"),
                **tool_call,
            }
            for record in records_list
            for tool_call in record.get("tool_calls") or []
        ),
    )
    _write_jsonl(
        output_dir / "validations.jsonl",
        (
            {
                "run_id": record.get("run_id"),
                "case_name": record.get("case_name"),
                **validation,
            }
            for record in records_list
            for validation in record.get("validations") or []
        ),
    )
    (output_dir / "reliability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    (output_dir / "reliability_report.md").write_text(_markdown_report(summary))
    (output_dir / "environment_manifest.json").write_text(
        json.dumps(environment_manifest, indent=2, sort_keys=True, default=str)
    )
    _write_human_review(output_dir / "human_review.csv", records_list)
    return summary


def save_system_comparison(
    output_dir: Path,
    records_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    robustness_summaries: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write paired JSON and manuscript-ready Markdown comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison = build_system_comparison(
        records_by_arm,
        robustness_summaries=robustness_summaries,
    )
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    comparison_md = output_dir / "comparison.md"
    comparison_md.write_text(
        _markdown_system_comparison(comparison),
        encoding="utf-8",
    )
    return comparison_md
