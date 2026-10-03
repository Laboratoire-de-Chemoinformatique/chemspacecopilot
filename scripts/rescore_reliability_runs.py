"""Re-score retained benchmark runs with the current validators.

Validators read saved artifacts, so a measurement defect in them can be
corrected without re-running a single model call. This replays the retained
runs through the current validator code and reports, per check, where the
stored verdict and the corrected verdict disagree.

It never calls a provider, never mutates the study tree, and never invents a
run: a disagreement here means the earlier verdict was a measurement artifact,
not that the system behaved differently.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "robustness"))

from reliability.validators import evaluate_run  # noqa: E402


def _validator_map(study_dir: Path) -> dict[str, str]:
    """Map case name -> validator name, reading the frozen batch configs."""
    import yaml

    mapping: dict[str, str] = {}
    for config in sorted((study_dir / "configs").glob("*.yaml")):
        data = yaml.safe_load(config.read_text(encoding="utf-8"))
        for name, test in (data.get("tests") or {}).items():
            if test.get("validator"):
                mapping[name] = test["validator"]
            for step in test.get("steps") or []:
                if step.get("validator"):
                    mapping[step.get("name") or name] = step["validator"]
    return mapping


def _load_output(run_dir: Path) -> dict | None:
    """Rebuild the dict the validators were originally given."""
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    output = json.loads(metadata_path.read_text(encoding="utf-8"))

    response = run_dir / "response.txt"
    output["response"] = (
        response.read_text(encoding="utf-8", errors="replace") if response.exists() else ""
    )

    state = run_dir / "session_state.json"
    output["session_state"] = {}
    if state.exists():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
        except ValueError:
            saved = {}
        # The artifact wraps the state under a "session_state" key; passing the
        # wrapper through would make every path lookup silently resolve to None.
        if isinstance(saved, dict):
            inner = saved.get("session_state")
            output["session_state"] = inner if isinstance(inner, dict) else saved
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_dir", type=Path, help="Directory holding configs/ and runs/")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the report JSON")
    args = parser.parse_args()

    validators = _validator_map(args.study_dir)
    rows: list[dict] = []
    flips: Counter = Counter()
    per_check: dict[str, Counter] = defaultdict(Counter)

    for run_dir in sorted(args.study_dir.glob("runs/*/*/*/run_*")):
        case = run_dir.parent.name
        validator = validators.get(case)
        if validator is None:
            continue
        output = _load_output(run_dir)
        if output is None:
            continue

        stored = output.get("validation") or {}
        stored_checks = {c["name"]: c["passed"] for c in stored.get("checks") or []}
        stored_success = bool(stored.get("task_success"))

        try:
            rescored = evaluate_run(validator, output)
        except Exception as exc:  # a validator crash is itself a finding
            rows.append(
                {"run": str(run_dir), "case": case, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue

        rescored_checks = {c["name"]: c["passed"] for c in rescored.get("checks") or []}
        rescored_success = bool(rescored.get("task_success"))

        for name, passed in rescored_checks.items():
            before = stored_checks.get(name)
            if before is not None and before != passed:
                per_check[name]["recovered" if passed else "lost"] += 1

        flips[(stored_success, rescored_success)] += 1
        rows.append(
            {
                "run": str(run_dir),
                "batch": run_dir.parents[2].name,
                "arm": run_dir.parents[1].name.split("_")[-1],
                "case": case,
                "validator": validator,
                "stored_task_success": stored_success,
                "rescored_task_success": rescored_success,
                "changed": stored_success != rescored_success,
                "newly_failing_checks": sorted(
                    n for n, p in rescored_checks.items() if not p and stored_checks.get(n) is True
                ),
                "newly_passing_checks": sorted(
                    n for n, p in rescored_checks.items() if p and stored_checks.get(n) is False
                ),
            }
        )

    scored = [r for r in rows if "error" not in r]
    report = {
        "purpose": "Retained runs re-scored with current validators; no model calls",
        "runs_rescored": len(scored),
        "stored_task_success": sum(1 for r in scored if r["stored_task_success"]),
        "rescored_task_success": sum(1 for r in scored if r["rescored_task_success"]),
        "verdict_transitions": {
            f"{'pass' if a else 'fail'}->{'pass' if b else 'fail'}": n
            for (a, b), n in flips.items()
        },
        "per_check_changes": {k: dict(v) for k, v in sorted(per_check.items())},
        "validator_errors": [r for r in rows if "error" in r],
        "runs": rows,
        "limitations": [
            "Re-scoring corrects how evidence was judged; it does not change what the agent did.",
            "Runs that crashed before producing artifacts cannot be recovered by re-scoring.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "runs_rescored",
                    "stored_task_success",
                    "rescored_task_success",
                    "verdict_transitions",
                    "per_check_changes",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
