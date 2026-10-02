"""Execute predeclared reliability batches in plan order and retain every outcome.

This driver performs no planning and no analysis. It runs the exact commands
recorded in ``study_plan.json``, in the recorded order, under the recorded
environment, and appends one immutable ledger entry per attempt.

A non-zero exit code from the runner is a measurement, not a driver error: the
benchmark exits 1 whenever any case fails. Attempts are never retried and never
substituted, so a failed or interrupted batch stays in the record exactly as it
occurred.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_ledger(ledger_path: Path, entry: dict) -> None:
    """Append one JSON line and flush it to disk before returning."""
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_batch(batch: dict, plan_env: dict, log_dir: Path, ledger_path: Path) -> int:
    batch_id = batch["batch_id"]
    log_path = log_dir / f"{batch_id}.log"
    if log_path.exists():
        raise SystemExit(
            f"refusing to overwrite an existing attempt log: {log_path}\n"
            "Retained evidence is immutable; move it aside deliberately instead."
        )

    env = os.environ.copy()
    env.update(plan_env)

    started = time.perf_counter()
    entry = {
        "batch_id": batch_id,
        "tier": batch["tier"],
        "system": batch["system"],
        "arm_order": batch["arm_order"],
        "global_prompt_variant": batch["global_prompt_variant"],
        "global_repetition": batch["global_repetition"],
        "expected_executions": batch["expected_executions"],
        "config_path": batch["config_path"],
        "config_sha256": batch["config_sha256"],
        "command": batch["command"],
        "log_path": str(log_path),
        "started_utc": _utc(),
    }

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# {batch_id} started {entry['started_utc']}\n")
        log.write(f"# command: {json.dumps(batch['command'])}\n\n")
        log.flush()
        completed = subprocess.run(
            batch["command"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(Path(batch["command"][1]).parent.parent.parent),
            check=False,
        )

    entry["finished_utc"] = _utc()
    entry["wall_time_seconds"] = time.perf_counter() - started
    entry["exit_code"] = completed.returncode
    # The benchmark exits 1 when any case fails; that is retained data.
    entry["exit_meaning"] = {
        0: "all cases passed",
        1: "completed with one or more failing cases",
        130: "interrupted",
    }.get(completed.returncode, "unexpected exit; inspect the log before continuing")
    _append_ledger(ledger_path, entry)

    print(
        f"{batch_id}: exit={completed.returncode} "
        f"({entry['exit_meaning']}) in {entry['wall_time_seconds']:.1f}s",
        flush=True,
    )
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="Path to the frozen study_plan.json")
    parser.add_argument(
        "--batch",
        action="append",
        dest="batches",
        help="Run only this batch id; repeatable. Default: every batch in plan order.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for attempt logs and the ledger (default: <plan dir>/execution).",
    )
    parser.add_argument(
        "--stop-on-unexpected-exit",
        action="store_true",
        help="Halt the study when a batch exits with a code other than 0, 1 or 130.",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    plan_env = dict(plan["environment"])

    log_dir = args.log_dir or args.plan.parent / "execution"
    log_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = log_dir / "execution_ledger.jsonl"

    selected = plan["batches"]
    if args.batches:
        wanted = list(args.batches)
        known = {batch["batch_id"] for batch in selected}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise SystemExit(f"unknown batch id(s): {', '.join(unknown)}")
        selected = [batch for batch in selected if batch["batch_id"] in wanted]

    if "DEEPSEEK_API_KEY" not in os.environ:
        # _get_model() falls back to an interactive prompt, which would hang.
        raise SystemExit("DEEPSEEK_API_KEY is not set; the runner would block on input.")

    print(f"executing {len(selected)} batch(es); ledger: {ledger_path}", flush=True)
    for batch in selected:
        code = run_batch(batch, plan_env, log_dir, ledger_path)
        if code not in (0, 1) and args.stop_on_unexpected_exit:
            print(
                f"halting: {batch['batch_id']} exited {code}; "
                "remaining batches are unattempted, not failed.",
                flush=True,
            )
            sys.exit(code)


if __name__ == "__main__":
    main()
