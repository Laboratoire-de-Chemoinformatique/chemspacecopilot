#!/usr/bin/env python
"""Summarize what enforce mode would change for recorded in-process chats.

In observe mode (``CS_COPILOT_AGNO_EXECUTION=observe``) every team tool call
is recorded with an ``execution_audit`` block describing how the kernel's
boundaries would have rewritten or denied it and how each write would be
classified by confined, create-only writes. This script aggregates those
blocks per tool, so a deployment can check a new toolkit or workflow in
observe mode before running it under ``enforce`` (the default).

Usage::

    uv run python scripts/agno_execution_audit.py --session <session-id> [--run <run-id>]
    uv run python scripts/agno_execution_audit.py --session <session-id> --json

Without ``--run`` every ``agno-session`` run of the session is summarized.
Storage follows the usual configuration (``USE_S3`` and friends); local runs
are read from ``./data/sessions/<session-id>`` relative to the working
directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any, Iterable

# Verdicts that enforce mode accepts; anything else would be denied.
ALLOWED_WRITE_VERDICTS = frozenset({"create_new"})


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="chat / storage session id")
    parser.add_argument("--run", action="append", default=[], help="run id (repeatable)")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    return parser.parse_args(argv)


def _session_run_ids(session_id: str) -> list[str]:
    import fsspec

    from cs_copilot.storage import S3, get_s3_config, is_s3_enabled

    url = S3.path("workflows")
    options = get_s3_config().to_storage_options() if is_s3_enabled() else {}
    filesystem, path = fsspec.core.url_to_fs(url, **options)
    try:
        entries = filesystem.ls(path, detail=False)
    except FileNotFoundError:
        return []
    return sorted(PurePosixPath(str(entry)).name for entry in entries)


def summarize(events: Iterable[Any]) -> dict[str, Any]:
    """Aggregate ``execution_audit`` blocks of ``tool_call_recorded`` events."""

    tools: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "calls": 0,
            "errors": 0,
            "write_verdicts": Counter(),
            "would_rewrite": 0,
            "would_deny_writes": Counter(),
            "would_deny_reads": Counter(),
            "registration_problems": 0,
            "kernel_errors": 0,
            "examples": [],
        }
    )
    for event in events:
        if event.event_type != "tool_call_recorded":
            continue
        payload = event.payload
        audit = payload.get("execution_audit")
        if not isinstance(audit, dict):
            continue
        entry = tools[str(payload.get("tool_name"))]
        entry["calls"] += 1
        entry["errors"] += int(payload.get("status") == "error")
        for write in audit.get("writes") or []:
            verdict = str(write.get("verdict"))
            entry["write_verdicts"][verdict] += 1
            if verdict not in ALLOWED_WRITE_VERDICTS and len(entry["examples"]) < 5:
                entry["examples"].append(f"write {verdict}: {write.get('path')}")
        if audit.get("would_rewrite"):
            entry["would_rewrite"] += 1
        for key in ("would_deny_writes", "would_deny_reads"):
            if audit.get(key):
                message = str(audit[key])
                entry[key][message[:200]] += 1
        entry["registration_problems"] += len(audit.get("registration_problems") or [])
        entry["kernel_errors"] += len(audit.get("kernel_errors") or [])

    report: dict[str, Any] = {}
    for tool_name, entry in sorted(tools.items()):
        denied_writes = sum(
            count
            for verdict, count in entry["write_verdicts"].items()
            if verdict not in ALLOWED_WRITE_VERDICTS
        )
        report[tool_name] = {
            **{key: value for key, value in entry.items() if not isinstance(value, Counter)},
            "write_verdicts": dict(entry["write_verdicts"]),
            "would_deny_writes": dict(entry["would_deny_writes"]),
            "would_deny_reads": dict(entry["would_deny_reads"]),
            "enforce_conflicts": denied_writes
            + sum(entry["would_deny_writes"].values())
            + sum(entry["would_deny_reads"].values()),
        }
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from cs_copilot.storage import S3
    from cs_copilot.workflows import RunContext

    S3.set_session_prefix(f"sessions/{args.session}")
    run_ids = args.run or _session_run_ids(args.session)
    runs: dict[str, Any] = {}
    for run_id in run_ids:
        try:
            context = RunContext.load(run_id, session_id=args.session)
        except Exception as exc:  # noqa: BLE001 - report and continue
            runs[run_id] = {"error": str(exc)}
            continue
        if not args.run and context.run.workflow_slug != "agno-session":
            continue
        runs[run_id] = summarize(context.events)

    if args.json:
        json.dump(runs, sys.stdout, indent=2, sort_keys=True)
        print()
        return 0
    if not runs:
        print(f"No agno-session runs found in session {args.session!r}.")
        return 0
    for run_id, report in runs.items():
        print(f"run {run_id}")
        if "error" in report:
            print(f"  could not load: {report['error']}")
            continue
        conflicts = 0
        for tool_name, entry in report.items():
            conflicts += entry["enforce_conflicts"]
            marker = "!" if entry["enforce_conflicts"] else " "
            print(
                f" {marker} {tool_name}: {entry['calls']} call(s), {entry['errors']} error(s), "
                f"writes {entry['write_verdicts'] or '{}'}"
            )
            for key in ("would_deny_writes", "would_deny_reads"):
                for message, count in entry[key].items():
                    print(f"      {key} x{count}: {message}")
            for example in entry["examples"]:
                print(f"      {example}")
        print(f"  enforce conflicts: {conflicts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
