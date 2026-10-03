#!/usr/bin/env python
"""Benchmark workflow-run event appends against the configured storage backend.

Every tool call appends three to four events to its run's immutable event
ledger, so append latency must not grow with the length of the run. The script
creates a scratch run, appends observational ``tool_progress`` events, and
reports per-append latency. With ``--check`` it fails unless:

* p95 latency stays below ``--max-p95-ms`` (default: 20 ms locally, 150 ms on
  S3/MinIO), and
* the mean of the last 50 appends is below ``--max-growth`` (default 2x) times
  the mean of the first 50.

Run locally with ``USE_S3=false uv run python scripts/benchmark_run_ledger.py``,
or against the Compose MinIO service with ``USE_S3=true`` and the usual S3
environment variables.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--events", type=int, default=300, help="events to append")
    parser.add_argument("--check", action="store_true", help="enforce the latency gate")
    parser.add_argument("--max-p95-ms", type=float, default=None)
    parser.add_argument("--max-growth", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    use_s3 = os.environ.get("USE_S3", "false").strip().lower() in {"1", "true", "yes"}
    workdir = tempfile.mkdtemp(prefix="ledger-bench-")
    if not use_s3:
        os.chdir(workdir)
    session_id = f"ledger-bench-{uuid.uuid4().hex[:8]}"
    os.environ["SESSION_ID"] = session_id

    from cs_copilot.storage import S3
    from cs_copilot.workflows import RunContext

    S3.set_session_prefix(f"sessions/{session_id}")
    context = RunContext.create("mcp-session", run_id=f"bench-{uuid.uuid4().hex[:8]}")
    latencies_ms: list[float] = []
    for index in range(args.events):
        payload = {
            "runtime": "benchmark",
            "tool_name": "benchmark_tool",
            "stage": "started" if index % 2 == 0 else "completed",
            "span_id": f"span-{index // 2}",
            "attempt": 0,
            "max_attempts": 1,
        }
        started = time.perf_counter()
        context.append_event("tool_progress", payload)
        latencies_ms.append((time.perf_counter() - started) * 1000)

    ordered = sorted(latencies_ms)
    p50 = statistics.median(ordered)
    p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
    window = min(50, len(latencies_ms) // 2) or 1
    first = statistics.mean(latencies_ms[:window])
    last = statistics.mean(latencies_ms[-window:])
    growth = last / first if first else float("inf")
    backend = "s3" if use_s3 else f"local ({Path(workdir)})"
    print(f"backend: {backend}")
    print(f"events: {args.events}")
    print(f"p50: {p50:.2f} ms  p95: {p95:.2f} ms  max: {ordered[-1]:.2f} ms")
    print(
        f"mean first {window}: {first:.2f} ms  mean last {window}: {last:.2f} ms  growth: {growth:.2f}x"
    )

    if not args.check:
        return 0
    max_p95 = args.max_p95_ms if args.max_p95_ms is not None else (150.0 if use_s3 else 20.0)
    failures = []
    if p95 > max_p95:
        failures.append(f"p95 {p95:.2f} ms exceeds {max_p95:.2f} ms")
    if growth > args.max_growth:
        failures.append(f"latency grew {growth:.2f}x (limit {args.max_growth:.2f}x)")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
