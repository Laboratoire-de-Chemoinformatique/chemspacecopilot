"""Incremental event-stream synchronization of workflow runs.

Every tool call appends three to four events, so an append must not re-list
and replay the whole stream. These tests pin the I/O budget of an append and
the replaced/missing-stream guarantees the full re-read used to provide.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cs_copilot.storage import S3
from cs_copilot.workflows import EventReplayError, RunContext, RunStatus
from cs_copilot.workflows import runtime as runtime_module


@pytest.fixture
def ledger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/ledger-sync")
    context = RunContext.create("mcp-session", run_id="ledger-run")
    return context, Path(S3.path(context.layout.events_rel_path))


def _observe(context: RunContext, index: int):
    # tool_call_recorded is observational and opens no tool span, so it never
    # blocks run or task transitions.
    return context.append_event(
        "tool_call_recorded",
        {"tool_name": "ledger_probe", "status": "success", "span_id": f"s{index}"},
    )


def _counting(monkeypatch, name: str) -> list[int]:
    calls: list[int] = []
    original = getattr(runtime_module, name)

    def wrapper(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime_module, name, wrapper)
    return calls


def test_append_300_has_a_constant_io_budget(ledger, monkeypatch):
    context, _ = ledger
    for index in range(298):
        _observe(context, index)
    assert len(context.events) == 299

    listings = _counting(monkeypatch, "_list_event_paths")
    reads = _counting(monkeypatch, "_read_event_path")
    applied = _counting(monkeypatch, "_apply_event")
    snapshot_writes = _counting(monkeypatch, "_write_json")

    event = _observe(context, 299)

    assert event.sequence == 300
    assert listings == []
    assert len(reads) <= 3
    assert len(applied) <= 2
    assert snapshot_writes == []


def test_state_changes_still_refresh_snapshots(ledger):
    context, events_dir = ledger
    for index in range(3):
        _observe(context, index)

    context.transition_run(RunStatus.PLANNING)

    manifest = json.loads((events_dir.parent / "manifest.json").read_text())
    assert manifest["status"] == RunStatus.PLANNING.value
    assert manifest["event_count"] == len(context.events) == 5


def test_a_stale_writer_catches_up_from_another_writer(ledger):
    first, _ = ledger
    second = RunContext.load("ledger-run", session_id="ledger-sync")
    _observe(first, 1)
    first.transition_run(RunStatus.PLANNING)

    event = _observe(second, 2)

    assert event.sequence == 4
    assert [item.event_id for item in second.events] == [
        f"{sequence:08d}" for sequence in range(1, 5)
    ]
    assert second.run.status is RunStatus.PLANNING
    first.refresh()
    assert len(first.events) == 4


def test_a_replaced_tail_is_rejected(ledger):
    context, events_dir = ledger
    _observe(context, 1)
    tail = events_dir / f"{context.events[-1].event_id}.jsonl"
    record = json.loads(tail.read_text())
    record["payload"]["status"] = "tampered"
    tail.write_text(json.dumps(record) + "\n")

    with pytest.raises(EventReplayError, match="replaced"):
        _observe(context, 2)


def test_a_disappeared_stream_is_rejected(ledger):
    context, events_dir = ledger
    _observe(context, 1)
    shutil.rmtree(events_dir)

    with pytest.raises(EventReplayError, match="disappeared"):
        _observe(context, 2)


def test_periodic_listing_check_detects_a_missing_known_event(ledger):
    context, events_dir = ledger
    for index in range(3):
        _observe(context, index)
    (events_dir / "00000002.jsonl").unlink()
    context._syncs_since_full = runtime_module._STREAM_CHECK_INTERVAL

    with pytest.raises(EventReplayError, match="not contiguous"):
        _observe(context, 4)


def test_refresh_is_incremental(ledger, monkeypatch):
    context, _ = ledger
    for index in range(10):
        _observe(context, index)
    listings = _counting(monkeypatch, "_list_event_paths")

    context.refresh()

    assert listings == []
