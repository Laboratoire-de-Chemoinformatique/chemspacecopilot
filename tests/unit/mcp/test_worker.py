"""Focused tests for the MCP worker result contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from cs_copilot.mcp import worker
from cs_copilot.storage import S3


def test_worker_main_serializes_normalized_retry_metadata(monkeypatch):
    written: dict[str, Any] = {}

    monkeypatch.setattr(worker, "_load_dotenv", lambda: None)
    monkeypatch.setattr(
        worker,
        "_read_json",
        lambda _path: {"tool_name": "chembl_fetch_compounds"},
    )

    def fail(_payload):
        raise ConnectionError("temporary upstream outage")

    monkeypatch.setattr(worker, "run_payload", fail)
    monkeypatch.setattr(worker, "_write_json", lambda _path, payload: written.update(payload))

    result = worker.main(["--job", "job.json", "--result", "result.json"])

    assert result == 0
    assert written["ok"] is False
    assert written["error_code"] == "transient_external"
    assert written["retryable"] is True
    assert "chembl_fetch_compounds failed" in written["error"]


@pytest.mark.parametrize(
    ("include_boundary", "boundary", "expected"),
    [
        (True, "", ""),
        (True, "workflows/run-1", "workflows/run-1"),
        (True, None, None),
        (False, None, None),
    ],
)
def test_worker_activates_only_string_write_boundaries(
    monkeypatch,
    include_boundary,
    boundary,
    expected,
):
    observed: list[str | None] = []
    ctx = SimpleNamespace(session_state={})

    def dispatch(_kwargs, _ctx):
        observed.append(S3.current_write_boundary())
        return "done"

    monkeypatch.setattr(worker, "_build_context", lambda _payload: ctx)
    monkeypatch.setitem(worker._DISPATCH, "boundary_probe", dispatch)
    payload: dict[str, Any] = {
        "tool_name": "boundary_probe",
        "kwargs": {},
        "session_state": {},
    }
    if include_boundary:
        payload["write_boundary"] = boundary

    result = worker.run_payload(payload)

    assert result == {
        "ok": True,
        "result": "done",
        "session_state": {},
        "staged_publications": {},
    }
    assert observed == [expected]
    assert S3.current_write_boundary() is None


def test_worker_activates_and_restores_registered_artifact_protection(monkeypatch):
    observed: list[frozenset[str]] = []
    ctx = SimpleNamespace(session_state={})

    def dispatch(_kwargs, _ctx):
        observed.append(S3.current_write_protected_paths())
        return "done"

    monkeypatch.setattr(worker, "_build_context", lambda _payload: ctx)
    monkeypatch.setitem(worker._DISPATCH, "protection_probe", dispatch)
    payload = {
        "tool_name": "protection_probe",
        "kwargs": {},
        "session_state": {},
        "write_boundary": "workflows/run-1",
        "write_protected_paths": ["workflows/run-1/inputs/request.json"],
    }

    result = worker.run_payload(payload)

    assert result == {
        "ok": True,
        "result": "done",
        "session_state": {},
        "staged_publications": {},
    }
    assert observed == [frozenset({"workflows/run-1/inputs/request.json"})]
    assert S3.current_write_protected_paths() == frozenset()
