"""Tests for subprocess-backed MCP job execution."""

from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPErrorCode, MCPToolError
from cs_copilot.mcp.jobs import DeferredToolJob, run_tool_job
from cs_copilot.mcp.tool_adapter import ToolSpec
from cs_copilot.storage import S3


def _spec(**kwargs: Any) -> ToolSpec:
    return ToolSpec(
        mcp_name="chembl_fetch_compounds",
        toolkit_factory=lambda: object(),
        method="fetch_compounds",
        summary="fetch",
        run_in_worker_process=True,
        **kwargs,
    )


class _FakePopen:
    returncode = 0
    pid = 12345

    def __init__(self, cmd, **_kwargs):
        self.cmd = cmd
        self.result_path = Path(cmd[cmd.index("--result") + 1])
        self.job_path = Path(cmd[cmd.index("--job") + 1])

    def communicate(self, timeout=None):
        job = json.loads(self.job_path.read_text(encoding="utf-8"))
        self.result_path.write_text(
            json.dumps(
                {
                    "ok": True,
                    "result": f"ran {job['tool_name']}",
                    "session_state": {"worker": True, "source": job["session_state"]["source"]},
                }
            ),
            encoding="utf-8",
        )
        return "", "worker stderr"


def test_run_tool_job_merges_worker_session_state(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    S3.set_session_prefix("sessions/job-test")
    ctx = MCPAgentContext(session_state={"source": "parent"})

    result = run_tool_job(_spec(), {"query": "CDK2"}, ctx)

    assert result == "ran chembl_fetch_compounds"
    assert ctx.session_state == {"worker": True, "source": "parent"}


def test_run_tool_job_defers_worker_state_until_parent_acceptance(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _FakePopen)
    S3.set_session_prefix("sessions/job-deferred")
    ctx = MCPAgentContext(session_state={"source": "parent"})

    with S3.confine_writes("workflows/run-1"):
        outcome = run_tool_job(
            _spec(),
            {"query": "CDK2"},
            ctx,
            defer_commit=True,
        )
        assert isinstance(outcome, DeferredToolJob)
        assert ctx.session_state == {"source": "parent"}
        outcome.accept()

    assert outcome.result == "ran chembl_fetch_compounds"
    assert ctx.session_state == {"worker": True, "source": "parent"}


@pytest.mark.parametrize("boundary", ["", "workflows/run-1"])
def test_run_tool_job_propagates_active_write_boundary(monkeypatch, boundary):
    captured: dict[str, Any] = {}

    class BoundaryPopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            captured.update(job)
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "bounded",
                        "session_state": job["session_state"],
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", BoundaryPopen)
    ctx = MCPAgentContext(session_state={"source": "parent"})
    protected = f"{boundary}/immutable.txt" if boundary else "immutable.txt"

    with S3.confine_writes(boundary, protected_paths=(protected,)):
        assert run_tool_job(_spec(), {"query": "CDK2"}, ctx) == "bounded"

    assert captured["write_boundary"] == boundary
    assert captured["write_protected_paths"] == [protected]


def test_concurrent_worker_jobs_preserve_disjoint_state_changes(monkeypatch):
    rendezvous = threading.Barrier(2)

    class ConcurrentPopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            rendezvous.wait(timeout=5)
            worker_state = dict(job["session_state"])
            worker_state[f"worker_{job['kwargs']['query']}"] = True
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": job["kwargs"]["query"],
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ConcurrentPopen)
    ctx = MCPAgentContext(session_state={"source": "parent"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run_tool_job, _spec(), {"query": query}, ctx) for query in ("CDK2", "EGFR")
        ]
        results = {future.result(timeout=10) for future in futures}

    assert results == {"CDK2", "EGFR"}
    assert ctx.session_state == {
        "source": "parent",
        "worker_CDK2": True,
        "worker_EGFR": True,
    }


def test_worker_merge_conflict_fails_atomically_and_preserves_parent(monkeypatch):
    ctx = MCPAgentContext(session_state={"shared": "base", "parent_only": "before"})

    class ConflictingPopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            ctx.session_state["shared"] = "parent"
            worker_state = dict(job["session_state"])
            worker_state.update({"shared": "worker", "worker_only": True})
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "unused",
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ConflictingPopen)

    with pytest.raises(
        MCPToolError,
        match=(
            r"worker session-state merge conflicted with concurrent parent updates "
            r"for keys: shared\. Parent state was preserved\."
        ),
    ) as raised:
        run_tool_job(_spec(), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False
    assert ctx.session_state == {"shared": "parent", "parent_only": "before"}


def test_worker_deletion_is_applied_without_clobbering_parent_addition(monkeypatch):
    ctx = MCPAgentContext(session_state={"remove_me": "base", "keep": True})

    class DeletingPopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            ctx.session_state["parent_added"] = True
            worker_state = dict(job["session_state"])
            del worker_state["remove_me"]
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "deleted",
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", DeletingPopen)

    assert run_tool_job(_spec(), {"query": "CDK2"}, ctx) == "deleted"
    assert ctx.session_state == {"keep": True, "parent_added": True}


def test_worker_deletion_conflict_preserves_parent_value(monkeypatch):
    ctx = MCPAgentContext(session_state={"remove_me": "base"})

    class ConflictingDeletePopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            ctx.session_state["remove_me"] = "parent"
            worker_state = dict(job["session_state"])
            del worker_state["remove_me"]
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "unused",
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ConflictingDeletePopen)

    with pytest.raises(MCPToolError, match="keys: remove_me") as raised:
        run_tool_job(_spec(idempotent=True), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is True
    assert ctx.session_state == {"remove_me": "parent"}


@pytest.mark.parametrize(
    ("key", "before", "after"),
    [
        (
            "output_context",
            {"session_id": "session-1", "run_id": "run-1", "workflow_slug": "pilot"},
            {"session_id": "session-1", "run_id": "run-2", "workflow_slug": "pilot"},
        ),
        ("active_task_id", "retrieve", "analyze"),
        ("active_task_attempt", 1, 2),
        ("active_handoff_id", "handoff-1", "handoff-2"),
    ],
)
def test_execution_scope_change_rejects_stale_worker_result(
    monkeypatch,
    key,
    before,
    after,
):
    ctx = MCPAgentContext(session_state={key: before, "source": "parent"})

    class StalePopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            ctx.session_state[key] = after
            worker_state = dict(job["session_state"])
            worker_state["worker_only"] = True
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "stale",
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", StalePopen)

    with pytest.raises(MCPToolError, match=rf"keys: {key}") as raised:
        run_tool_job(_spec(idempotent=True), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False
    assert ctx.session_state == {key: after, "source": "parent"}


@pytest.mark.parametrize(
    ("key", "before", "worker_value"),
    [
        (
            "output_context",
            {"session_id": "session-1", "run_id": "run-1", "workflow_slug": "pilot"},
            {"session_id": "session-1", "run_id": "run-2", "workflow_slug": "pilot"},
        ),
        ("active_task_id", "retrieve", "analyze"),
        ("active_role", "chembl_downloader", "gtm_agent"),
        ("active_profile", "chembl-retrieval", "gtm-analysis"),
        ("active_task_attempt", 1, 2),
        ("active_handoff_id", "handoff-1", "handoff-2"),
        ("mcp_profile", "standard", "bootstrap"),
    ],
)
def test_worker_cannot_change_authoritative_execution_scope(
    monkeypatch,
    key,
    before,
    worker_value,
):
    ctx = MCPAgentContext(session_state={key: before, "source": "parent"})

    class ScopeChangingPopen(_FakePopen):
        def communicate(self, timeout=None):
            job = json.loads(self.job_path.read_text(encoding="utf-8"))
            worker_state = dict(job["session_state"])
            worker_state.update({key: worker_value, "worker_only": True})
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": True,
                        "result": "unauthorized",
                        "session_state": worker_state,
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ScopeChangingPopen)

    with pytest.raises(MCPToolError, match=f"keys: {key}") as raised:
        run_tool_job(_spec(idempotent=True), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False
    assert ctx.session_state == {key: before, "source": "parent"}


def test_run_tool_job_raises_worker_error(monkeypatch):
    class ErrorPopen(_FakePopen):
        def communicate(self, timeout=None):
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "worker exploded",
                        "traceback": "trace",
                        "session_state": {"partial": True},
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ErrorPopen)
    ctx = MCPAgentContext(session_state={"source": "parent"})

    with pytest.raises(MCPToolError, match="worker exploded") as raised:
        run_tool_job(_spec(), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False
    assert ctx.session_state == {"source": "parent"}


def test_run_tool_job_propagates_worker_error_metadata(monkeypatch):
    class ErrorPopen(_FakePopen):
        def communicate(self, timeout=None):
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "upstream unavailable",
                        "error_code": "transient_external",
                        "retryable": True,
                        "session_state": {},
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ErrorPopen)

    with pytest.raises(MCPToolError, match="upstream unavailable") as raised:
        run_tool_job(_spec(), {"query": "CDK2"}, MCPAgentContext(session_state={}))

    assert raised.value.code == MCPErrorCode.TRANSIENT_EXTERNAL.value
    assert raised.value.retryable is True


@pytest.mark.parametrize(
    ("error_code", "retryable"),
    [("unknown_code", True), ("timeout", "yes")],
)
def test_run_tool_job_rejects_malformed_worker_error_metadata(monkeypatch, error_code, retryable):
    class ErrorPopen(_FakePopen):
        def communicate(self, timeout=None):
            self.result_path.write_text(
                json.dumps(
                    {
                        "ok": False,
                        "error": "bad metadata",
                        "error_code": error_code,
                        "retryable": retryable,
                        "session_state": {},
                    }
                ),
                encoding="utf-8",
            )
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", ErrorPopen)

    with pytest.raises(MCPToolError, match="bad metadata") as raised:
        run_tool_job(_spec(), {}, MCPAgentContext(session_state={}))

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False


def test_run_tool_job_classifies_malformed_worker_result_as_internal(monkeypatch):
    class MalformedPopen(_FakePopen):
        def communicate(self, timeout=None):
            self.result_path.write_text("{not-json", encoding="utf-8")
            return "", ""

    monkeypatch.setattr(subprocess, "Popen", MalformedPopen)

    with pytest.raises(MCPToolError, match="malformed JSON") as raised:
        run_tool_job(_spec(), {}, MCPAgentContext(session_state={}))

    assert raised.value.code == MCPErrorCode.INTERNAL.value
    assert raised.value.retryable is False


def test_run_tool_job_classifies_nonserializable_input_as_invalid(monkeypatch):
    def unexpected_popen(*_args, **_kwargs):
        raise AssertionError("worker must not start for an invalid payload")

    monkeypatch.setattr(subprocess, "Popen", unexpected_popen)

    with pytest.raises(MCPToolError, match="not JSON-serializable") as raised:
        run_tool_job(_spec(), {"query": object()}, MCPAgentContext(session_state={}))

    assert raised.value.code == MCPErrorCode.INVALID_INPUT.value
    assert raised.value.retryable is False


def test_run_tool_job_classifies_serialization_exhaustion_as_resource_limit(monkeypatch):
    def out_of_memory(*_args, **_kwargs):
        raise MemoryError

    monkeypatch.setattr("cs_copilot.mcp.jobs.json.dump", out_of_memory)

    with pytest.raises(MCPToolError, match="serialization resource") as raised:
        run_tool_job(_spec(), {}, MCPAgentContext(session_state={}))

    assert raised.value.code == MCPErrorCode.RESOURCE_LIMIT.value
    assert raised.value.retryable is False


def test_run_tool_job_terminates_timed_out_worker(monkeypatch):
    killed: list[int] = []

    class TimeoutPopen(_FakePopen):
        def communicate(self, timeout=None):
            if not killed:
                raise subprocess.TimeoutExpired(self.cmd, timeout=timeout)
            return "", "still running"

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", TimeoutPopen)
    monkeypatch.setattr("cs_copilot.mcp.jobs.os.killpg", lambda pid, sig: killed.append(sig))
    ctx = MCPAgentContext(session_state={"source": "parent"})

    with pytest.raises(MCPToolError, match="timed out") as raised:
        run_tool_job(_spec(worker_timeout_s=0.01), {"query": "CDK2"}, ctx)

    assert raised.value.code == MCPErrorCode.TIMEOUT.value
    assert raised.value.retryable is True
    assert killed
