"""Tests for subprocess-backed MCP job execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.jobs import run_tool_job
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

    with pytest.raises(MCPToolError, match="worker exploded"):
        run_tool_job(_spec(), {"query": "CDK2"}, ctx)

    assert ctx.session_state == {"source": "parent"}


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

    with pytest.raises(MCPToolError, match="timed out"):
        run_tool_job(_spec(worker_timeout_s=0.01), {"query": "CDK2"}, ctx)

    assert killed
