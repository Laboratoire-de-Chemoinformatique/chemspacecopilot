"""Integration smoke for the remote MCP readiness checker."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[3]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _terminate_process(proc: asyncio.subprocess.Process) -> tuple[str, str]:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:  # pragma: no cover - defensive cleanup
            proc.kill()
            await proc.wait()
    stdout, stderr = await proc.communicate()
    return stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _run_existing_url_check(tmp_path: Path) -> None:
    port = _free_port()
    env = {
        **os.environ,
        "PYTHONPATH": f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
        "USE_S3": "false",
        "AGNO_TELEMETRY": "false",
    }
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "cs_copilot.mcp",
        "--transport",
        "streamable-http",
        "--session-id",
        "readiness-existing-url",
        "--workflow-slug",
        "smoke",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "error",
        cwd=tmp_path,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        from cs_copilot.mcp.check import _parse_args, run_check

        report = await run_check(
            _parse_args(["--url", f"http://127.0.0.1:{port}/mcp", "--timeout", "30"])
        )

        assert report.mode == "existing-url"
        assert report.endpoint_url == f"http://127.0.0.1:{port}/mcp"
        assert report.tool_count >= 5
        assert report.annotated_tool_count == report.tool_count
        assert report.read_only_tool_count >= 1
        assert report.write_tool_count >= 1
        assert 0 < report.instructions_length <= 512
        assert report.prompt_count >= 1
        assert report.workflow_prompt_id == "prompt:cs_copilot_workflow"
        assert report.fetch_id == "tool:chembl_fetch_compounds"
    finally:
        stdout, stderr = await _terminate_process(proc)
        if proc.returncode not in (0, -15):
            raise AssertionError(
                f"MCP server exited with {proc.returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


def test_readiness_check_entrypoint(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )

    from cs_copilot.mcp.check import main

    result = main(["--port", "0", "--timeout", "30"])

    captured = capsys.readouterr()
    assert result == 0
    assert "cs_copilot MCP readiness check passed" in captured.out
    assert "server_instructions: ok" in captured.out
    assert "tool_annotations: ok" in captured.out
    assert "workflow_prompt: ok (prompt:cs_copilot_workflow)" in captured.out
    assert "search_fetch: ok" in captured.out
    assert "chatgpt_connector_name: cs_copilot" in captured.out
    assert "chatgpt_smoke_prompt:" in captured.out
    assert "chatgpt_connector_name: cs_copilot" in captured.out
    assert "chatgpt_smoke_prompt:" in captured.out
    assert "endpoint_url: http://127.0.0.1:" in captured.out




def test_readiness_check_json_output(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )

    from cs_copilot.mcp.check import main

    result = main(["--port", "0", "--timeout", "30", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert result == 0
    assert payload["status"] == "passed"
    assert payload["mode"] == "temporary-server"
    assert payload["auth"] == "none"
    assert payload["tool_count"] >= 5
    assert payload["annotated_tool_count"] == payload["tool_count"]
    assert payload["workflow_prompt_id"] == "prompt:cs_copilot_workflow"
    assert payload["fetch_id"] == "tool:chembl_fetch_compounds"
    assert payload["chatgpt_connector_name"] == "cs_copilot"
    assert "fetch prompt:cs_copilot_workflow" in payload["chatgpt_smoke_prompt"]
    assert "chembl_fetch_compounds" in "\n".join(payload["chatgpt_expected_evidence"])


def test_readiness_check_entrypoint_with_bearer_auth(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(
        "PYTHONPATH",
        f"{ROOT / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}",
    )
    monkeypatch.setenv("CS_COPILOT_MCP_AUTH_TOKEN", "readiness-secret")

    from cs_copilot.mcp.check import main

    result = main(["--port", "0", "--timeout", "30", "--auth-scope", "mcp:read"])

    captured = capsys.readouterr()
    assert result == 0
    assert "cs_copilot MCP readiness check passed" in captured.out
    assert "auth: bearer-token" in captured.out
    assert "server_instructions: ok" in captured.out
    assert "tool_annotations: ok" in captured.out
    assert "workflow_prompt: ok (prompt:cs_copilot_workflow)" in captured.out
    assert "search_fetch: ok" in captured.out


def test_readiness_check_existing_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    asyncio.run(_run_existing_url_check(tmp_path))
