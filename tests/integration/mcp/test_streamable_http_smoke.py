"""Integration smoke: serve cs_copilot MCP over streamable HTTP.

Marked ``live`` because it spawns a subprocess and opens a localhost port.
Run with::

    uv run --extra mcp pytest tests/integration/mcp/test_streamable_http_smoke.py -m live
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

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


async def _assert_remote_capabilities(url: str, proc: asyncio.subprocess.Process) -> None:
    last_error: Exception | None = None
    for _attempt in range(30):
        if proc.returncode is not None:
            raise AssertionError(f"MCP server exited early with code {proc.returncode}")
        try:
            async with streamable_http_client(url) as (read, write, get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tools_by_name = {tool.name: tool for tool in tools.tools}
                    tool_names = set(tools_by_name)
                    assert "search" in tool_names
                    assert "fetch" in tool_names
                    assert "mcp_bootstrap" in tool_names
                    assert tools_by_name["search"].annotations.readOnlyHint is True
                    assert tools_by_name["fetch"].annotations.readOnlyHint is True
                    assert tools_by_name["mcp_bootstrap"].annotations.readOnlyHint is True
                    assert tools_by_name["chembl_fetch_compounds"].annotations.readOnlyHint is False

                    search_result = await session.call_tool(
                        "search",
                        {"query": "chembl fetch compounds"},
                    )
                    assert search_result.structuredContent
                    results = search_result.structuredContent["results"]
                    assert results

                    fetch_result = await session.call_tool(
                        "fetch",
                        {"id": results[0]["id"]},
                    )
                    assert fetch_result.structuredContent
                    assert fetch_result.structuredContent["id"] == results[0]["id"]
                    assert fetch_result.content

                    workflow_result = await session.call_tool(
                        "fetch",
                        {"id": "prompt:cs_copilot_mcp_workflow"},
                    )
                    workflow_payload = workflow_result.structuredContent
                    assert workflow_payload
                    assert workflow_payload["id"] == "prompt:cs_copilot_mcp_workflow"
                    assert "external MCP reasoner" in workflow_payload["text"]
                    assert get_session_id()
                    return
        except Exception as exc:  # noqa: BLE001 - retry until server is ready
            last_error = exc
            await asyncio.sleep(0.25)
    raise AssertionError(f"Timed out connecting to {url}: {last_error}")


async def _run_streamable_http_smoke(tmp_path: Path) -> None:
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
        "http-smoke",
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
        await _assert_remote_capabilities(f"http://127.0.0.1:{port}/mcp", proc)
    finally:
        stdout, stderr = await _terminate_process(proc)
        if proc.returncode not in (0, -15):
            raise AssertionError(
                f"MCP server exited with {proc.returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


def test_streamable_http_search_fetch(tmp_path: Path):
    asyncio.run(_run_streamable_http_smoke(tmp_path))
