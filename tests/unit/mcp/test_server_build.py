"""End-to-end smoke test for FastMCP server assembly.

Builds the server in-process (no stdio) and asserts that every registered
tool, prompt, and the resource manifest are present.
"""

from __future__ import annotations

import asyncio
import os

import pytest

mcp = pytest.importorskip("mcp")

from cs_copilot.mcp.session import (  # noqa: E402  (import-after-skip)
    BootstrapConfig,
    apply_session_id,
    bootstrap,
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    work = tmp_path_factory.mktemp("mcp-server")
    os.chdir(work)
    apply_session_id("mcp-server-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-server-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    return build_server(ctx)


def test_tools_registered(server):
    names = sorted(t.name for t in server._tool_manager.list_tools())
    assert "chembl_fetch_compounds" in names
    assert "gtm_optimization" in names
    assert "report_save_markdown" in names


def test_prompts_registered(server):
    names = sorted(p.name for p in server._prompt_manager.list_prompts())
    assert "chemspace_workflow" in names
    assert "chembl_retrieval_judge" in names


def test_resources_include_manifest(server):
    entries = asyncio.run(server.list_resources())
    uris = {str(e.uri) for e in entries}
    assert "cscopilot://session/manifest.json" in uris
