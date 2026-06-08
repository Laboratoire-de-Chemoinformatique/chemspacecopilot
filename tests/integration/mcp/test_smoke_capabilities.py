"""Integration smoke: spawn cscopilot-mcp over stdio and list capabilities.

Marked ``live`` because it spawns a subprocess and depends on the optional
``[mcp]`` extra being installed. Skipped by default. Run with::

    uv run --extra mcp pytest tests/integration/mcp/ -m live
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

mcp = pytest.importorskip("mcp")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

pytestmark = pytest.mark.live


async def _list_capabilities() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "cs_copilot.mcp", "--session-id", "smoke", "--log-level", "warning"],
        env={**os.environ, "USE_S3": "false"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            initialize_result = await session.initialize()
            instructions = initialize_result.instructions or ""
            assert "external MCP client is the reasoning layer" in instructions
            assert "cs_copilot_workflow" in instructions
            assert len(instructions) <= 512

            tools = await session.list_tools()
            tool_names = {t.name for t in tools.tools}
            assert "search" in tool_names
            assert "fetch" in tool_names
            assert "chembl_fetch_compounds" in tool_names
            assert "gtm_optimization" in tool_names

            prompts = await session.list_prompts()
            prompt_names = {p.name for p in prompts.prompts}
            assert "cs_copilot_workflow" in prompt_names
            assert "chembl_retrieval_judge" in prompt_names

            resources = await session.list_resources()
            resource_uris = {str(r.uri) for r in resources.resources}
            assert "cscopilot://session/manifest.json" in resource_uris


def test_list_capabilities():
    asyncio.run(_list_capabilities())
