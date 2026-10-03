"""The MCP adapter and the sync orchestrator produce the same protocol."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from cs_copilot.execution.context import MCP_RUNTIME
from cs_copilot.execution.runner import bind_sync_invoker, execute_sync
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.tool_adapter import build_tool

from ._kernel_helpers import make_spec, payloads, tool_events


class _Toolkit:
    def structured(self) -> dict[str, Any]:
        return {"can_proceed": True}

    def boom(self) -> str:
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    ("method", "spec_kwargs"),
    [
        ("structured", {}),
        ("structured", {"result_artifact_type": "analysis_result"}),
        ("boom", {}),
    ],
)
def test_mcp_and_sync_paths_emit_the_same_protocol(bound_context, method, spec_kwargs):
    spec = make_spec(method, _Toolkit, **spec_kwargs)

    mcp_bound = bound_context(
        f"parity-mcp-{method}", workflow_slug="mcp-session", runtime=MCP_RUNTIME
    )
    mcp_ctx = MCPAgentContext(session_state=mcp_bound.session_state)
    mcp_ctx.run_context = mcp_bound.run_context
    mcp_envelope = asyncio.run(build_tool(spec, _Toolkit(), mcp_ctx)())

    sync_ctx = bound_context(
        f"parity-sync-{method}", workflow_slug="mcp-session", runtime=MCP_RUNTIME
    )
    sync_outcome = execute_sync(
        spec,
        sync_ctx,
        {},
        invoke=bind_sync_invoker(spec, getattr(_Toolkit(), method)),
    )

    assert tool_events(mcp_ctx) == tool_events(sync_ctx)
    assert set(mcp_envelope) == set(sync_outcome.envelope)
    assert mcp_envelope["status"] == sync_outcome.envelope["status"]
    assert mcp_envelope["error"] == sync_outcome.envelope["error"]
    for event_type in ("tool_progress", "tool_call_recorded"):
        assert [set(p) for p in payloads(mcp_ctx, event_type)] == [
            set(p) for p in payloads(sync_ctx, event_type)
        ]
