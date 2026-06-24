"""Tests for the MCP tool adapter — schema stripping, injection, errors."""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import ToolSpec, build_tool
from cs_copilot.storage import S3, ensure_output_context


class _DummyToolkit:
    """Plain class used as a stand-in for an Agno-style toolkit instance."""

    def echo(
        self,
        text: str,
        count: int = 1,
        flag: bool = True,
        agent: Optional[Any] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Repeat *text* *count* times and mutate session state."""

        assert agent is not None and session_state is not None
        agent.session_state["last_text"] = text
        session_state["last_count"] = count
        return text * count

    def boom(self) -> str:
        raise RuntimeError("boom")

    def big_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"x": list(range(5))})

    def secret_echo(self, api_key: str, token_value: str) -> dict[str, str]:
        return {"status": "ok", "api_key": api_key, "token_value": token_value}


def _spec(method: str, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        mcp_name=f"dummy_{method}",
        toolkit_factory=lambda: _DummyToolkit(),
        method=method,
        summary=f"Dummy {method}",
        **kwargs,
    )


def test_adapter_strips_injected_params_from_public_signature():
    ctx = MCPAgentContext()
    tool = build_tool(_spec("echo"), _DummyToolkit(), ctx)
    sig = inspect.signature(tool)
    assert "agent" not in sig.parameters
    assert "session_state" not in sig.parameters
    assert "text" in sig.parameters
    assert "count" in sig.parameters


def test_adapter_injects_agent_and_session_state():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("echo"), instance, ctx)
    result = asyncio.run(tool(text="ab", count=3))
    assert result == "ababab"
    # State mutated via both the agent and session_state injection points.
    assert ctx.session_state["last_text"] == "ab"
    assert ctx.session_state["last_count"] == 3


def test_adapter_routes_worker_process_specs(monkeypatch):
    ctx = MCPAgentContext(session_state={"existing": True})
    instance = _DummyToolkit()
    calls: list[tuple[str, dict[str, Any], MCPAgentContext]] = []

    def fake_run_tool_job(spec: ToolSpec, kwargs: Dict[str, Any], job_ctx: MCPAgentContext):
        calls.append((spec.mcp_name, dict(kwargs), job_ctx))
        job_ctx.session_state["worker_ran"] = True
        return "from worker"

    import cs_copilot.mcp.jobs as jobs

    monkeypatch.setattr(jobs, "run_tool_job", fake_run_tool_job)
    spec = _spec("echo", run_in_worker_process=True, forces={"flag": False})
    tool = build_tool(spec, instance, ctx)
    result = asyncio.run(tool(text="ab", count=2))

    assert result == "from worker"
    assert calls == [(spec.mcp_name, {"text": "ab", "count": 2, "flag": False}, ctx)]
    assert ctx.session_state["worker_ran"] is True
    assert "last_text" not in ctx.session_state


def test_adapter_forces_override_kwargs_and_hides_them():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    spec = _spec("echo", forces={"flag": False})
    tool = build_tool(spec, instance, ctx)
    sig = inspect.signature(tool)
    assert "flag" not in sig.parameters


def test_adapter_wraps_exceptions_as_mcp_tool_error():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("boom"), instance, ctx)
    with pytest.raises(MCPToolError) as excinfo:
        asyncio.run(tool())
    assert "boom" in str(excinfo.value)


def test_adapter_coerces_dataframe_return():
    ctx = MCPAgentContext()
    instance = _DummyToolkit()
    tool = build_tool(_spec("big_dataframe"), instance, ctx)
    result = asyncio.run(tool())
    assert isinstance(result, dict)
    assert result["row_count"] == 5
    assert result["columns"] == ["x"]


def _manifest_payloads(tmp_path, session_name: str):
    manifest_root = (
        tmp_path
        / "data"
        / "sessions"
        / session_name
        / "workflows"
        / session_name
        / "manifests"
        / "mcp"
    )
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(manifest_root.glob("*.json"))
    ]


def test_adapter_writes_success_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-success"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    ensure_output_context(ctx.session_state, workflow_slug="smoke")
    tool = build_tool(_spec("echo"), _DummyToolkit(), ctx)

    asyncio.run(tool(text="ab", count=2))

    payloads = _manifest_payloads(tmp_path, session_name)
    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["runtime"] == "mcp"
    assert payload["tool_name"] == "dummy_echo"
    assert payload["status"] == "success"
    assert payload["public_args"]["text"] == "ab"
    assert payload["output_summary"]["type"] == "str"


def test_adapter_writes_error_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-error"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    ensure_output_context(ctx.session_state, workflow_slug="smoke")
    tool = build_tool(_spec("boom"), _DummyToolkit(), ctx)

    with pytest.raises(MCPToolError):
        asyncio.run(tool())

    payloads = _manifest_payloads(tmp_path, session_name)
    assert len(payloads) == 1
    assert payloads[0]["status"] == "error"
    assert "boom" in payloads[0]["error"]


def test_adapter_redacts_secret_manifest_args(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "manifest-redaction"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    ensure_output_context(ctx.session_state, workflow_slug="smoke")
    tool = build_tool(_spec("secret_echo"), _DummyToolkit(), ctx)

    asyncio.run(tool(api_key="secret", token_value="token"))

    payload = _manifest_payloads(tmp_path, session_name)[0]
    assert payload["public_args"]["api_key"] == "<redacted>"
    assert payload["public_args"]["token_value"] == "<redacted>"
