"""Tests for the MCP tool adapter — schema stripping, injection, errors."""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import ToolSpec, build_tool


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
