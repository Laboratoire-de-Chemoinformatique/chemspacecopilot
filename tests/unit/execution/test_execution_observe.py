"""Observe mode: record the kernel protocol without changing behaviour."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from cs_copilot.execution.runner import observe_sync
from cs_copilot.execution.scope import _InvocationScope
from cs_copilot.storage import S3

from ._kernel_helpers import make_spec, payloads, tool_events


class _Toolkit:
    def __init__(self) -> None:
        self.calls = 0
        self.received: dict[str, Any] = {}

    def write_output(self, output_path: str) -> dict[str, str]:
        self.calls += 1
        self.received = {"output_path": output_path}
        with S3.open(output_path, "w") as handle:
            handle.write("observed")
        return {"output_path": output_path}

    def frame(self) -> pd.DataFrame:
        self.calls += 1
        return pd.DataFrame({"x": [1, 2]})

    def boom(self) -> str:
        self.calls += 1
        raise RuntimeError("boom")

    def interrupted(self) -> str:
        raise KeyboardInterrupt


def _call(toolkit: _Toolkit, method: str):
    return lambda arguments: getattr(toolkit, method)(**arguments)


def test_observe_records_the_protocol_and_registers_new_files(bound_context):
    ctx = bound_context("observe-success")
    toolkit = _Toolkit()
    relative = f"{ctx.run_context.layout.run_root}/results/out.csv"

    value = observe_sync(
        make_spec("write_output", _Toolkit, write_scope="session"),
        ctx,
        {"output_path": relative},
        invoke=_call(toolkit, "write_output"),
        extra={"owner": {"pid": 1}},
    )

    assert value == {"output_path": relative}
    assert toolkit.calls == 1
    assert tool_events(ctx) == [
        ("tool_progress", "started"),
        ("artifact_registered", None),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    started = payloads(ctx, "tool_progress")[0]
    assert started["runtime"] == "agno"
    assert started["owner"] == {"pid": 1}
    [artifact] = ctx.run_context.run.artifacts.values()
    assert artifact.relative_path == "results/out.csv"
    assert artifact.producer_tool == "kernel_write_output"
    assert artifact.provenance["registration"] == "automatic"
    [call] = payloads(ctx, "tool_call_recorded")
    audit = call["execution_audit"]
    assert audit["mode"] == "observe"
    assert audit["writes"] == [
        {
            "path": f"{ctx.run_context.layout.run_root}/results/out.csv",
            "verdict": "create_new",
            "created": True,
            "mode": "w",
        }
    ]


def test_observe_audits_instead_of_rewriting_or_denying(bound_context):
    ctx = bound_context("observe-audit")
    toolkit = _Toolkit()

    observe_sync(
        make_spec("write_output", _Toolkit, write_scope="session"),
        ctx,
        {"output_path": "session-root.csv"},
        invoke=_call(toolkit, "write_output"),
    )

    # The call ran with its original argument and wrote where it asked to.
    assert toolkit.received == {"output_path": "session-root.csv"}
    with S3.open("session-root.csv", "r") as handle:
        assert handle.read() == "observed"
    [call] = payloads(ctx, "tool_call_recorded")
    audit = call["execution_audit"]
    assert audit["would_rewrite"]["output_path"].startswith(ctx.run_context.layout.run_root)
    assert audit["writes"][0]["verdict"] == "outside_boundary"
    assert ctx.run_context.run.artifacts == {}

    observe_sync(
        make_spec("write_output", _Toolkit, write_scope="session"),
        ctx,
        {"output_path": "../escape.csv"},
        invoke=lambda arguments: {"output_path": arguments["output_path"]},
    )
    denied = payloads(ctx, "tool_call_recorded")[-1]["execution_audit"]
    assert "would_deny_writes" in denied


def test_observe_returns_raw_values(bound_context):
    ctx = bound_context("observe-raw")

    value = observe_sync(make_spec("frame", _Toolkit), ctx, {}, invoke=_call(_Toolkit(), "frame"))

    assert isinstance(value, pd.DataFrame)


def test_observe_propagates_tool_failures(bound_context):
    ctx = bound_context("observe-failure")
    toolkit = _Toolkit()

    with pytest.raises(RuntimeError, match="boom"):
        observe_sync(make_spec("boom", _Toolkit), ctx, {}, invoke=_call(toolkit, "boom"))

    assert toolkit.calls == 1
    assert tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "failed"),
        ("tool_call_recorded", "error"),
    ]


def test_observe_closes_the_span_on_interrupt(bound_context):
    ctx = bound_context("observe-interrupt")

    with pytest.raises(KeyboardInterrupt):
        observe_sync(
            make_spec("interrupted", _Toolkit),
            ctx,
            {},
            invoke=_call(_Toolkit(), "interrupted"),
        )

    assert tool_events(ctx) == [("tool_progress", "started"), ("tool_progress", "cancelled")]
    assert ctx.run_context.pending_tool_invocations() == ()


def test_kernel_failures_never_block_or_repeat_the_call(bound_context, monkeypatch):
    ctx = bound_context("observe-kernel-failure")
    toolkit = _Toolkit()

    def broken_append(*args: Any, **kwargs: Any) -> Any:
        raise OSError("ledger unavailable")

    monkeypatch.setattr(ctx.run_context, "append_event", broken_append)

    value = observe_sync(make_spec("frame", _Toolkit), ctx, {}, invoke=_call(toolkit, "frame"))

    assert toolkit.calls == 1
    assert isinstance(value, pd.DataFrame)


def test_observe_uses_the_supplied_attribution(bound_context):
    ctx = bound_context("observe-scope")
    scope = _InvocationScope(
        run_id=ctx.run_context.run.run_id,
        task_id=None,
        role="gtm_agent",
        profile="gtm-analysis",
    )

    observe_sync(
        make_spec("frame", _Toolkit), ctx, {}, invoke=_call(_Toolkit(), "frame"), scope=scope
    )

    assert {(p["role"], p["profile"]) for p in payloads(ctx, "tool_progress")} == {
        ("gtm_agent", "gtm-analysis")
    }
