"""Pin the durable tool-event protocol emitted by the MCP tool adapter.

``cs_copilot.workflows.runtime`` consumes these ``tool_progress`` stages and
their ordering (pending-span detection, automatic artifact registration,
lifecycle transition guards), and replay/evaluation reads the
``tool_call_recorded`` payloads. The execution kernel shared with the Agno
runtime must reproduce this protocol exactly, so it is pinned here
independently of the adapter's internals.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.tool_adapter import ToolSpec, build_tool
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext

ENVELOPE_KEYS = {
    "schema_version",
    "status",
    "data",
    "artifact_ids",
    "warnings",
    "error",
    "metrics",
    "trace",
}
METRIC_KEYS = {"duration_ms", "cached", "attempts", "retries", "output_bytes"}
TRACE_KEYS = {"run_id", "trace_id", "span_id", "parent_span_id"}
PROGRESS_KEYS = {
    "runtime",
    "session_id",
    "run_id",
    "workflow_slug",
    "trace_id",
    "span_id",
    "parent_span_id",
    "tool_name",
    "task_id",
    "role",
    "profile",
    "stage",
    "attempt",
    "max_attempts",
    "cached",
    "task_attempt",
    "handoff_id",
}
CALL_KEYS = {
    "runtime",
    "session_id",
    "run_id",
    "workflow_slug",
    "trace_id",
    "span_id",
    "parent_span_id",
    "tool_name",
    "task_id",
    "role",
    "profile",
    "status",
    "duration_ms",
    "attempts",
    "retries",
    "cached",
    "public_args",
    "forced_args",
    "output_summary",
}


class _ProtocolToolkit:
    def __init__(self) -> None:
        self.flaky_calls = 0
        self.started = asyncio.Event()

    def echo(
        self,
        text: str,
        agent: Optional[Any] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        return text

    def structured(self) -> dict[str, Any]:
        return {"can_proceed": True, "plan": ["retrieve", "analyze"]}

    def flaky(self) -> str:
        self.flaky_calls += 1
        if self.flaky_calls == 1:
            raise ConnectionError("temporary outage")
        return "recovered"

    def boom(self) -> str:
        raise RuntimeError("boom")

    async def wait_forever(self) -> str:
        self.started.set()
        await asyncio.Event().wait()
        return "unreachable"


def _spec(method: str, **kwargs: Any) -> ToolSpec:
    return ToolSpec(
        mcp_name=f"protocol_{method}",
        toolkit_factory=_ProtocolToolkit,
        method=method,
        summary=f"Protocol {method}",
        **kwargs,
    )


def _bound_ctx(tmp_path, monkeypatch, name: str, *, workflow_slug: str = "mcp-session"):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix(f"sessions/{name}")
    ctx = MCPAgentContext()
    ctx.run_context = RunContext.create(
        workflow_slug,
        session_state=ctx.session_state,
        run_id=name,
    )
    return ctx


def _tool_events(ctx: MCPAgentContext) -> list[tuple[str, str | None]]:
    """Return ``(event_type, stage|status)`` pairs after run creation."""

    pairs = []
    for event in ctx.run_context.events:
        if event.event_type == "tool_progress":
            pairs.append((event.event_type, event.payload["stage"]))
        elif event.event_type == "tool_call_recorded":
            pairs.append((event.event_type, event.payload["status"]))
        elif event.event_type == "artifact_registered":
            pairs.append((event.event_type, None))
    return pairs


def _payloads(ctx: MCPAgentContext, event_type: str) -> list[dict[str, Any]]:
    return [event.payload for event in ctx.run_context.events if event.event_type == event_type]


def _assert_envelope_shape(envelope: dict[str, Any]) -> None:
    assert set(envelope) == ENVELOPE_KEYS
    assert set(envelope["metrics"]) == METRIC_KEYS
    assert set(envelope["trace"]) == TRACE_KEYS
    assert envelope["schema_version"] == 2


def _assert_one_span(ctx: MCPAgentContext, envelope: dict[str, Any]) -> None:
    span_ids = {
        payload["span_id"]
        for event_type in ("tool_progress", "tool_call_recorded")
        for payload in _payloads(ctx, event_type)
    }
    assert span_ids == {envelope["trace"]["span_id"]}


def test_success_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-success")
    tool = build_tool(_spec("echo"), _ProtocolToolkit(), ctx)

    envelope = asyncio.run(tool(text="ok"))

    assert envelope["status"] == "success"
    _assert_envelope_shape(envelope)
    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    for payload in _payloads(ctx, "tool_progress"):
        assert set(payload) == PROGRESS_KEYS
        assert payload["runtime"] == "mcp"
        assert payload["tool_name"] == "protocol_echo"
    [call] = _payloads(ctx, "tool_call_recorded")
    assert set(call) == CALL_KEYS
    assert call["runtime"] == "mcp"
    assert call["public_args"] == {"text": "ok"}
    assert call["attempts"] == 1
    _assert_one_span(ctx, envelope)


def test_result_artifact_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-artifact")
    tool = build_tool(
        _spec("structured", result_artifact_type="analysis_result"),
        _ProtocolToolkit(),
        ctx,
    )

    envelope = asyncio.run(tool())

    assert envelope["status"] == "success"
    assert len(envelope["artifact_ids"]) == 1
    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("artifact_registered", None),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    [registered] = _payloads(ctx, "artifact_registered")
    started_span = _payloads(ctx, "tool_progress")[0]["span_id"]
    artifact = registered["artifact"]
    assert artifact["artifact_id"] == envelope["artifact_ids"][0]
    assert artifact["artifact_type"] == "analysis_result"
    assert artifact["producer_tool"] == "protocol_structured"
    assert artifact["provenance"]["registration"] == "automatic"
    assert artifact["provenance"]["invocation_span_id"] == started_span
    _assert_one_span(ctx, envelope)


def test_retry_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-retry")
    tool = build_tool(
        _spec("flaky", idempotent=True, max_retries=1, retry_backoff_s=0),
        _ProtocolToolkit(),
        ctx,
    )

    envelope = asyncio.run(tool())

    assert envelope["status"] == "success"
    assert envelope["metrics"]["attempts"] == 2
    assert envelope["metrics"]["retries"] == 1
    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "retrying"),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    retrying = _payloads(ctx, "tool_progress")[1]
    assert retrying["attempt"] == 1
    assert retrying["max_attempts"] == 2
    assert "retrying in" in retrying["message"]


def test_cache_hit_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-cache")
    tool = build_tool(_spec("structured", idempotent=True), _ProtocolToolkit(), ctx)

    first = asyncio.run(tool(idempotency_key="plan-1"))
    events_before = len(ctx.run_context.events)
    second = asyncio.run(tool(idempotency_key="plan-1"))

    assert first["status"] == second["status"] == "success"
    assert second["metrics"]["cached"] is True
    assert (second["metrics"]["attempts"], second["metrics"]["retries"]) == (0, 0)
    assert second["data"] == first["data"]
    assert second["trace"]["span_id"] != first["trace"]["span_id"]
    new_events = ctx.run_context.events[events_before:]
    assert [
        (event.event_type, event.payload.get("stage") or event.payload.get("status"))
        for event in new_events
    ] == [
        ("tool_progress", "started"),
        ("tool_progress", "cache_hit"),
        ("tool_call_recorded", "success"),
    ]
    cache_hit = new_events[1].payload
    assert cache_hit["attempt"] == 0
    assert cache_hit["cached"] is True
    call = new_events[2].payload
    assert call["idempotency_fingerprint"]
    assert call["idempotency_fingerprint"] != "plan-1"
    assert call["public_args"]["idempotency_key"] == call["idempotency_fingerprint"]


def test_authorization_failure_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(
        tmp_path,
        monkeypatch,
        "protocol-denied",
        workflow_slug="chembl-to-gtm-report",
    )
    tool = build_tool(_spec("echo"), _ProtocolToolkit(), ctx)

    envelope = asyncio.run(tool(text="denied"))

    assert envelope["status"] == "error"
    assert envelope["error"]["code"] == "permission_denied"
    assert envelope["metrics"]["attempts"] == 0
    _assert_envelope_shape(envelope)
    assert _tool_events(ctx) == [
        ("tool_progress", "failed"),
        ("tool_call_recorded", "error"),
    ]


def test_failure_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-failure")
    tool = build_tool(_spec("boom"), _ProtocolToolkit(), ctx)

    envelope = asyncio.run(tool())

    assert envelope["status"] == "error"
    assert envelope["error"] == {
        "code": "internal",
        "message": "protocol_boom failed: boom",
        "retryable": False,
    }
    assert envelope["metrics"]["attempts"] == 1
    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "failed"),
        ("tool_call_recorded", "error"),
    ]
    [call] = _payloads(ctx, "tool_call_recorded")
    assert call["error"] == "protocol_boom failed: boom"
    _assert_one_span(ctx, envelope)


def test_cancellation_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-cancel")
    instance = _ProtocolToolkit()
    tool = build_tool(_spec("wait_forever"), instance, ctx)

    async def exercise() -> None:
        task = asyncio.create_task(tool())
        await instance.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(exercise())

    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "cancelled"),
    ]
    cancelled = _payloads(ctx, "tool_progress")[1]
    assert cancelled["message"] == "client cancelled the tool invocation"
    assert cancelled["attempt"] == 1


def test_worker_protocol(tmp_path, monkeypatch):
    ctx = _bound_ctx(tmp_path, monkeypatch, "protocol-worker")
    import cs_copilot.mcp.jobs as jobs

    def fake_run_tool_job(spec, kwargs, job_ctx, *, defer_commit=False, **_: Any):
        assert defer_commit is True
        return jobs.DeferredToolJob(
            result="from worker",
            ctx=job_ctx,
            base_session_state=dict(job_ctx.session_state),
            worker_session_state=dict(job_ctx.session_state),
            tool_name=spec.mcp_name,
            retryable=spec.idempotent,
            publications={},
            write_boundary=None,
            staging_id=None,
        )

    monkeypatch.setattr(jobs, "run_tool_job", fake_run_tool_job)
    tool = build_tool(_spec("echo", run_in_worker_process=True), _ProtocolToolkit(), ctx)

    envelope = asyncio.run(tool(text="worker"))

    assert envelope["status"] == "success"
    assert envelope["data"] == "from worker"
    assert _tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    _assert_one_span(ctx, envelope)
