"""The synchronous orchestrator used by thread-based runtimes."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from cs_copilot.execution.errors import ToolExecutionError
from cs_copilot.execution.runner import bind_sync_invoker, execute_sync
from cs_copilot.execution.tracing import current_tool_span_id
from cs_copilot.storage import S3

from ._kernel_helpers import make_spec, payloads, tool_events


class _Toolkit:
    def __init__(self) -> None:
        self.flaky_calls = 0
        self.calls = 0
        self.seen_boundary: str | None = None

    def echo(
        self,
        text: str,
        agent: Optional[Any] = None,
        session_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.calls += 1
        return text

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"x": [1, 2, 3]})

    def flaky(self) -> str:
        self.flaky_calls += 1
        if self.flaky_calls == 1:
            raise ConnectionError("temporary outage")
        return "recovered"

    def boom(self) -> str:
        raise RuntimeError("boom")

    def interrupted(self) -> str:
        raise KeyboardInterrupt

    def write_output(self, output_path: str) -> dict[str, Any]:
        self.calls += 1
        self.seen_boundary = S3.current_write_boundary()
        with S3.open(output_path, "w") as handle:
            handle.write("bounded")
        return {"output_path": output_path}

    async def coroutine(self) -> str:
        return "never"

    def own_span(self) -> str | None:
        return current_tool_span_id()


def _invoke(toolkit: _Toolkit, method: str):
    return bind_sync_invoker(make_spec(method, _Toolkit), getattr(toolkit, method))


def test_success_records_agno_events_and_keeps_raw_value(bound_context):
    ctx = bound_context("sync-success")
    toolkit = _Toolkit()

    outcome = execute_sync(
        make_spec("frame", _Toolkit),
        ctx,
        {},
        invoke=_invoke(toolkit, "frame"),
    )

    assert outcome.ok
    assert isinstance(outcome.value, pd.DataFrame)
    assert outcome.envelope["data"]["records"] == [{"x": 1}, {"x": 2}, {"x": 3}]
    assert tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "result_accepted"),
        ("tool_progress", "completed"),
        ("tool_call_recorded", "success"),
    ]
    assert {payload["runtime"] for payload in payloads(ctx, "tool_progress")} == {"agno"}
    [call] = payloads(ctx, "tool_call_recorded")
    assert call["runtime"] == "agno"
    assert call["workflow_slug"] == "agno-session"


def test_injector_receives_the_session_view(bound_context):
    ctx = bound_context("sync-inject")
    toolkit = _Toolkit()
    injected: list[dict[str, Any]] = []

    def inject(call_kwargs: dict[str, Any], session_view: dict[str, Any] | None) -> None:
        call_kwargs["session_state"] = ctx.session_state
        injected.append(dict(call_kwargs))

    outcome = execute_sync(
        make_spec("echo", _Toolkit),
        ctx,
        {"text": "hi"},
        invoke=_invoke(toolkit, "echo"),
        inject=inject,
    )

    assert outcome.value == "hi"
    assert injected == [{"text": "hi", "session_state": ctx.session_state}]


def test_retries_idempotent_failures(bound_context):
    ctx = bound_context("sync-retry")
    toolkit = _Toolkit()
    spec = make_spec("flaky", _Toolkit, idempotent=True, max_retries=1, retry_backoff_s=0)

    outcome = execute_sync(spec, ctx, {}, invoke=bind_sync_invoker(spec, toolkit.flaky))

    assert outcome.value == "recovered"
    assert outcome.envelope["metrics"]["attempts"] == 2
    assert tool_events(ctx)[:2] == [("tool_progress", "started"), ("tool_progress", "retrying")]


def test_failure_returns_error_envelope_and_exception(bound_context):
    ctx = bound_context("sync-failure")

    outcome = execute_sync(
        make_spec("boom", _Toolkit),
        ctx,
        {},
        invoke=_invoke(_Toolkit(), "boom"),
    )

    assert not outcome.ok
    assert isinstance(outcome.error, RuntimeError)
    assert outcome.envelope["error"]["message"] == "kernel_boom failed: boom"
    assert tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "failed"),
        ("tool_call_recorded", "error"),
    ]


def test_interrupt_closes_the_span(bound_context):
    ctx = bound_context("sync-interrupt")

    with pytest.raises(KeyboardInterrupt):
        execute_sync(
            make_spec("interrupted", _Toolkit),
            ctx,
            {},
            invoke=_invoke(_Toolkit(), "interrupted"),
        )

    assert tool_events(ctx) == [
        ("tool_progress", "started"),
        ("tool_progress", "cancelled"),
    ]
    assert ctx.run_context.pending_tool_invocations() == ()


def test_session_writes_are_rewritten_into_the_run(bound_context):
    ctx = bound_context("sync-write")
    toolkit = _Toolkit()
    spec = make_spec("write_output", _Toolkit, write_scope="session")

    outcome = execute_sync(
        spec,
        ctx,
        {"output_path": "out/result.txt"},
        invoke=bind_sync_invoker(spec, toolkit.write_output),
    )

    assert outcome.ok
    rewritten = outcome.value["output_path"]
    run_root = ctx.run_context.layout.run_root
    assert rewritten.startswith(f"{run_root}/")
    assert toolkit.seen_boundary == run_root
    with S3.open(rewritten, "r") as handle:
        assert handle.read() == "bounded"


@pytest.mark.parametrize("destination", ["/etc/passwd", "../escape.txt", "file:///tmp/x"])
def test_write_escapes_are_denied_before_execution(bound_context, destination):
    ctx = bound_context("sync-escape")
    toolkit = _Toolkit()
    spec = make_spec("write_output", _Toolkit, write_scope="session")

    outcome = execute_sync(
        spec,
        ctx,
        {"output_path": destination},
        invoke=bind_sync_invoker(spec, toolkit.write_output),
    )

    assert outcome.envelope["error"]["code"] == "permission_denied"
    assert toolkit.calls == 0


def test_confinement_propagates_through_to_thread(bound_context):
    ctx = bound_context("sync-to-thread")
    toolkit = _Toolkit()
    spec = make_spec("write_output", _Toolkit, write_scope="session")

    async def run_like_agno() -> Any:
        return await asyncio.to_thread(
            execute_sync,
            spec,
            ctx,
            {"output_path": "threaded.txt"},
            invoke=bind_sync_invoker(spec, toolkit.write_output),
        )

    outcome = asyncio.run(run_like_agno())

    assert outcome.ok
    assert toolkit.seen_boundary == ctx.run_context.layout.run_root


def test_plain_thread_with_copied_context(bound_context):
    ctx = bound_context("sync-plain-thread")
    toolkit = _Toolkit()
    spec = make_spec("write_output", _Toolkit, write_scope="session")
    outcomes: list[Any] = []
    context = contextvars.copy_context()

    def worker() -> None:
        outcomes.append(
            context.run(
                execute_sync,
                spec,
                ctx,
                {"output_path": "plain-thread.txt"},
                invoke=bind_sync_invoker(spec, toolkit.write_output),
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert outcomes[0].ok


def test_catalog_runs_require_a_running_workflow(bound_context):
    ctx = bound_context("sync-catalog", workflow_slug="chembl-to-gtm-report")
    toolkit = _Toolkit()

    outcome = execute_sync(
        make_spec("echo", _Toolkit),
        ctx,
        {"text": "denied"},
        invoke=_invoke(toolkit, "echo"),
    )

    assert outcome.envelope["error"]["code"] == "permission_denied"
    assert toolkit.calls == 0


def test_bind_sync_invoker_rejects_unsafe_bindings():
    toolkit = _Toolkit()
    with pytest.raises(TypeError, match="coroutine"):
        bind_sync_invoker(make_spec("coroutine", _Toolkit), toolkit.coroutine)
    with pytest.raises(ValueError, match="cannot safely cancel"):
        bind_sync_invoker(make_spec("echo", _Toolkit, timeout_s=1.0), toolkit.echo)


def test_nested_session_write_on_the_same_run_is_refused(bound_context):
    ctx = bound_context("sync-nested")
    inner_spec = make_spec("write_output", _Toolkit, write_scope="session")
    inner_toolkit = _Toolkit()
    inner: list[Any] = []

    class _Outer:
        def outer(self) -> str:
            inner.append(
                execute_sync(
                    inner_spec,
                    ctx,
                    {"output_path": "inner.txt"},
                    invoke=bind_sync_invoker(inner_spec, inner_toolkit.write_output),
                )
            )
            return "outer"

    outer_spec = make_spec("outer", _Outer, write_scope="session")
    outcome = execute_sync(
        outer_spec, ctx, {}, invoke=bind_sync_invoker(outer_spec, _Outer().outer)
    )

    assert outcome.ok
    assert inner[0].envelope["error"]["code"] == "internal"
    assert "re-enter" in inner[0].envelope["error"]["message"]
    assert isinstance(inner[0].error, ToolExecutionError)
    assert inner_toolkit.calls == 0


def test_a_running_tool_can_read_its_own_span_id(bound_context):
    ctx = bound_context("sync-own-span")
    toolkit = _Toolkit()

    outcome = execute_sync(
        make_spec("own_span", _Toolkit),
        ctx,
        {},
        invoke=_invoke(toolkit, "own_span"),
    )

    assert outcome.value == outcome.envelope["trace"]["span_id"]
    assert current_tool_span_id() is None
