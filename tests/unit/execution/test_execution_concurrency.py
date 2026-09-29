"""Idempotency and per-run write serialization across loops and threads."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from cs_copilot.execution.context import MCP_RUNTIME
from cs_copilot.execution.errors import ToolExecutionError
from cs_copilot.execution.idempotency import (
    _idempotency_mutex,
    _IdempotencyReservation,
    _wait_for_idempotent_owner,
)
from cs_copilot.execution.locks import _async_run_write_locks, _sync_run_write_lock
from cs_copilot.execution.runner import bind_sync_invoker, execute_sync
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.tool_adapter import build_tool

from ._kernel_helpers import make_spec, tool_events


class _Toolkit:
    def __init__(self) -> None:
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.intervals: list[tuple[float, float]] = []
        self.lock = threading.Lock()

    def plan(self) -> dict[str, Any]:
        self.calls += 1
        return {"plan": ["retrieve"], "call": self.calls}

    def blocking_plan(self) -> dict[str, Any]:
        self.calls += 1
        self.started.set()
        assert self.release.wait(5)
        return {"plan": ["retrieve"], "call": self.calls}

    def slow_write(self, output_path: str) -> str:
        begin = time.perf_counter()
        time.sleep(0.05)
        with self.lock:
            self.intervals.append((begin, time.perf_counter()))
        return output_path


def test_idempotency_store_survives_separate_event_loops(bound_context):
    ctx = bound_context("concurrency-loops", workflow_slug="mcp-session", runtime=MCP_RUNTIME)
    mcp_ctx = MCPAgentContext(session_state=ctx.session_state)
    mcp_ctx.run_context = ctx.run_context
    toolkit = _Toolkit()
    tool = build_tool(make_spec("plan", _Toolkit, idempotent=True), toolkit, mcp_ctx)

    first = asyncio.run(tool(idempotency_key="same"))
    second = asyncio.run(tool(idempotency_key="same"))

    assert toolkit.calls == 1
    assert second["metrics"]["cached"] is True
    assert second["data"] == first["data"]


def test_sync_waiter_receives_the_owner_envelope(bound_context):
    ctx = bound_context("concurrency-waiter")
    toolkit = _Toolkit()
    spec = make_spec("blocking_plan", _Toolkit, idempotent=True)
    invoke = bind_sync_invoker(spec, toolkit.blocking_plan)
    outcomes: dict[str, Any] = {}

    def run(name: str) -> None:
        outcomes[name] = execute_sync(
            spec,
            ctx,
            {"idempotency_key": "coalesced"},
            invoke=invoke,
            owner_wait_timeout_s=5,
        )

    owner = threading.Thread(target=run, args=("owner",))
    owner.start()
    assert toolkit.started.wait(5)
    waiter = threading.Thread(target=run, args=("waiter",))
    waiter.start()
    time.sleep(0.05)
    toolkit.release.set()
    owner.join(5)
    waiter.join(5)

    assert toolkit.calls == 1
    assert outcomes["owner"].ok and outcomes["waiter"].ok
    assert outcomes["waiter"].envelope["metrics"]["cached"] is True
    assert outcomes["waiter"].envelope["data"] == outcomes["owner"].envelope["data"]
    assert ("tool_progress", "cache_hit") in tool_events(ctx)


def test_sync_waiter_refuses_to_block_a_running_loop():
    import concurrent.futures

    reservation = _IdempotencyReservation(
        identity=("run", "task", "handoff", "tool", "key"),
        request_digest="digest",
        future=concurrent.futures.Future(),
        owner=False,
    )

    async def wait_inside_loop() -> None:
        _wait_for_idempotent_owner(reservation, timeout=0.01)

    with pytest.raises(ToolExecutionError, match="running event loop"):
        asyncio.run(wait_inside_loop())


def test_idempotency_mutex_creation_is_race_free():
    ctx = MCPAgentContext()
    barrier = threading.Barrier(8)
    seen: list[Any] = []

    def grab() -> None:
        barrier.wait()
        seen.append(_idempotency_mutex(ctx))

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len({id(mutex) for mutex in seen}) == 1


def test_session_writes_on_one_run_are_serialized_across_threads(bound_context):
    ctx = bound_context("concurrency-serialized")
    toolkit = _Toolkit()
    spec = make_spec("slow_write", _Toolkit, write_scope="session")
    invoke = bind_sync_invoker(spec, toolkit.slow_write)

    threads = [
        threading.Thread(
            target=execute_sync,
            args=(spec, ctx, {"output_path": f"out-{index}.txt"}),
            kwargs={"invoke": invoke},
        )
        for index in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    intervals = sorted(toolkit.intervals)
    assert len(intervals) == 3
    for (_, previous_end), (next_begin, _) in zip(intervals, intervals[1:], strict=False):
        assert next_begin >= previous_end


def test_async_and_thread_writers_on_one_run_exclude_each_other(bound_context):
    ctx = bound_context("concurrency-mixed", workflow_slug="mcp-session", runtime=MCP_RUNTIME)
    mcp_ctx = MCPAgentContext(session_state=ctx.session_state)
    mcp_ctx.run_context = ctx.run_context
    toolkit = _Toolkit()
    spec = make_spec("slow_write", _Toolkit, write_scope="session")
    tool = build_tool(spec, toolkit, mcp_ctx)
    invoke = bind_sync_invoker(spec, toolkit.slow_write)

    async def scenario() -> list[dict[str, Any]]:
        # Thread-based callers (the in-process Agno team) and async MCP calls
        # share the run's thread lock.
        threads = [
            threading.Thread(
                target=execute_sync,
                args=(spec, ctx, {"output_path": f"sync-{index}.txt"}),
                kwargs={"invoke": invoke},
            )
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        envelopes = await asyncio.gather(
            *(tool(output_path=f"async-{index}.txt") for index in range(2))
        )
        for thread in threads:
            await asyncio.to_thread(thread.join)
        return list(envelopes)

    envelopes = asyncio.run(scenario())

    assert [envelope["status"] for envelope in envelopes] == ["success", "success"]
    intervals = sorted(toolkit.intervals)
    assert len(intervals) == 4
    for (_, previous_end), (next_begin, _) in zip(intervals, intervals[1:], strict=False):
        assert next_begin >= previous_end


def test_a_cancelled_lock_waiter_does_not_keep_the_run_locked(bound_context):
    ctx = bound_context("concurrency-cancelled", workflow_slug="mcp-session", runtime=MCP_RUNTIME)
    spec = make_spec("slow_write", _Toolkit, write_scope="session")
    held, release = threading.Event(), threading.Event()

    def hold_like_an_agno_call() -> None:
        with _sync_run_write_lock(spec, ctx):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold_like_an_agno_call)
    holder.start()
    assert held.wait(5)

    async def enter() -> bool:
        async with _async_run_write_locks(spec, ctx) as locked:
            return locked

    async def scenario() -> bool:
        waiter = asyncio.create_task(enter())
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        release.set()
        await asyncio.to_thread(holder.join)
        # The cancelled waiter's late acquisition is released again.
        return await asyncio.wait_for(enter(), timeout=2)

    assert asyncio.run(scenario()) is True
