"""Orchestrators that drive one tool invocation through the kernel phases.

:func:`execute_async` is what the MCP server awaits for every tool call.
:func:`execute_sync` is the blocking counterpart for runtimes that execute
tools on worker threads without a running event loop (the in-process Agno
runtime dispatches synchronous toolkit functions through
``asyncio.to_thread``). Both produce the same durable events, boundaries, and
result envelope; they differ only in how they wait, lock, and sleep.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Awaitable, Callable, Mapping

from .context import ExecutionContext
from .errors import ToolErrorCode, ToolExecutionError
from .idempotency import _await_idempotent_owner, _wait_for_idempotent_owner
from .kernel import (
    AttemptContext,
    ExecutedCall,
    Injector,
    InvocationOutcome,
    _tool_write_scope,
    accept_result,
    attempt_context,
    begin_attempt,
    begin_invocation,
    build_call_arguments,
    commit_result,
    complete_from_shared,
    finalize_success,
    prepare_invocation,
    prepare_retry,
    record_cancellation,
    record_failure,
    revalidate_after_lock,
)
from .locks import _run_write_lock, _sync_run_write_lock
from .scope import _handoff_timeout_error
from .spec import ToolSpec

AsyncExecutor = Callable[[dict[str, Any], AttemptContext], Awaitable[ExecutedCall]]
SyncInvoker = Callable[[dict[str, Any]], Any]


async def execute_async(
    spec: ToolSpec,
    ctx: ExecutionContext,
    arguments: Mapping[str, Any],
    *,
    executor: AsyncExecutor,
    inject: Injector | None = None,
) -> InvocationOutcome:
    """Run one invocation on the current event loop.

    Arguments are used as given; callers that bypass a schema-validating
    transport are responsible for applying defaults, because recorded
    arguments and idempotency digests depend on them.
    """

    inv = begin_invocation(spec, ctx, arguments)
    try:
        admission = prepare_invocation(inv)
        if admission.cached_envelope is not None:
            return InvocationOutcome(envelope=complete_from_shared(inv, admission.cached_envelope))
        if admission.wait_for is not None:
            shared = await _await_idempotent_owner(admission.wait_for)
            return InvocationOutcome(envelope=complete_from_shared(inv, shared))

        call_kwargs = build_call_arguments(inv, inject=inject)
        write_lock = _run_write_lock(spec, ctx)
        lock_acquired = False
        try:
            if write_lock is not None:
                await write_lock.acquire()
                lock_acquired = True
                revalidate_after_lock(inv)

            while True:
                begin_attempt(inv)
                deferred = None
                local_publications: dict[str, dict[str, Any]] = {}
                try:
                    attempt = attempt_context(inv)
                    with _tool_write_scope(spec, ctx) as local_publications:
                        # The executor's task or thread must be created inside
                        # the write scope so it inherits the confinement.
                        executed = await executor(call_kwargs, attempt)
                        deferred = executed.deferred
                        accepted = accept_result(inv, executed)
                    committed = commit_result(
                        inv,
                        accepted,
                        local_publications=local_publications,
                        deferred=deferred,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    delay_s = prepare_retry(
                        inv,
                        exc,
                        local_publications=local_publications,
                        deferred=deferred,
                    )
                    if delay_s is None:
                        raise
                    if delay_s:
                        await asyncio.sleep(delay_s)
        finally:
            if lock_acquired:
                write_lock.release()
    except asyncio.CancelledError:
        record_cancellation(inv, message="client cancelled the tool invocation")
        raise
    except Exception as exc:  # noqa: BLE001
        return InvocationOutcome(envelope=record_failure(inv, exc), error=exc)

    return InvocationOutcome(envelope=finalize_success(inv, committed), value=committed.raw)


def execute_sync(
    spec: ToolSpec,
    ctx: ExecutionContext,
    arguments: Mapping[str, Any],
    *,
    invoke: SyncInvoker,
    inject: Injector | None = None,
    owner_wait_timeout_s: float | None = None,
) -> InvocationOutcome:
    """Run one invocation to completion on the calling thread.

    The calling thread must not be running an event loop: idempotent waiters
    and retry backoff block. Any ``BaseException`` (for example a
    ``KeyboardInterrupt``) still records a terminal ``cancelled`` event so the
    run is never left with an open tool span.
    """

    inv = begin_invocation(spec, ctx, arguments)
    try:
        admission = prepare_invocation(inv)
        if admission.cached_envelope is not None:
            return InvocationOutcome(envelope=complete_from_shared(inv, admission.cached_envelope))
        if admission.wait_for is not None:
            shared = _wait_for_idempotent_owner(admission.wait_for, timeout=owner_wait_timeout_s)
            return InvocationOutcome(envelope=complete_from_shared(inv, shared))

        call_kwargs = build_call_arguments(inv, inject=inject)
        with _sync_run_write_lock(spec, ctx) as locked:
            if locked:
                revalidate_after_lock(inv)

            while True:
                begin_attempt(inv)
                deferred = None
                local_publications: dict[str, dict[str, Any]] = {}
                try:
                    attempt_context(inv)
                    with _tool_write_scope(spec, ctx) as local_publications:
                        executed = _as_executed_call(invoke(call_kwargs))
                        deferred = executed.deferred
                        accepted = accept_result(inv, executed)
                    committed = commit_result(
                        inv,
                        accepted,
                        local_publications=local_publications,
                        deferred=deferred,
                    )
                    break
                except Exception as exc:  # noqa: BLE001
                    delay_s = prepare_retry(
                        inv,
                        exc,
                        local_publications=local_publications,
                        deferred=deferred,
                    )
                    if delay_s is None:
                        raise
                    if delay_s:
                        time.sleep(delay_s)
    except Exception as exc:  # noqa: BLE001
        return InvocationOutcome(envelope=record_failure(inv, exc), error=exc)
    except BaseException:
        record_cancellation(
            inv,
            message="the tool invocation was interrupted before completion",
        )
        raise

    return InvocationOutcome(envelope=finalize_success(inv, committed), value=committed.raw)


def in_process_async_executor(
    spec: ToolSpec,
    bound_method: Callable[..., Any],
) -> AsyncExecutor:
    """Execute a toolkit method in this process from an event loop.

    Python cannot safely cancel a running thread, so synchronous methods are
    drained to completion even when the caller is cancelled. Async methods
    honor ``spec.timeout_s`` and the remaining handoff deadline.
    """

    async def execute(call_kwargs: dict[str, Any], attempt: AttemptContext) -> ExecutedCall:
        if not inspect.iscoroutinefunction(bound_method):
            # Tool construction rejects timeout_s for this execution mode, and
            # caller cancellation is delayed until the call has completed so
            # mutations cannot race a retry in the background.
            return ExecutedCall(await _run_sync_to_completion(bound_method, **call_kwargs))

        pending = bound_method(**call_kwargs)
        effective_timeout = spec.timeout_s
        handoff_limited = False
        if attempt.handoff_remaining_s is not None and (
            effective_timeout is None or attempt.handoff_remaining_s <= effective_timeout
        ):
            effective_timeout = attempt.handoff_remaining_s
            handoff_limited = True
        if effective_timeout is None:
            return ExecutedCall(await pending)
        try:
            return ExecutedCall(await asyncio.wait_for(pending, timeout=effective_timeout))
        except asyncio.TimeoutError as exc:
            if handoff_limited:
                raise _handoff_timeout_error(attempt.scope) from exc
            raise ToolExecutionError(
                f"timed out after {effective_timeout:g}s",
                code=ToolErrorCode.TIMEOUT,
                retryable=True,
            ) from exc

    return execute


def bind_sync_invoker(spec: ToolSpec, bound_method: Callable[..., Any]) -> SyncInvoker:
    """Return an invoker for :func:`execute_sync`, rejecting unsafe bindings."""

    if inspect.iscoroutinefunction(bound_method):
        raise TypeError(f"{spec.mcp_name}: execute_sync cannot drive a coroutine function")
    if spec.timeout_s is not None:
        raise ValueError(
            f"{spec.mcp_name}: timeout_s cannot safely cancel an in-process "
            "synchronous tool; use a worker process or an async method"
        )

    def invoke(call_kwargs: dict[str, Any]) -> Any:
        return bound_method(**call_kwargs)

    return invoke


def _as_executed_call(value: Any) -> ExecutedCall:
    return value if isinstance(value, ExecutedCall) else ExecutedCall(value)


async def _run_sync_to_completion(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a sync call without ever abandoning its worker thread."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while True:
            try:
                await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                # Repeated caller cancellation must not orphan a mutating
                # thread. Drain it before restoring cancellation.
                continue
            except Exception:  # noqa: BLE001
                # Retrieve any worker exception to avoid an unobserved task.
                break
        raise cancelled
