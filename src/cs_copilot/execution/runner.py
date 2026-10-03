"""Orchestrators that drive one tool invocation through the kernel phases.

:func:`execute_async` is what the MCP server awaits for every tool call.
:func:`execute_sync` is the blocking counterpart for runtimes that execute
tools on worker threads without a running event loop (the in-process Agno
runtime dispatches synchronous toolkit functions through
``asyncio.to_thread``). Both produce the same durable events, boundaries, and
result envelope; they differ only in how they wait, lock, and sleep.

:func:`observe_sync` is the adoption mode for a runtime that must not change
behaviour yet: it records the same event protocol and registers the files a
call created, but only *audits* what the boundaries would have rewritten or
denied, and never retries, blocks, or repeats the call.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import inspect
import json
import logging
import time
from time import perf_counter
from typing import Any, Awaitable, Callable, Mapping

from .artifacts import _register_observed_writes, _register_result_artifacts
from .context import ExecutionContext, _active_output_layout
from .envelopes import _coerce_return_value, _error_envelope, _success_envelope
from .errors import ToolErrorCode, ToolExecutionError, normalize_error
from .events import record_tool_call, record_tool_progress
from .idempotency import _await_idempotent_owner, _wait_for_idempotent_owner
from .kernel import (
    AttemptContext,
    ExecutedCall,
    Injector,
    InvocationOutcome,
    ToolInvocation,
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
from .read_boundary import _enforce_read_boundary
from .scope import _handoff_timeout_error, _InvocationScope
from .spec import ToolSpec
from .tracing import _refresh_tool_trace
from .write_boundary import _enforce_write_boundary

logger = logging.getLogger(__name__)

_AUDIT_ITEMS = 50

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
    publication_policy: str = "result_paths",
    verify_artifacts_on_lock: bool = True,
    scope: _InvocationScope | None = None,
    extra: Mapping[str, Any] | None = None,
) -> InvocationOutcome:
    """Run one invocation to completion on the calling thread.

    The calling thread must not be running an event loop: idempotent waiters
    and retry backoff block. Any ``BaseException`` (for example a
    ``KeyboardInterrupt``) still records a terminal ``cancelled`` event so the
    run is never left with an open tool span.

    ``publication_policy="all_published"`` publishes each confined write when
    its file closes and registers every file the call wrote; see
    :func:`cs_copilot.execution.kernel.begin_invocation` for the other options.
    """

    inv = begin_invocation(
        spec,
        ctx,
        arguments,
        publication_policy=publication_policy,
        verify_artifacts_on_lock=verify_artifacts_on_lock,
        scope=scope,
        extra=extra,
    )
    commit_policy = "on_close" if publication_policy == "all_published" else "on_exit"
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
                    with _tool_write_scope(
                        spec, ctx, commit_policy=commit_policy
                    ) as local_publications:
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


def observe_sync(
    spec: ToolSpec,
    ctx: ExecutionContext,
    arguments: Mapping[str, Any],
    *,
    invoke: SyncInvoker,
    scope: _InvocationScope | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Any:
    """Run a tool exactly once while recording what the kernel would enforce.

    The durable ``started`` / ``artifact_registered`` / ``result_accepted`` /
    ``completed`` / ``tool_call_recorded`` protocol is recorded, and files the
    call created inside its run are registered as artifacts. Arguments, writes,
    and the returned value are left untouched: what the read and write
    boundaries would have rewritten or denied, and how each write would have
    been classified by confined writes, is recorded in the ``execution_audit``
    field of ``tool_call_recorded`` instead. Kernel bookkeeping failures are
    logged and never prevent or repeat the call; the tool's own exceptions
    propagate unchanged. ``extra`` is added to the ``started`` event (for
    example the owning process, for crash recovery).
    """

    audit: dict[str, Any] = {"mode": "observe"}
    call_arguments = dict(arguments)
    inv: ToolInvocation | None = None
    observation_scope: Any = contextlib.nullcontext(None)
    try:
        inv = begin_invocation(spec, ctx, call_arguments)
        if scope is not None:
            inv.scope = scope
        inv.max_attempts = 1
        record_tool_progress(
            ctx=ctx,
            tool_name=spec.mcp_name,
            trace=inv.trace,
            stage="started",
            attempt=0,
            max_attempts=1,
            execution_scope=inv.scope.as_dict(),
            extra=extra,
        )
        _audit_boundaries(spec, ctx, inv.public_args, audit)
        observation_scope = _observe_run_writes(ctx)
    except Exception as exc:  # noqa: BLE001 - observation never blocks the call
        _audit_kernel_problem(audit, "prepare", spec, exc)

    with observation_scope as observation:
        if inv is not None:
            inv.attempts = 1
        try:
            value = invoke(call_arguments)
        except BaseException as exc:
            _observe_failure(spec, ctx, inv, exc, audit, observation)
            raise
    _observe_success(spec, ctx, inv, value, audit, observation)
    return value


def _observe_run_writes(ctx: ExecutionContext) -> Any:
    from cs_copilot.storage import S3

    layout = _active_output_layout(ctx)
    run = getattr(getattr(ctx, "run_context", None), "run", None)
    if layout is None or run is None:
        return contextlib.nullcontext(None)
    protected = [
        layout.artifact_rel_path(record.relative_path) for record in run.artifacts.values()
    ]
    return S3.observe_writes(layout.run_root, protected_paths=protected)


def _audit_boundaries(
    spec: ToolSpec,
    ctx: ExecutionContext,
    public_args: Mapping[str, Any],
    audit: dict[str, Any],
) -> None:
    """Dry-run the read and write boundaries against a detached state view."""

    probe = copy.copy(ctx)
    session_state = getattr(ctx, "session_state", None)
    probe.session_state = dict(session_state) if isinstance(session_state, dict) else {}
    arguments = dict(public_args)
    try:
        read_boundary = _enforce_read_boundary(spec, arguments, probe)
        resolved = _changed_arguments(arguments, read_boundary.arguments)
        if resolved:
            audit["would_resolve_reads"] = resolved
    except Exception as exc:  # noqa: BLE001
        audit["would_deny_reads"] = str(exc)[:300]
    try:
        rewritten = _enforce_write_boundary(spec, arguments, probe)
        changes = _changed_arguments(arguments, rewritten)
        if changes:
            audit["would_rewrite"] = changes
    except Exception as exc:  # noqa: BLE001
        audit["would_deny_writes"] = str(exc)[:300]


def _changed_arguments(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in after.items() if before.get(key) != value}


def _observe_failure(
    spec: ToolSpec,
    ctx: ExecutionContext,
    inv: ToolInvocation | None,
    exc: BaseException,
    audit: dict[str, Any],
    observation: Any,
) -> None:
    if inv is None:
        return
    try:
        if observation is not None:
            audit["writes"] = observation.snapshot()[:_AUDIT_ITEMS]
        inv.trace = _refresh_tool_trace(ctx, inv.trace)
        normalized = normalize_error(exc, tool_name=spec.mcp_name)
        interrupted = not isinstance(exc, Exception)
        record_tool_progress(
            ctx=ctx,
            tool_name=spec.mcp_name,
            trace=inv.trace,
            stage="cancelled" if interrupted else "failed",
            attempt=inv.attempts,
            max_attempts=1,
            message=(
                "the tool invocation was interrupted before completion"
                if interrupted
                else normalized.message
            ),
            execution_scope=inv.scope.as_dict(),
        )
        if interrupted:
            return
        duration_ms = (perf_counter() - inv.started) * 1000
        record_tool_call(
            ctx=ctx,
            tool_name=spec.mcp_name,
            public_args=inv.manifest_args,
            forced_args={},
            status="error",
            duration_ms=duration_ms,
            error=normalized.message,
            result=_error_envelope(
                normalized=normalized.as_dict(),
                duration_ms=duration_ms,
                trace=inv.trace,
                attempts=inv.attempts,
            ),
            execution_scope=inv.scope.as_dict(),
            extra={"execution_audit": audit},
        )
    except Exception as problem:  # noqa: BLE001
        logger.warning("Observe-mode failure bookkeeping failed for %s: %s", spec.mcp_name, problem)


def _observe_success(
    spec: ToolSpec,
    ctx: ExecutionContext,
    inv: ToolInvocation | None,
    value: Any,
    audit: dict[str, Any],
    observation: Any,
) -> None:
    if inv is None:
        return
    try:
        created = observation.created_paths() if observation is not None else []
        if observation is not None:
            audit["writes"] = observation.snapshot()[:_AUDIT_ITEMS]
        producer_task_id = _running_task(ctx, inv.scope)
        coerced = _coerce_return_value(value)
        # Result paths first, so outputs get the typed contracts enforcement
        # relies on (for example gtm_model_path for trusted model reads)...
        artifact_ids, warnings = _register_result_artifacts(
            spec,
            coerced,
            ctx,
            active_task_id=producer_task_id,
            invocation_span_id=inv.trace.get("span_id"),
            publication_leases={key: {} for key in created},
            strict_result_paths=False,
        )
        if warnings:
            audit["registration_warnings"] = warnings[:_AUDIT_ITEMS]
        # ...then every other new file the call created inside the run.
        observed_ids, problems = _register_observed_writes(
            spec,
            ctx,
            _unregistered(ctx, created),
            producer_task_id=producer_task_id,
            invocation_span_id=inv.trace.get("span_id"),
        )
        artifact_ids = list(dict.fromkeys([*artifact_ids, *observed_ids]))
        if problems:
            audit["registration_problems"] = problems[:_AUDIT_ITEMS]
        inv.trace = _refresh_tool_trace(ctx, inv.trace)
        try:
            output_bytes = len(json.dumps(coerced, ensure_ascii=False, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            output_bytes = 0
        duration_ms = (perf_counter() - inv.started) * 1000
        envelope = _success_envelope(
            coerced,
            duration_ms=duration_ms,
            trace=inv.trace,
            attempts=inv.attempts,
            output_bytes=output_bytes,
            artifact_ids=artifact_ids,
        )
        for stage in ("result_accepted", "completed"):
            record_tool_progress(
                ctx=ctx,
                tool_name=spec.mcp_name,
                trace=inv.trace,
                stage=stage,
                attempt=inv.attempts,
                max_attempts=1,
                execution_scope=inv.scope.as_dict(),
            )
        record_tool_call(
            ctx=ctx,
            tool_name=spec.mcp_name,
            public_args=inv.manifest_args,
            forced_args={},
            status="success",
            duration_ms=duration_ms,
            result=envelope,
            execution_scope=inv.scope.as_dict(),
            extra={"execution_audit": audit},
        )
    except Exception as exc:  # noqa: BLE001
        _audit_kernel_problem(audit, "record", spec, exc)


def _unregistered(ctx: ExecutionContext, keys: list[str]) -> list[str]:
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    layout = getattr(run_context, "layout", None)
    if run is None or layout is None:
        return list(keys)
    registered = {
        layout.artifact_rel_path(record.relative_path) for record in run.artifacts.values()
    }
    return [key for key in keys if key not in registered]


def _running_task(ctx: ExecutionContext, scope: _InvocationScope) -> str | None:
    run = getattr(getattr(ctx, "run_context", None), "run", None)
    task = run.tasks.get(scope.task_id) if run is not None and scope.task_id else None
    if task is None or getattr(task.status, "value", str(task.status)) != "running":
        return None
    if scope.task_attempt is not None and int(scope.task_attempt) != int(task.attempts):
        return None
    return str(task.task_id)


def _audit_kernel_problem(
    audit: dict[str, Any],
    phase: str,
    spec: ToolSpec,
    exc: BaseException,
) -> None:
    audit.setdefault("kernel_errors", []).append({"phase": phase, "error": str(exc)[:300]})
    logger.warning("Observe-mode %s bookkeeping failed for %s: %s", phase, spec.mcp_name, exc)


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
