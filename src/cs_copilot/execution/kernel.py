"""Invocation phases shared by the MCP and in-process runtimes."""

from __future__ import annotations

import copy
import logging
import typing
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Mapping

from .artifacts import _register_result_artifacts, _rollback_unregistered_publications
from .context import ExecutionContext, _active_output_layout, resolve_runtime
from .envelopes import (
    _cached_envelope,
    _coerce_return_value,
    _enforce_output_limit,
    _error_envelope,
    _success_envelope,
)
from .errors import ToolErrorCode, ToolExecutionError, normalize_error
from .events import record_tool_call, record_tool_progress
from .idempotency import (
    _abort_idempotency,
    _cache_store,
    _finish_idempotency,
    _idempotency_fingerprint,
    _IdempotencyReservation,
    _request_digest,
    _reserve_idempotency,
    _validate_idempotency_key,
)
from .read_boundary import _enforce_read_boundary
from .scope import (
    _assert_invocation_epoch_current,
    _authorize_invocation,
    _capture_invocation_scope,
    _enforce_execution_budget,
    _InvocationScope,
    _remaining_handoff_seconds,
)
from .spec import ToolSpec, _is_control_plane_spec
from .tracing import _new_tool_trace, _refresh_tool_trace
from .write_boundary import _enforce_write_boundary

logger = logging.getLogger(__name__)


def _record_tool_start(
    *,
    spec: ToolSpec,
    ctx: ExecutionContext,
    trace: Mapping[str, str | None],
    max_attempts: int,
    invocation_scope: _InvocationScope,
) -> None:
    """Reserve one observable task tool-call budget before execution."""

    def reserve_budget(_run: Any, events: typing.Sequence[Any]) -> None:
        _assert_invocation_epoch_current(spec, ctx, invocation_scope)
        _enforce_execution_budget(
            ctx,
            invocation_scope,
            events=events,
        )

    event_path = record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=trace,
        stage="started",
        attempt=0,
        max_attempts=max_attempts,
        execution_scope=invocation_scope.as_dict(),
        precondition=reserve_budget if invocation_scope.catalog_task else None,
        required=invocation_scope.catalog_task,
    )
    if invocation_scope.catalog_task and event_path is None:
        raise ToolExecutionError(
            "could not durably reserve the task tool-call budget",
            code=ToolErrorCode.INTERNAL,
        )


def _record_tool_acceptance(
    *,
    spec: ToolSpec,
    ctx: ExecutionContext,
    trace: Mapping[str, str | None],
    attempts: int,
    max_attempts: int,
    invocation_scope: _InvocationScope,
) -> None:
    """Linearize result acceptance before staged artifacts become visible."""

    def accept_current_epoch(_run: Any, _events: typing.Sequence[Any]) -> None:
        _assert_invocation_epoch_current(spec, ctx, invocation_scope)

    event_path = record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=trace,
        stage="result_accepted",
        attempt=attempts,
        max_attempts=max_attempts,
        execution_scope=invocation_scope.as_dict(),
        precondition=accept_current_epoch if invocation_scope.catalog_task else None,
        required=invocation_scope.catalog_task,
    )
    if invocation_scope.catalog_task and event_path is None:
        raise ToolExecutionError(
            "could not durably accept the task tool result",
            code=ToolErrorCode.INTERNAL,
        )


@contextmanager
def _tool_write_scope(spec: ToolSpec, ctx: ExecutionContext):
    """Pin artifact reads and stage bounded writes around domain tool code."""

    publications: dict[str, dict[str, Any]] = {}
    if _is_control_plane_spec(spec):
        yield publications
        return
    from cs_copilot.storage import S3

    layout = _active_output_layout(ctx)
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if layout is None or run is None:
        yield publications
        return
    protected_paths = tuple(
        layout.artifact_rel_path(record.relative_path) for record in run.artifacts.values()
    )
    verified_reads = {
        layout.artifact_rel_path(record.relative_path): (
            record.sha256,
            record.size_bytes,
        )
        for record in run.artifacts.values()
    }
    with ExitStack() as stack:
        stack.enter_context(S3.confine_artifact_reads(verified_reads))
        if spec.write_scope == "session" and not spec.read_only:
            stack.enter_context(
                S3.confine_writes(
                    layout.run_root,
                    protected_paths=protected_paths,
                    publication_receipt=publications,
                )
            )
        yield publications


def _retry_delay(spec: ToolSpec, attempt: int) -> float:
    multiplier = 2 ** min(max(0, int(attempt) - 1), 10)
    return min(float(spec.retry_backoff_s) * multiplier, 30.0)


# ---------------------------------------------------------------------------
# Invocation phases
#
# One tool call moves through these phases in order. Each phase reproduces one
# span of the original MCP adapter pipeline exactly; the orchestrators in
# ``cs_copilot.execution.runner`` only add the runtime-specific pieces
# (awaiting vs. blocking, locks, sleeping between retries).
# ---------------------------------------------------------------------------


@dataclass
class ToolInvocation:
    """Mutable state of one tool invocation.

    ``run_context`` is deliberately never cached here: control-plane tools may
    replace ``ctx.run_context`` while a call is in flight, and later events must
    follow the live run.
    """

    spec: ToolSpec
    ctx: ExecutionContext
    started: float
    trace: dict[str, str | None]
    max_attempts: int
    public_args: dict[str, Any]
    manifest_args: dict[str, Any]
    scope: _InvocationScope
    supplied_idempotency_key: Any = None
    idempotency_key: str | None = None
    request_digest: str | None = None
    reservation: _IdempotencyReservation | None = None
    session_view: dict[str, Any] | None = None
    attempts: int = 0


@dataclass(frozen=True)
class Admission:
    """Outcome of preparing an invocation: run it, replay a cache, or wait."""

    cached_envelope: dict[str, Any] | None = None
    wait_for: _IdempotencyReservation | None = None


@dataclass(frozen=True)
class AttemptContext:
    """What an executor needs to know about the current attempt."""

    attempt: int
    handoff_remaining_s: float | None
    scope: _InvocationScope


@dataclass(frozen=True)
class ExecutedCall:
    """Raw return value of one execution, plus a deferred worker commit if any."""

    value: Any
    deferred: Any = None


@dataclass(frozen=True)
class AcceptedResult:
    raw: Any
    coerced: Any
    output_bytes: int


@dataclass(frozen=True)
class CommittedResult:
    raw: Any
    coerced: Any
    output_bytes: int
    artifact_ids: list[str]
    warnings: list[str]


@dataclass(frozen=True)
class InvocationOutcome:
    """Final result: the v2 envelope and, on success, the raw tool value."""

    envelope: dict[str, Any]
    value: Any = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        return self.envelope.get("status") == "success"


Injector = Callable[[dict[str, Any], "dict[str, Any] | None"], None]


def begin_invocation(
    spec: ToolSpec,
    ctx: ExecutionContext,
    arguments: Mapping[str, Any],
) -> ToolInvocation:
    """Start timing and capture attribution before anything can await or fail."""

    started = perf_counter()
    trace = _new_tool_trace(ctx)
    public_args: dict[str, Any] = dict(arguments)
    supplied_idempotency_key = public_args.pop("idempotency_key", None)
    return ToolInvocation(
        spec=spec,
        ctx=ctx,
        started=started,
        trace=trace,
        max_attempts=spec.max_retries + 1,
        public_args=public_args,
        manifest_args=dict(public_args),
        scope=_capture_invocation_scope(spec, ctx),
        supplied_idempotency_key=supplied_idempotency_key,
    )


def prepare_invocation(inv: ToolInvocation) -> Admission:
    """Authorize, record the start, apply boundaries, and reserve idempotency.

    Synchronous by design: no await may happen between scope capture and the
    idempotency reservation.
    """

    spec, ctx = inv.spec, inv.ctx
    inv.scope = _authorize_invocation(spec, ctx)
    if inv.scope.catalog_task:
        ctx.run_context.verify_task_inputs(inv.scope.task_id)
    _record_tool_start(
        spec=spec,
        ctx=ctx,
        trace=inv.trace,
        max_attempts=inv.max_attempts,
        invocation_scope=inv.scope,
    )
    read_boundary = _enforce_read_boundary(spec, inv.public_args, ctx)
    inv.session_view = read_boundary.session_state
    inv.public_args = _enforce_write_boundary(spec, read_boundary.arguments, ctx)
    inv.manifest_args = copy.deepcopy(inv.public_args)
    inv.idempotency_key = _validate_idempotency_key(inv.supplied_idempotency_key, spec)
    if inv.idempotency_key is None:
        return Admission()
    inv.manifest_args["idempotency_key"] = _idempotency_fingerprint(inv.idempotency_key)
    inv.request_digest = _request_digest(inv.public_args, spec.forces)
    inv.reservation = _reserve_idempotency(
        ctx,
        spec=spec,
        trace=inv.trace,
        invocation_scope=inv.scope,
        idempotency_key=inv.idempotency_key,
        request_digest=inv.request_digest,
    )
    if inv.reservation.cached is not None:
        return Admission(cached_envelope=inv.reservation.cached)
    if not inv.reservation.owner:
        return Admission(wait_for=inv.reservation)
    return Admission()


def complete_from_shared(inv: ToolInvocation, shared: Mapping[str, Any]) -> dict[str, Any]:
    """Return a cached or coalesced owner envelope and record the cache hit."""

    spec, ctx = inv.spec, inv.ctx
    duration_ms = (perf_counter() - inv.started) * 1000
    envelope = _cached_envelope(shared, duration_ms=duration_ms, trace=inv.trace)
    record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=inv.trace,
        stage="cache_hit",
        attempt=0,
        max_attempts=inv.max_attempts,
        cached=True,
        execution_scope=inv.scope.as_dict(),
        required=inv.scope.catalog_task,
    )
    record_tool_call(
        ctx=ctx,
        tool_name=spec.mcp_name,
        public_args=inv.manifest_args,
        forced_args=spec.forces,
        status=str(envelope.get("status") or "error"),
        duration_ms=duration_ms,
        error=(
            str((envelope.get("error") or {}).get("message") or "")
            if envelope.get("status") == "error"
            else None
        ),
        result=envelope,
        execution_scope=inv.scope.as_dict(),
    )
    return envelope


def build_call_arguments(inv: ToolInvocation, *, inject: Injector | None = None) -> dict[str, Any]:
    """Public arguments, then runtime-injected context, then forced values."""

    call_kwargs: dict[str, Any] = dict(inv.public_args)
    if inject is not None:
        inject(call_kwargs, inv.session_view)
    call_kwargs.update(inv.spec.forces)
    return call_kwargs


def revalidate_after_lock(inv: ToolInvocation) -> None:
    """Re-read the run once serialized; scope may have changed while waiting."""

    run_context = getattr(inv.ctx, "run_context", None)
    if run_context is not None and hasattr(run_context, "refresh"):
        run_context.refresh(verify_artifacts=True)
    # Scope may have changed while this invocation waited for another
    # artifact-producing call to commit.
    _assert_invocation_epoch_current(inv.spec, inv.ctx, inv.scope)
    if inv.scope.catalog_task:
        inv.ctx.run_context.verify_task_inputs(inv.scope.task_id)


def begin_attempt(inv: ToolInvocation) -> None:
    """Count one execution attempt (before the attempt's failure handling)."""

    inv.attempts += 1


def attempt_context(inv: ToolInvocation) -> AttemptContext:
    """Describe the current attempt; raises once the handoff deadline has passed."""

    return AttemptContext(
        attempt=inv.attempts,
        handoff_remaining_s=_remaining_handoff_seconds(inv.scope),
        scope=inv.scope,
    )


def accept_result(inv: ToolInvocation, executed: ExecutedCall) -> AcceptedResult:
    """Validate the epoch, coerce and bound the output, and accept a worker commit."""

    worker_job = executed.deferred
    try:
        _assert_invocation_epoch_current(inv.spec, inv.ctx, inv.scope)
        _remaining_handoff_seconds(inv.scope)
        inv.trace = _refresh_tool_trace(inv.ctx, inv.trace)
        coerced = _coerce_return_value(executed.value)
        output_bytes = _enforce_output_limit(inv.spec, coerced)
        if worker_job is not None:
            worker_job.accept()
    except BaseException:
        if worker_job is not None:
            worker_job.abort()
        raise
    return AcceptedResult(raw=executed.value, coerced=coerced, output_bytes=output_bytes)


def commit_result(
    inv: ToolInvocation,
    accepted: AcceptedResult,
    *,
    local_publications: Mapping[str, dict[str, Any]],
    deferred: Any = None,
) -> CommittedResult:
    """Register result artifacts, release unregistered bytes, and linearize acceptance."""

    spec, ctx = inv.spec, inv.ctx
    publication_leases = {
        **local_publications,
        **(dict(deferred.publications) if deferred is not None else {}),
    }
    artifact_ids, artifact_warnings = _register_result_artifacts(
        spec,
        accepted.coerced,
        ctx,
        active_task_id=inv.scope.task_id,
        invocation_span_id=inv.trace.get("span_id"),
        publication_leases=publication_leases,
    )
    _rollback_unregistered_publications(ctx, publication_leases)
    _record_tool_acceptance(
        spec=spec,
        ctx=ctx,
        trace=inv.trace,
        attempts=inv.attempts,
        max_attempts=inv.max_attempts,
        invocation_scope=inv.scope,
    )
    return CommittedResult(
        raw=accepted.raw,
        coerced=accepted.coerced,
        output_bytes=accepted.output_bytes,
        artifact_ids=artifact_ids,
        warnings=artifact_warnings,
    )


def prepare_retry(
    inv: ToolInvocation,
    exc: BaseException,
    *,
    local_publications: Mapping[str, dict[str, Any]],
    deferred: Any = None,
) -> float | None:
    """Release a failed attempt; return the retry delay, or ``None`` to re-raise."""

    spec, ctx = inv.spec, inv.ctx
    worker_publications = dict(deferred.publications) if deferred is not None else {}
    _rollback_unregistered_publications(
        ctx,
        {
            **local_publications,
            **worker_publications,
        },
    )
    if deferred is not None and not deferred.settled:
        deferred.abort()
    normalized_attempt = normalize_error(
        exc,
        tool_name=spec.mcp_name,
        idempotent=spec.idempotent,
    )
    if inv.attempts >= inv.max_attempts or not normalized_attempt.retryable:
        return None
    delay_s = _retry_delay(spec, inv.attempts)
    record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=inv.trace,
        stage="retrying",
        attempt=inv.attempts,
        max_attempts=inv.max_attempts,
        message=(f"{normalized_attempt.message}; retrying in {delay_s:g}s"),
        execution_scope=inv.scope.as_dict(),
    )
    return delay_s


def record_cancellation(inv: ToolInvocation, *, message: str) -> None:
    """Close the span of an interrupted call and release idempotent waiters."""

    if inv.reservation is not None and inv.reservation.owner:
        _abort_idempotency(inv.ctx, inv.reservation)
    record_tool_progress(
        ctx=inv.ctx,
        tool_name=inv.spec.mcp_name,
        trace=inv.trace,
        stage="cancelled",
        attempt=inv.attempts,
        max_attempts=inv.max_attempts,
        message=message,
        execution_scope=inv.scope.as_dict(),
        required=inv.scope.catalog_task,
    )


def record_failure(inv: ToolInvocation, exc: BaseException) -> dict[str, Any]:
    """Normalize a failure into an error envelope and record it."""

    spec, ctx = inv.spec, inv.ctx
    inv.trace = _refresh_tool_trace(ctx, inv.trace)
    duration_ms = (perf_counter() - inv.started) * 1000
    normalized = normalize_error(
        exc,
        tool_name=spec.mcp_name,
        idempotent=spec.idempotent,
    )
    envelope = _error_envelope(
        normalized=normalized.as_dict(),
        duration_ms=duration_ms,
        trace=inv.trace,
        attempts=inv.attempts,
    )
    if inv.reservation is not None and inv.reservation.owner:
        _finish_idempotency(
            ctx,
            inv.reservation,
            envelope=envelope,
        )
    record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=inv.trace,
        stage="failed",
        attempt=inv.attempts,
        max_attempts=inv.max_attempts,
        message=normalized.message,
        execution_scope=inv.scope.as_dict(),
        required=inv.scope.catalog_task,
    )
    record_tool_call(
        ctx=ctx,
        tool_name=spec.mcp_name,
        public_args=inv.manifest_args,
        forced_args=spec.forces,
        status="error",
        duration_ms=duration_ms,
        error=normalized.message,
        result=envelope,
        execution_scope=inv.scope.as_dict(),
    )
    logger.error(
        "%s tool %s failed [%s]: %s",
        resolve_runtime(ctx).label.upper(),
        spec.mcp_name,
        normalized.code,
        normalized.message,
        exc_info=not isinstance(exc, ToolExecutionError),
    )
    return envelope


def finalize_success(inv: ToolInvocation, committed: CommittedResult) -> dict[str, Any]:
    """Build the success envelope, publish it to waiters, and record completion."""

    spec, ctx = inv.spec, inv.ctx
    duration_ms = (perf_counter() - inv.started) * 1000
    envelope = _success_envelope(
        committed.coerced,
        duration_ms=duration_ms,
        trace=inv.trace,
        attempts=inv.attempts,
        output_bytes=committed.output_bytes,
        artifact_ids=committed.artifact_ids,
        warnings=committed.warnings,
    )
    if inv.idempotency_key is not None and inv.request_digest is not None:
        _cache_store(
            ctx,
            spec=spec,
            trace=inv.trace,
            invocation_scope=inv.scope,
            idempotency_key=inv.idempotency_key,
            request_digest=inv.request_digest,
            envelope=envelope,
        )
    if inv.reservation is not None and inv.reservation.owner:
        _finish_idempotency(
            ctx,
            inv.reservation,
            envelope=envelope,
        )
    record_tool_progress(
        ctx=ctx,
        tool_name=spec.mcp_name,
        trace=inv.trace,
        stage="completed",
        attempt=inv.attempts,
        max_attempts=inv.max_attempts,
        cached=bool(envelope["metrics"]["cached"]),
        execution_scope=inv.scope.as_dict(),
    )
    record_tool_call(
        ctx=ctx,
        tool_name=spec.mcp_name,
        public_args=inv.manifest_args,
        forced_args=spec.forces,
        status="success",
        duration_ms=duration_ms,
        result=envelope,
        execution_scope=inv.scope.as_dict(),
    )
    return envelope
