"""Task-scoped idempotency keys, request digests, and in-flight coalescing."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from .context import ExecutionContext
from .errors import ToolErrorCode, ToolExecutionError
from .scope import _InvocationScope
from .spec import ToolSpec

MAX_IDEMPOTENCY_ENTRIES = 128


@dataclass(frozen=True)
class _IdempotencyReservation:
    """One owner or waiter slot for a concurrent idempotent request."""

    identity: tuple[str, str, str, str, str]
    request_digest: str
    future: concurrent.futures.Future[dict[str, Any]] | None
    owner: bool
    cached: dict[str, Any] | None = None


def _validate_idempotency_key(value: Any, spec: ToolSpec) -> str | None:
    if value is None:
        return None
    if not spec.idempotent:
        raise ToolExecutionError(
            "idempotency_key is only supported for idempotent tools",
            code=ToolErrorCode.INVALID_INPUT,
        )
    if not isinstance(value, str) or not value.strip():
        raise ToolExecutionError(
            "idempotency_key must be a non-empty string",
            code=ToolErrorCode.INVALID_INPUT,
        )
    normalized = value.strip()
    if len(normalized) > 256:
        raise ToolExecutionError(
            "idempotency_key cannot exceed 256 characters",
            code=ToolErrorCode.INVALID_INPUT,
        )
    return normalized


def _idempotency_fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _request_digest(
    public_args: Mapping[str, Any],
    forced_args: Mapping[str, Any],
) -> str:
    encoded = json.dumps(
        {"public_args": public_args, "forced_args": forced_args},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_identity(
    spec: ToolSpec,
    trace: Mapping[str, str | None],
    invocation_scope: _InvocationScope,
    idempotency_key: str,
) -> tuple[str, str, str, str, str]:
    return (
        str(trace.get("run_id") or "unscoped"),
        str(invocation_scope.task_id or "unscoped"),
        str(invocation_scope.handoff_id or "ad-hoc"),
        spec.mcp_name,
        idempotency_key,
    )


def _idempotency_cache(ctx: ExecutionContext) -> OrderedDict:
    cache = getattr(ctx, "_execution_idempotency_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        ctx._execution_idempotency_cache = cache
    return cache


def _idempotency_digests(ctx: ExecutionContext) -> OrderedDict:
    digests = getattr(ctx, "_execution_idempotency_digests", None)
    if not isinstance(digests, OrderedDict):
        digests = OrderedDict()
        ctx._execution_idempotency_digests = digests
    return digests


def _idempotency_inflight(
    ctx: ExecutionContext,
) -> dict[tuple[str, str, str, str, str], _IdempotencyReservation]:
    inflight = getattr(ctx, "_execution_idempotency_inflight", None)
    if not isinstance(inflight, dict):
        inflight = {}
        ctx._execution_idempotency_inflight = inflight
    return inflight


_MUTEX_CREATION_LOCK = threading.Lock()


def _idempotency_mutex(ctx: ExecutionContext) -> threading.RLock:
    mutex = getattr(ctx, "_execution_idempotency_mutex", None)
    if mutex is None:
        # Two threads may reach a fresh context at once; exactly one mutex
        # must win, or their reservations would not exclude each other.
        with _MUTEX_CREATION_LOCK:
            mutex = getattr(ctx, "_execution_idempotency_mutex", None)
            if mutex is None:
                mutex = threading.RLock()
                ctx._execution_idempotency_mutex = mutex
    return mutex


def _reserve_idempotency(
    ctx: ExecutionContext,
    *,
    spec: ToolSpec,
    trace: Mapping[str, str | None],
    invocation_scope: _InvocationScope,
    idempotency_key: str,
    request_digest: str,
) -> _IdempotencyReservation:
    """Atomically return a cached result, owner slot, or waiter slot."""

    identity = _cache_identity(spec, trace, invocation_scope, idempotency_key)
    with _idempotency_mutex(ctx):
        digests = _idempotency_digests(ctx)
        previous_digest = digests.get(identity)
        if previous_digest is not None and previous_digest != request_digest:
            raise ToolExecutionError(
                "idempotency_key was already used with different arguments",
                code=ToolErrorCode.INVALID_INPUT,
            )
        digests[identity] = request_digest
        digests.move_to_end(identity)

        cached = _cache_lookup_unlocked(
            ctx,
            identity=identity,
            request_digest=request_digest,
        )
        if cached is not None:
            _trim_idempotency_digests(ctx)
            return _IdempotencyReservation(
                identity=identity,
                request_digest=request_digest,
                future=None,
                owner=False,
                cached=cached,
            )

        inflight = _idempotency_inflight(ctx)
        current = inflight.get(identity)
        if current is not None:
            if current.request_digest != request_digest:
                raise ToolExecutionError(
                    "idempotency_key was already used with different arguments",
                    code=ToolErrorCode.INVALID_INPUT,
                )
            return _IdempotencyReservation(
                identity=identity,
                request_digest=request_digest,
                future=current.future,
                owner=False,
            )

        reservation = _IdempotencyReservation(
            identity=identity,
            request_digest=request_digest,
            future=concurrent.futures.Future(),
            owner=True,
        )
        inflight[identity] = reservation
        _trim_idempotency_digests(ctx)
        return reservation


async def _await_idempotent_owner(
    reservation: _IdempotencyReservation,
) -> dict[str, Any]:
    """Wait for the owner without letting waiter cancellation cancel it."""

    if reservation.future is None:
        raise ToolExecutionError(
            "idempotency reservation is missing its in-flight result",
            code=ToolErrorCode.INTERNAL,
        )
    return await asyncio.shield(asyncio.wrap_future(reservation.future))


def _wait_for_idempotent_owner(
    reservation: _IdempotencyReservation,
    *,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Block until the owning call publishes its envelope (thread-only runtimes)."""

    if reservation.future is None:
        raise ToolExecutionError(
            "idempotency reservation is missing its in-flight result",
            code=ToolErrorCode.INTERNAL,
        )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise ToolExecutionError(
            "a synchronous idempotent waiter cannot block a running event loop",
            code=ToolErrorCode.INTERNAL,
        )
    try:
        return reservation.future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        raise ToolExecutionError(
            "timed out waiting for the owning idempotent call",
            code=ToolErrorCode.TIMEOUT,
            retryable=True,
        ) from exc


def _finish_idempotency(
    ctx: ExecutionContext,
    reservation: _IdempotencyReservation,
    *,
    envelope: Mapping[str, Any],
) -> None:
    """Publish one owner result to all concurrent waiters."""

    with _idempotency_mutex(ctx):
        inflight = _idempotency_inflight(ctx)
        current = inflight.get(reservation.identity)
        if current is not reservation:
            return
        inflight.pop(reservation.identity, None)
        future = reservation.future
        if future is not None and not future.done():
            future.set_result(copy.deepcopy(dict(envelope)))


def _abort_idempotency(
    ctx: ExecutionContext,
    reservation: _IdempotencyReservation,
) -> None:
    """Release waiters when an owning call is externally cancelled."""

    with _idempotency_mutex(ctx):
        inflight = _idempotency_inflight(ctx)
        current = inflight.get(reservation.identity)
        if current is not reservation:
            return
        inflight.pop(reservation.identity, None)
        future = reservation.future
        if future is not None and not future.done():
            future.set_exception(
                ToolExecutionError(
                    "the owning idempotent call was cancelled before publishing a result",
                    code=ToolErrorCode.TRANSIENT_EXTERNAL,
                    retryable=True,
                )
            )


def _trim_idempotency_digests(ctx: ExecutionContext) -> None:
    digests = _idempotency_digests(ctx)
    inflight = _idempotency_inflight(ctx)
    while len(digests) > MAX_IDEMPOTENCY_ENTRIES:
        removable = next((identity for identity in digests if identity not in inflight), None)
        if removable is None:
            return
        digests.pop(removable, None)


def _cache_lookup_unlocked(
    ctx: ExecutionContext,
    *,
    identity: tuple[str, str, str, str, str],
    request_digest: str,
) -> dict[str, Any] | None:
    cache = _idempotency_cache(ctx)
    entry = cache.get(identity)
    if entry is None:
        return None
    cache.move_to_end(identity)
    if entry["request_digest"] != request_digest:
        raise ToolExecutionError(
            "idempotency_key was already used with different arguments",
            code=ToolErrorCode.INVALID_INPUT,
        )
    return copy.deepcopy(entry["envelope"])


def _cache_store(
    ctx: ExecutionContext,
    *,
    spec: ToolSpec,
    trace: Mapping[str, str | None],
    invocation_scope: _InvocationScope,
    idempotency_key: str,
    request_digest: str,
    envelope: Mapping[str, Any],
) -> None:
    identity = _cache_identity(spec, trace, invocation_scope, idempotency_key)
    with _idempotency_mutex(ctx):
        cache = _idempotency_cache(ctx)
        cache[identity] = {
            "request_digest": request_digest,
            "envelope": copy.deepcopy(dict(envelope)),
        }
        cache.move_to_end(identity)
        while len(cache) > MAX_IDEMPOTENCY_ENTRIES:
            cache.popitem(last=False)
