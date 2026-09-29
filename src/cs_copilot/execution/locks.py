"""Per-run serialization of artifact-producing tool invocations."""

from __future__ import annotations

import asyncio
import threading
import typing
import weakref
from contextlib import contextmanager
from typing import Iterator

from .context import ExecutionContext, _active_output_layout
from .errors import ToolErrorCode, ToolExecutionError
from .spec import ToolSpec, _is_control_plane_spec

_RUN_WRITE_LOCKS_GUARD = threading.Lock()
_RUN_WRITE_LOCKS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop,
    dict[tuple[str, str], asyncio.Lock],
] = weakref.WeakKeyDictionary()
_ARTIFACT_CONTROL_PLANE_TOOLS = frozenset(
    {
        "workflow_register_artifact",
        "workflow_start_run",
    }
)


def _serializes_artifacts(spec: ToolSpec) -> bool:
    return (
        spec.write_scope == "session" and not spec.read_only and not _is_control_plane_spec(spec)
    ) or (spec.result_artifact_type is not None or spec.mcp_name in _ARTIFACT_CONTROL_PLANE_TOOLS)


def _run_write_lock_key(spec: ToolSpec, ctx: ExecutionContext) -> tuple[str, str] | None:
    """Return the ``(session_id, run_id)`` whose writes ``spec`` must serialize."""

    if not _serializes_artifacts(spec):
        return None
    layout = _active_output_layout(ctx)
    if layout is None:
        return None
    return (layout.session_id, layout.run_id)


def _run_write_lock(
    spec: ToolSpec,
    ctx: ExecutionContext,
) -> asyncio.Lock | None:
    """Return an event-loop-local lock for one artifact-producing workflow run."""

    key = _run_write_lock_key(spec, ctx)
    if key is None:
        return None

    loop = asyncio.get_running_loop()
    with _RUN_WRITE_LOCKS_GUARD:
        registry = _RUN_WRITE_LOCKS.setdefault(loop, {})
        lock = registry.get(key)
        if lock is None:
            lock = asyncio.Lock()
            registry[key] = lock
        return typing.cast(asyncio.Lock, lock)


class _SyncRunLock:
    """Weak-referenceable holder for one run's thread lock."""

    __slots__ = ("lock", "__weakref__")

    def __init__(self) -> None:
        self.lock = threading.Lock()


_SYNC_RUN_WRITE_LOCKS_GUARD = threading.Lock()
_SYNC_RUN_WRITE_LOCKS: weakref.WeakValueDictionary[tuple[str, str], _SyncRunLock] = (
    weakref.WeakValueDictionary()
)
_SYNC_LOCKS_HELD = threading.local()


@contextmanager
def _sync_run_write_lock(spec: ToolSpec, ctx: ExecutionContext) -> Iterator[bool]:
    """Serialize artifact-producing calls for one run across threads.

    Yields whether a lock was taken. This is the blocking counterpart of
    :func:`_run_write_lock` for runtimes that execute tools on worker threads.
    It does not exclude a concurrent async writer on the same run in the same
    process; the durable ledger's own preconditions still keep artifact
    attribution sound in that case. Re-entering the lock of the same run from
    the same thread raises instead of deadlocking.
    """

    key = _run_write_lock_key(spec, ctx)
    if key is None:
        yield False
        return
    held: set[tuple[str, str]] | None = getattr(_SYNC_LOCKS_HELD, "keys", None)
    if held is None:
        held = set()
        _SYNC_LOCKS_HELD.keys = held
    if key in held:
        raise ToolExecutionError(
            "a tool invocation cannot re-enter the write lock of its own workflow run",
            code=ToolErrorCode.INTERNAL,
        )
    with _SYNC_RUN_WRITE_LOCKS_GUARD:
        holder = _SYNC_RUN_WRITE_LOCKS.get(key)
        if holder is None:
            holder = _SyncRunLock()
            _SYNC_RUN_WRITE_LOCKS[key] = holder
    with holder.lock:
        held.add(key)
        try:
            yield True
        finally:
            held.discard(key)
