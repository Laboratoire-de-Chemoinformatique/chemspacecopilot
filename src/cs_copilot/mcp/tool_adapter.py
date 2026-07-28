"""Adapt cs_copilot toolkit methods into bounded, typed MCP v2 tools.

The adapter owns context injection, worker dispatch, output normalization,
timeouts, result envelopes, and stable error semantics.  Scientific logic
remains in the toolkit methods themselves.
"""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import copy
import hashlib
import inspect
import json
import logging
import math
import mimetypes
import re
import threading
import typing
import uuid
import weakref
from collections import OrderedDict
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import unquote, urlsplit

from .context import MCPAgentContext
from .errors import MCPErrorCode, MCPToolError, normalize_error

logger = logging.getLogger(__name__)

INJECTED_PARAMS = ("agent", "session_state")
RESULT_SCHEMA_VERSION = 2
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000
MAX_IDEMPOTENCY_ENTRIES = 128
_DATAFRAME_PREVIEW_ROWS = 200
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_WRITE_SCOPES = frozenset({"none", "session", "external"})
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_BACKTICK_PATH_RE = re.compile(r"`([^`\r\n]+)`")
_KNOWN_LABELED_BACKTICK_RE = re.compile(
    r"(?im)^[^`\r\n]*?(?P<label>"
    r"Clean dataset(?: \([^)]*\))?|Raw dataset|Descriptor Parquet|"
    r"Standardization report|Filtered rows|HTML|PDF|Markdown"
    r"):\s*`(?P<path>[^`\r\n]+)`"
)
_GTM_SAVED_PATHS_RE = re.compile(
    r"^\s*dataset_path:\s*(?P<dataset>[^;\r\n]+?)\s*;" r"\s*gtm_path:\s*(?P<gtm>[^\r\n]+?)\s*$"
)
_WRITE_PATH_KEYS = frozenset(
    {
        "destination",
        "destination_file",
        "destination_filename",
        "destination_path",
        "dest_file",
        "dest_path",
        "export_file",
        "export_filename",
        "export_path",
        "output_file",
        "output_filename",
        "output_path",
        "save_file",
        "save_filename",
        "save_path",
        "save_to",
        "target_file",
        "target_path",
        "write_path",
    }
)
_WRITE_PATH_CONTAINER_KEYS = frozenset(
    {
        "destination",
        "destinations",
        "exports",
        "output",
        "outputs",
        "sink",
        "sinks",
        "target",
        "targets",
    }
)
_GENERIC_PATH_KEYS = frozenset(
    {
        "file",
        "filename",
        "file_path",
        "filepath",
        "filepath_or_buffer",
        "buf",
        "excel_writer",
        "path",
        "path_or_buffer",
        "path_or_buf",
        "uri",
        "url",
    }
)
_PARAMETER_BAG_KEYS = frozenset(
    {
        "function_parameters",
        "kwargs",
        "operation_parameters",
        "options",
        "parameters",
    }
)
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)
_DERIVED_WRITE_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "gtm_optimization": ("dataset_name", "gtm_name"),
    "gtm_save_model_and_data": ("dataset_name", "gtm_name"),
    "gtm_train_on_latent_space": ("dataset_name", "gtm_name"),
}
_PANDAS_WRITE_OPERATION_ALIASES = {
    "export_csv": "to_csv",
    "save_csv": "to_csv",
    "write_csv": "to_csv",
}
_PANDAS_EXTERNAL_WRITE_OPERATIONS = frozenset({"to_clipboard", "to_gbq", "to_sql"})
_PANDAS_SAFE_CREATE_FUNCTIONS = frozenset(
    {
        "DataFrame",
        "DataFrame.from_dict",
        "DataFrame.from_records",
        "from_dict",
        "from_file",
        "from_records",
        "from_s3",
        "read_csv",
    }
)
_PANDAS_FILE_CREATE_FUNCTIONS = frozenset({"from_file", "from_s3", "read_csv"})
_PANDAS_FILE_PARAMETER_KEYS: dict[str, tuple[str, ...]] = {
    "read_csv": (
        "path_or_buf",
        "path_or_buffer",
        "filepath_or_buffer",
        "file_path",
        "filepath",
        "path",
    ),
    "from_s3": ("s3_path",),
    "from_file": ("file_path",),
}
_PANDAS_LOADABLE_SUFFIXES = (
    ".csv",
    ".csv.gz",
    ".tsv",
    ".tab",
    ".txt",
)
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
_REPORT_FIGURE_PATH_KEYS = frozenset(
    {
        "artifact_path",
        "html_path",
        "image_path",
        "interactive_path",
        "path",
        "png_path",
        "src",
    }
)


@dataclass(frozen=True)
class ToolSpec:
    """Declarative capability and execution contract for one MCP tool."""

    mcp_name: str
    toolkit_factory: Callable[[], Any]
    method: str
    summary: str
    group: str | None = None
    forces: Mapping[str, Any] = field(default_factory=dict)
    read_only: bool = False
    destructive: bool = False
    open_world: bool = False
    idempotent: bool = False
    risk: str = "low"
    roles: tuple[str, ...] = ()
    profiles: tuple[str, ...] = ()
    timeout_s: Optional[float] = None
    max_output_bytes: Optional[int] = DEFAULT_MAX_OUTPUT_BYTES
    max_retries: int = 0
    retry_backoff_s: float = 0.25
    requires_network: bool = False
    write_scope: str = "none"
    read_artifact_fields: tuple[str, ...] = ()
    trusted_pickle_fields: tuple[str, ...] = ()
    result_artifact_type: str | None = None
    run_in_worker_process: bool = False
    worker_timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.risk not in _RISK_LEVELS:
            raise ValueError(f"{self.mcp_name}: invalid risk {self.risk!r}")
        if self.write_scope not in _WRITE_SCOPES:
            raise ValueError(f"{self.mcp_name}: invalid write_scope {self.write_scope!r}")
        normalized_read_fields = tuple(
            dict.fromkeys(str(item).strip() for item in self.read_artifact_fields)
        )
        if any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item) for item in normalized_read_fields
        ):
            raise ValueError(f"{self.mcp_name}: read_artifact_fields must contain parameter names")
        object.__setattr__(self, "read_artifact_fields", normalized_read_fields)
        normalized_pickle_fields = tuple(
            dict.fromkeys(str(item).strip() for item in self.trusted_pickle_fields)
        )
        if not set(normalized_pickle_fields).issubset(normalized_read_fields):
            raise ValueError(
                f"{self.mcp_name}: trusted_pickle_fields must also be declared "
                "in read_artifact_fields"
            )
        object.__setattr__(self, "trusted_pickle_fields", normalized_pickle_fields)
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"{self.mcp_name}: timeout_s must be positive")
        if self.worker_timeout_s is not None and self.worker_timeout_s <= 0:
            raise ValueError(f"{self.mcp_name}: worker_timeout_s must be positive")
        if self.max_output_bytes is not None and self.max_output_bytes <= 0:
            raise ValueError(f"{self.mcp_name}: max_output_bytes must be positive")
        if self.max_retries < 0:
            raise ValueError(f"{self.mcp_name}: max_retries cannot be negative")
        if self.max_retries and not self.idempotent:
            raise ValueError(f"{self.mcp_name}: retries require an idempotent tool")
        if self.retry_backoff_s < 0:
            raise ValueError(f"{self.mcp_name}: retry_backoff_s cannot be negative")
        if self.result_artifact_type is not None and not _ARTIFACT_TYPE_RE.fullmatch(
            self.result_artifact_type
        ):
            raise ValueError(
                f"{self.mcp_name}: invalid result_artifact_type " f"{self.result_artifact_type!r}"
            )
        if self.result_artifact_type is not None and self.read_only:
            raise ValueError(f"{self.mcp_name}: a result artifact cannot be declared read-only")
        if self.result_artifact_type is not None and self.write_scope == "none":
            object.__setattr__(self, "write_scope", "session")


@dataclass(frozen=True)
class _IdempotencyReservation:
    """One owner or waiter slot for a concurrent idempotent request."""

    identity: tuple[str, str, str, str, str]
    request_digest: str
    future: concurrent.futures.Future[dict[str, Any]] | None
    owner: bool
    cached: dict[str, Any] | None = None


@dataclass(frozen=True)
class _ReadBoundaryResult:
    """Canonical arguments plus an optional invocation-local session view."""

    arguments: dict[str, Any]
    session_state: dict[str, Any] | None = None


class _SessionStateReadSnapshot(dict[str, Any]):
    """A live-write session view with selected keys frozen for deterministic reads."""

    def __init__(
        self,
        target: dict[str, Any],
        *,
        frozen: Mapping[str, Any],
    ) -> None:
        super().__init__(target)
        self._target = target
        self._frozen_keys = frozenset(frozen)
        dict.update(self, frozen)

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        if key not in self._frozen_keys:
            self._target[key] = value

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        if key not in self._frozen_keys:
            del self._target[key]

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in self:
            return self[key]
        self[key] = default
        return default


@dataclass(frozen=True)
class _InvocationScope:
    """Immutable role/task attribution captured before a tool can await."""

    run_id: str | None
    task_id: str | None
    role: str | None
    profile: str | None
    task_attempt: int | None = None
    catalog_task: bool = False
    handoff_id: str | None = None
    handoff_created_at: str | None = None
    max_tool_calls: int | None = None
    timeout_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "role": self.role,
            "profile": self.profile,
            "task_attempt": self.task_attempt,
            "handoff_id": self.handoff_id,
        }


def _coerce_return_value(value: Any) -> Any:
    """Best-effort JSON-friendly coercion of toolkit return values."""

    import pandas as pd

    if isinstance(value, pd.DataFrame):
        row_count = int(len(value))
        columns = [str(c) for c in value.columns]
        if row_count <= _DATAFRAME_PREVIEW_ROWS:
            return {
                "row_count": row_count,
                "columns": columns,
                "records": value.to_dict("records"),
            }
        return {
            "row_count": row_count,
            "columns": columns,
            "preview": value.head(_DATAFRAME_PREVIEW_ROWS).to_dict("records"),
            "note": (
                f"DataFrame exceeded {_DATAFRAME_PREVIEW_ROWS} rows; only the "
                "first rows are inlined. Use a tool that persists the dataset "
                "or read the run resource for the full content."
            ),
        }
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _resolve_annotations(method: Callable[..., Any]) -> Dict[str, Any]:
    target = getattr(method, "__func__", method)
    try:
        return typing.get_type_hints(target, include_extras=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Falling back to raw annotations for %s: %s", method, exc)
        return getattr(target, "__annotations__", {}) or {}


def _public_parameters(method: Callable[..., Any], forces: Mapping[str, Any]):
    """Return signature parameters exposed to the MCP client."""

    sig = inspect.signature(method)
    resolved = _resolve_annotations(method)
    hidden = set(INJECTED_PARAMS) | set(forces)
    kept = []
    for name, param in sig.parameters.items():
        if name == "self" or name in hidden:
            continue
        annotation = resolved.get(name, param.annotation)
        kept.append(param.replace(annotation=annotation))
    return sig, kept


def _with_idempotency_parameter(
    parameters: list[inspect.Parameter],
    spec: ToolSpec,
) -> list[inspect.Parameter]:
    if not spec.idempotent:
        return parameters
    if any(param.name == "idempotency_key" for param in parameters):
        raise ValueError(
            f"{spec.mcp_name}: idempotency_key is reserved by the MCP execution policy"
        )
    idempotency_param = inspect.Parameter(
        "idempotency_key",
        kind=inspect.Parameter.KEYWORD_ONLY,
        default=None,
        annotation=Optional[str],
    )
    kept = list(parameters)
    for index, param in enumerate(kept):
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            kept.insert(index, idempotency_param)
            break
    else:
        kept.append(idempotency_param)
    return kept


def build_tool(
    spec: ToolSpec,
    instance: Any,
    ctx: MCPAgentContext,
) -> Callable[..., Any]:
    """Return an async FastMCP callable with a stable v2 result envelope."""

    bound_method = getattr(instance, spec.method)
    if (
        spec.timeout_s is not None
        and not spec.run_in_worker_process
        and not inspect.iscoroutinefunction(bound_method)
    ):
        raise ValueError(
            f"{spec.mcp_name}: timeout_s cannot safely cancel an in-process "
            "synchronous tool; use a worker process or an async method"
        )
    sig, public_params = _public_parameters(bound_method, spec.forces)
    public_param_names = {param.name for param in public_params}
    unknown_read_fields = sorted(set(spec.read_artifact_fields) - public_param_names)
    if unknown_read_fields:
        raise ValueError(
            f"{spec.mcp_name}: read_artifact_fields reference unknown public "
            f"parameters: {', '.join(unknown_read_fields)}"
        )
    public_params = _with_idempotency_parameter(public_params, spec)
    public_signature = inspect.Signature(
        parameters=public_params,
        return_annotation=dict[str, Any],
    )
    public_annotations: Dict[str, Any] = {
        param.name: param.annotation
        for param in public_params
        if param.annotation is not inspect.Parameter.empty
    }
    public_annotations["return"] = dict[str, Any]

    async def _invoke(**kwargs: Any) -> dict[str, Any]:
        from .manifests import record_tool_call, record_tool_progress

        started = perf_counter()
        trace = _new_tool_trace(ctx)
        max_attempts = spec.max_retries + 1
        attempts = 0
        public_args: Dict[str, Any] = dict(kwargs)
        supplied_idempotency_key = public_args.pop("idempotency_key", None)
        manifest_args = dict(public_args)
        idempotency_reservation: _IdempotencyReservation | None = None
        invocation_scope = _capture_invocation_scope(spec, ctx)

        try:
            invocation_scope = _authorize_invocation(spec, ctx)
            if invocation_scope.catalog_task:
                ctx.run_context.verify_task_inputs(invocation_scope.task_id)
            _record_tool_start(
                spec=spec,
                ctx=ctx,
                trace=trace,
                max_attempts=max_attempts,
                invocation_scope=invocation_scope,
            )
            read_boundary = _enforce_read_boundary(spec, public_args, ctx)
            public_args = read_boundary.arguments
            public_args = _enforce_write_boundary(spec, public_args, ctx)
            manifest_args = copy.deepcopy(public_args)
            idempotency_key = _validate_idempotency_key(supplied_idempotency_key, spec)
            request_digest: str | None = None
            if idempotency_key is not None:
                manifest_args["idempotency_key"] = _idempotency_fingerprint(idempotency_key)
                request_digest = _request_digest(public_args, spec.forces)
                idempotency_reservation = _reserve_idempotency(
                    ctx,
                    spec=spec,
                    trace=trace,
                    invocation_scope=invocation_scope,
                    idempotency_key=idempotency_key,
                    request_digest=request_digest,
                )
                cached = idempotency_reservation.cached
                if cached is not None:
                    duration_ms = (perf_counter() - started) * 1000
                    envelope = _cached_envelope(cached, duration_ms=duration_ms, trace=trace)
                    record_tool_progress(
                        ctx=ctx,
                        tool_name=spec.mcp_name,
                        trace=trace,
                        stage="cache_hit",
                        attempt=0,
                        max_attempts=max_attempts,
                        cached=True,
                        execution_scope=invocation_scope.as_dict(),
                        required=invocation_scope.catalog_task,
                    )
                    record_tool_call(
                        ctx=ctx,
                        tool_name=spec.mcp_name,
                        public_args=manifest_args,
                        forced_args=spec.forces,
                        status="success",
                        duration_ms=duration_ms,
                        result=envelope,
                        execution_scope=invocation_scope.as_dict(),
                    )
                    return envelope
                if not idempotency_reservation.owner:
                    shared = await _await_idempotent_owner(idempotency_reservation)
                    duration_ms = (perf_counter() - started) * 1000
                    envelope = _cached_envelope(shared, duration_ms=duration_ms, trace=trace)
                    record_tool_progress(
                        ctx=ctx,
                        tool_name=spec.mcp_name,
                        trace=trace,
                        stage="cache_hit",
                        attempt=0,
                        max_attempts=max_attempts,
                        cached=True,
                        execution_scope=invocation_scope.as_dict(),
                        required=invocation_scope.catalog_task,
                    )
                    record_tool_call(
                        ctx=ctx,
                        tool_name=spec.mcp_name,
                        public_args=manifest_args,
                        forced_args=spec.forces,
                        status=str(envelope.get("status") or "error"),
                        duration_ms=duration_ms,
                        error=(
                            str((envelope.get("error") or {}).get("message") or "")
                            if envelope.get("status") == "error"
                            else None
                        ),
                        result=envelope,
                        execution_scope=invocation_scope.as_dict(),
                    )
                    return envelope

            call_kwargs: Dict[str, Any] = dict(public_args)
            if not spec.run_in_worker_process:
                if "agent" in sig.parameters:
                    if read_boundary.session_state is None:
                        call_kwargs["agent"] = ctx
                    else:
                        invocation_agent = copy.copy(ctx)
                        invocation_agent.session_state = read_boundary.session_state
                        call_kwargs["agent"] = invocation_agent
                if "session_state" in sig.parameters:
                    call_kwargs["session_state"] = (
                        read_boundary.session_state
                        if read_boundary.session_state is not None
                        else ctx.session_state
                    )
            call_kwargs.update(spec.forces)

            write_lock = _run_write_lock(spec, ctx)
            lock_acquired = False
            try:
                if write_lock is not None:
                    await write_lock.acquire()
                    lock_acquired = True
                    run_context = getattr(ctx, "run_context", None)
                    if run_context is not None and hasattr(run_context, "refresh"):
                        run_context.refresh(verify_artifacts=True)
                    # Scope may have changed while this invocation waited for
                    # another artifact-producing call to commit.
                    _assert_invocation_epoch_current(spec, ctx, invocation_scope)
                    if invocation_scope.catalog_task:
                        ctx.run_context.verify_task_inputs(invocation_scope.task_id)

                while True:
                    attempts += 1
                    worker_job = None
                    local_publications: dict[str, dict[str, Any]] = {}
                    try:
                        handoff_remaining_s = _remaining_handoff_seconds(invocation_scope)
                        with _tool_write_scope(spec, ctx) as local_publications:
                            result = await _execute(
                                spec,
                                bound_method,
                                call_kwargs,
                                ctx,
                                handoff_remaining_s=handoff_remaining_s,
                                invocation_scope=invocation_scope,
                            )
                            if spec.run_in_worker_process:
                                from .jobs import DeferredToolJob

                                if not isinstance(result, DeferredToolJob):
                                    raise MCPToolError(
                                        "worker result was not deferred for parent acceptance",
                                        code=MCPErrorCode.INTERNAL,
                                    )
                                worker_job = result
                                result = worker_job.result
                            try:
                                _assert_invocation_epoch_current(
                                    spec,
                                    ctx,
                                    invocation_scope,
                                )
                                _remaining_handoff_seconds(invocation_scope)
                                trace = _refresh_tool_trace(ctx, trace)
                                coerced = _coerce_return_value(result)
                                output_bytes = _enforce_output_limit(spec, coerced)
                                if worker_job is not None:
                                    worker_job.accept()
                            except BaseException:
                                if worker_job is not None:
                                    worker_job.abort()
                                raise
                        publication_leases = {
                            **local_publications,
                            **(dict(worker_job.publications) if worker_job is not None else {}),
                        }
                        artifact_ids, artifact_warnings = _register_result_artifacts(
                            spec,
                            coerced,
                            ctx,
                            active_task_id=invocation_scope.task_id,
                            invocation_span_id=trace.get("span_id"),
                            publication_leases=publication_leases,
                        )
                        _rollback_unregistered_publications(ctx, publication_leases)
                        _record_tool_acceptance(
                            spec=spec,
                            ctx=ctx,
                            trace=trace,
                            attempts=attempts,
                            max_attempts=max_attempts,
                            invocation_scope=invocation_scope,
                        )
                        break
                    except Exception as exc:  # noqa: BLE001
                        worker_publications = (
                            dict(worker_job.publications) if worker_job is not None else {}
                        )
                        _rollback_unregistered_publications(
                            ctx,
                            {
                                **local_publications,
                                **worker_publications,
                            },
                        )
                        if worker_job is not None and not worker_job.settled:
                            worker_job.abort()
                        normalized_attempt = normalize_error(
                            exc,
                            tool_name=spec.mcp_name,
                            idempotent=spec.idempotent,
                        )
                        if attempts >= max_attempts or not normalized_attempt.retryable:
                            raise
                        delay_s = _retry_delay(spec, attempts)
                        record_tool_progress(
                            ctx=ctx,
                            tool_name=spec.mcp_name,
                            trace=trace,
                            stage="retrying",
                            attempt=attempts,
                            max_attempts=max_attempts,
                            message=(f"{normalized_attempt.message}; retrying in {delay_s:g}s"),
                            execution_scope=invocation_scope.as_dict(),
                        )
                        if delay_s:
                            await asyncio.sleep(delay_s)

            finally:
                if lock_acquired:
                    write_lock.release()
        except asyncio.CancelledError:
            if idempotency_reservation is not None and idempotency_reservation.owner:
                _abort_idempotency(ctx, idempotency_reservation)
            record_tool_progress(
                ctx=ctx,
                tool_name=spec.mcp_name,
                trace=trace,
                stage="cancelled",
                attempt=attempts,
                max_attempts=max_attempts,
                message="client cancelled the tool invocation",
                execution_scope=invocation_scope.as_dict(),
                required=invocation_scope.catalog_task,
            )
            raise
        except Exception as exc:  # noqa: BLE001
            trace = _refresh_tool_trace(ctx, trace)
            duration_ms = (perf_counter() - started) * 1000
            normalized = normalize_error(
                exc,
                tool_name=spec.mcp_name,
                idempotent=spec.idempotent,
            )
            envelope = _error_envelope(
                normalized=normalized.as_dict(),
                duration_ms=duration_ms,
                trace=trace,
                attempts=attempts,
            )
            if idempotency_reservation is not None and idempotency_reservation.owner:
                _finish_idempotency(
                    ctx,
                    idempotency_reservation,
                    envelope=envelope,
                )
            record_tool_progress(
                ctx=ctx,
                tool_name=spec.mcp_name,
                trace=trace,
                stage="failed",
                attempt=attempts,
                max_attempts=max_attempts,
                message=normalized.message,
                execution_scope=invocation_scope.as_dict(),
                required=invocation_scope.catalog_task,
            )
            record_tool_call(
                ctx=ctx,
                tool_name=spec.mcp_name,
                public_args=manifest_args,
                forced_args=spec.forces,
                status="error",
                duration_ms=duration_ms,
                error=normalized.message,
                result=envelope,
                execution_scope=invocation_scope.as_dict(),
            )
            logger.error(
                "MCP tool %s failed [%s]: %s",
                spec.mcp_name,
                normalized.code,
                normalized.message,
                exc_info=not isinstance(exc, MCPToolError),
            )
            return envelope

        duration_ms = (perf_counter() - started) * 1000
        envelope = _success_envelope(
            coerced,
            duration_ms=duration_ms,
            trace=trace,
            attempts=attempts,
            output_bytes=output_bytes,
            artifact_ids=artifact_ids,
            warnings=artifact_warnings,
        )
        if idempotency_key is not None and request_digest is not None:
            _cache_store(
                ctx,
                spec=spec,
                trace=trace,
                invocation_scope=invocation_scope,
                idempotency_key=idempotency_key,
                request_digest=request_digest,
                envelope=envelope,
            )
        if idempotency_reservation is not None and idempotency_reservation.owner:
            _finish_idempotency(
                ctx,
                idempotency_reservation,
                envelope=envelope,
            )
        record_tool_progress(
            ctx=ctx,
            tool_name=spec.mcp_name,
            trace=trace,
            stage="completed",
            attempt=attempts,
            max_attempts=max_attempts,
            cached=bool(envelope["metrics"]["cached"]),
            execution_scope=invocation_scope.as_dict(),
        )
        record_tool_call(
            ctx=ctx,
            tool_name=spec.mcp_name,
            public_args=manifest_args,
            forced_args=spec.forces,
            status="success",
            duration_ms=duration_ms,
            result=envelope,
            execution_scope=invocation_scope.as_dict(),
        )
        return envelope

    _invoke.__name__ = spec.mcp_name
    _invoke.__qualname__ = spec.mcp_name
    _invoke.__doc__ = spec.summary or (bound_method.__doc__ or "").strip()
    _invoke.__signature__ = public_signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = public_annotations
    _invoke.__wrapped__ = bound_method  # type: ignore[attr-defined]
    return _invoke


async def _execute(
    spec: ToolSpec,
    bound_method: Callable[..., Any],
    call_kwargs: Mapping[str, Any],
    ctx: MCPAgentContext,
    *,
    handoff_remaining_s: float | None = None,
    invocation_scope: _InvocationScope | None = None,
) -> Any:
    if spec.run_in_worker_process:
        from .jobs import run_tool_job

        # The worker runner owns process-group termination and uses
        # ``worker_timeout_s``. Cancellation is delayed until the cleanup
        # thread has reaped that process, so no mutating job is orphaned.
        effective_spec = spec
        handoff_limited = False
        if handoff_remaining_s is not None and (
            spec.worker_timeout_s is None or handoff_remaining_s <= spec.worker_timeout_s
        ):
            effective_spec = replace(spec, worker_timeout_s=handoff_remaining_s)
            handoff_limited = True
        try:
            return await _run_deferred_worker_to_completion(
                run_tool_job,
                effective_spec,
                call_kwargs,
                ctx,
                defer_commit=True,
            )
        except MCPToolError as exc:
            if (
                handoff_limited
                and exc.code == MCPErrorCode.TIMEOUT.value
                and invocation_scope is not None
            ):
                raise _handoff_timeout_error(invocation_scope) from exc
            raise

    if not inspect.iscoroutinefunction(bound_method):
        # Python cannot safely cancel a running thread. Tool construction
        # rejects timeout_s for this execution mode, and caller cancellation
        # is delayed until the call has completed so mutations cannot race a
        # retry in the background.
        return await _run_sync_to_completion(bound_method, **call_kwargs)

    pending = bound_method(**call_kwargs)
    effective_timeout = spec.timeout_s
    handoff_limited = False
    if handoff_remaining_s is not None and (
        effective_timeout is None or handoff_remaining_s <= effective_timeout
    ):
        effective_timeout = handoff_remaining_s
        handoff_limited = True
    if effective_timeout is None:
        return await pending
    try:
        return await asyncio.wait_for(pending, timeout=effective_timeout)
    except asyncio.TimeoutError as exc:
        if handoff_limited and invocation_scope is not None:
            raise _handoff_timeout_error(invocation_scope) from exc
        raise MCPToolError(
            f"timed out after {effective_timeout:g}s",
            code=MCPErrorCode.TIMEOUT,
            retryable=True,
        ) from exc


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


async def _run_deferred_worker_to_completion(
    function: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Wait through cancellation so a returned worker staging lease is aborted."""

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as cancelled:
        while True:
            try:
                outcome = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                # Keep ownership of the worker lease across repeated
                # cancellation requests until the process has been reaped.
                continue
            except Exception:  # noqa: BLE001
                outcome = None
                break
        from .jobs import DeferredToolJob

        if isinstance(outcome, DeferredToolJob):
            outcome.abort()
        raise cancelled


def _enforce_output_limit(spec: ToolSpec, value: Any) -> int:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MCPToolError(
            f"returned a value that is not JSON serializable: {exc}",
            code=MCPErrorCode.INTERNAL,
        ) from exc
    output_bytes = len(encoded)
    if spec.max_output_bytes is not None and output_bytes > spec.max_output_bytes:
        raise MCPToolError(
            f"output was {output_bytes} bytes; limit is {spec.max_output_bytes} bytes. "
            "Persist the full result as a run artifact and return its artifact id.",
            code=MCPErrorCode.RESOURCE_LIMIT,
        )
    return output_bytes


def _success_envelope(
    data: Any,
    *,
    duration_ms: float,
    trace: Mapping[str, str | None],
    attempts: int,
    output_bytes: int,
    artifact_ids: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    combined_artifacts = list(dict.fromkeys([*_artifact_ids(data), *(artifact_ids or [])]))
    combined_warnings = list(dict.fromkeys([*_warnings(data), *(warnings or [])]))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "success",
        "data": data,
        "artifact_ids": combined_artifacts,
        "warnings": combined_warnings,
        "error": None,
        "metrics": {
            "duration_ms": round(float(duration_ms), 3),
            "cached": _cache_hit(data),
            "attempts": int(attempts),
            "retries": max(0, int(attempts) - 1),
            "output_bytes": int(output_bytes),
        },
        "trace": dict(trace),
    }


def _error_envelope(
    *,
    normalized: Mapping[str, Any],
    duration_ms: float,
    trace: Mapping[str, str | None],
    attempts: int,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "error",
        "data": None,
        "artifact_ids": [],
        "warnings": [],
        "error": dict(normalized),
        "metrics": {
            "duration_ms": round(float(duration_ms), 3),
            "cached": False,
            "attempts": int(attempts),
            "retries": max(0, int(attempts) - 1),
            "output_bytes": 0,
        },
        "trace": dict(trace),
    }


def _new_tool_trace(ctx: MCPAgentContext) -> dict[str, str | None]:
    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, dict):
        output_context = {}
    trace_id = str(output_context.get("trace_id") or state.get("trace_id") or uuid.uuid4().hex)
    if not output_context.get("trace_id"):
        output_context["trace_id"] = trace_id
    return {
        "run_id": _optional_str(output_context.get("run_id")),
        "trace_id": trace_id,
        "span_id": uuid.uuid4().hex,
        "parent_span_id": _optional_str(
            output_context.get("span_id") or output_context.get("parent_span_id")
        ),
    }


def _refresh_tool_trace(
    ctx: MCPAgentContext,
    trace: Mapping[str, str | None],
) -> dict[str, str | None]:
    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, Mapping):
        return dict(trace)
    return {
        "run_id": _optional_str(output_context.get("run_id")) or trace.get("run_id"),
        "trace_id": (_optional_str(output_context.get("trace_id")) or trace.get("trace_id")),
        "span_id": trace.get("span_id"),
        "parent_span_id": trace.get("parent_span_id"),
    }


def _capture_invocation_scope(
    spec: ToolSpec,
    ctx: MCPAgentContext,
) -> _InvocationScope:
    """Capture authoritative task attribution without granting authorization."""

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    task_id = _optional_str(state.get("active_task_id"))
    role = _optional_str(state.get("active_role"))
    profile = _optional_str(state.get("active_profile")) or _optional_str(state.get("mcp_profile"))
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    task = run.tasks.get(task_id) if run is not None and task_id is not None else None
    if task is not None:
        role = str(task.role)
        profile = str(task.profile)
    if _is_control_plane_spec(spec):
        role = "supervisor"
        profile = _optional_str(state.get("mcp_profile")) or profile
    return _InvocationScope(
        run_id=(str(run.run_id) if run is not None else None),
        task_id=task_id,
        role=role,
        profile=profile,
        task_attempt=(int(task.attempts) if task is not None else None),
    )


def _authorize_invocation(
    spec: ToolSpec,
    ctx: MCPAgentContext,
) -> _InvocationScope:
    """Enforce pinned catalog task, role, and tool boundaries for domain calls."""

    scope = _capture_invocation_scope(spec, ctx)
    if _is_control_plane_spec(spec):
        return scope
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if run is None or run.workflow_slug == "mcp-session":
        return scope
    if getattr(run.status, "value", str(run.status)) != "running":
        raise MCPToolError(
            "a catalog workflow domain tool requires an active RUNNING workflow run",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    workflow_contract = run.workflow_contract if isinstance(run.workflow_contract, Mapping) else {}
    task_contracts = workflow_contract.get("tasks")
    if not isinstance(task_contracts, list) or not task_contracts:
        # Legacy/taskless catalog workflows retain the startup profile boundary.
        return scope

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    task_id = _optional_str(state.get("active_task_id"))
    task = run.tasks.get(task_id) if task_id is not None else None
    if task is None or getattr(task.status, "value", str(task.status)) != "running":
        raise MCPToolError(
            "a catalog workflow domain tool requires an active RUNNING task",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    if spec.roles and task.role not in spec.roles:
        raise MCPToolError(
            f"active role {task.role!r} is not allowed to call {spec.mcp_name}",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    if spec.profiles and task.profile not in spec.profiles:
        raise MCPToolError(
            f"active task profile {task.profile!r} does not expose {spec.mcp_name}",
            code=MCPErrorCode.PERMISSION_DENIED,
        )

    contract = next(
        (
            item
            for item in task_contracts
            if isinstance(item, Mapping) and str(item.get("task_id") or "") == task.task_id
        ),
        None,
    )
    if contract is None:
        raise MCPToolError(
            f"active task {task.task_id!r} is not declared by the pinned workflow contract",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    required_tools = contract.get("required_tools")
    optional_tools = workflow_contract.get("optional_tools")
    allowed_tools = {str(item) for item in required_tools or ()} | {
        str(item) for item in optional_tools or ()
    }
    if spec.mcp_name not in allowed_tools:
        raise MCPToolError(
            f"tool {spec.mcp_name!r} is outside active task {task.task_id!r}'s allowlist",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    handoff = next(
        (item for item in reversed(run.handoffs) if item.task_id == task.task_id),
        None,
    )
    if handoff is None:
        raise MCPToolError(
            f"active task {task.task_id!r} has no structured handoff",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    expected_handoff_attempt = max(0, int(task.attempts) - 1)
    if handoff.task_attempt != expected_handoff_attempt:
        raise MCPToolError(
            f"active task {task.task_id!r} has no handoff for its current attempt",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    budget = handoff.budget if isinstance(handoff.budget, Mapping) else {}
    return _InvocationScope(
        run_id=str(run.run_id),
        task_id=task.task_id,
        role=task.role,
        profile=task.profile,
        task_attempt=int(task.attempts),
        catalog_task=True,
        handoff_id=handoff.handoff_id,
        handoff_created_at=handoff.created_at,
        max_tool_calls=_required_positive_int_budget(budget, "max_tool_calls"),
        timeout_seconds=_required_positive_float_budget(budget, "timeout_seconds"),
    )


def _assert_invocation_epoch_current(
    spec: ToolSpec,
    ctx: MCPAgentContext,
    captured: _InvocationScope,
) -> None:
    """Reject stale catalog results after a task attempt or handoff changes."""

    if not captured.catalog_task:
        return
    run_context = getattr(ctx, "run_context", None)
    if run_context is not None and hasattr(run_context, "refresh"):
        run_context.refresh()
    current = _authorize_invocation(spec, ctx)
    captured_epoch = (
        captured.run_id,
        captured.task_id,
        captured.task_attempt,
        captured.handoff_id,
    )
    current_epoch = (
        current.run_id,
        current.task_id,
        current.task_attempt,
        current.handoff_id,
    )
    if current_epoch != captured_epoch:
        raise MCPToolError(
            "the workflow task attempt or structured handoff changed while the "
            f"{spec.mcp_name} call was running; its stale result was discarded",
            code=MCPErrorCode.PERMISSION_DENIED,
        )


def _record_tool_start(
    *,
    spec: ToolSpec,
    ctx: MCPAgentContext,
    trace: Mapping[str, str | None],
    max_attempts: int,
    invocation_scope: _InvocationScope,
) -> None:
    """Reserve one observable task tool-call budget before execution."""

    from .manifests import record_tool_progress

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
        raise MCPToolError(
            "could not durably reserve the task tool-call budget",
            code=MCPErrorCode.INTERNAL,
        )


def _record_tool_acceptance(
    *,
    spec: ToolSpec,
    ctx: MCPAgentContext,
    trace: Mapping[str, str | None],
    attempts: int,
    max_attempts: int,
    invocation_scope: _InvocationScope,
) -> None:
    """Linearize result acceptance before staged artifacts become visible."""

    from .manifests import record_tool_progress

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
        raise MCPToolError(
            "could not durably accept the task tool result",
            code=MCPErrorCode.INTERNAL,
        )


def _enforce_execution_budget(
    ctx: MCPAgentContext,
    invocation_scope: _InvocationScope,
    *,
    events: typing.Sequence[Any] | None = None,
) -> None:
    run_context = getattr(ctx, "run_context", None)
    selected_events = events if events is not None else getattr(run_context, "events", ())
    handoff_sequence = 0
    for event in selected_events:
        if (
            event.event_type == "handoff_recorded"
            and str(event.payload.get("handoff", {}).get("handoff_id") or "")
            == invocation_scope.handoff_id
        ):
            handoff_sequence = event.sequence
    if handoff_sequence <= 0:
        raise MCPToolError(
            "the active structured handoff is missing from the authoritative event stream",
            code=MCPErrorCode.PERMISSION_DENIED,
        )

    started_calls = sum(
        1
        for event in selected_events
        if event.sequence > handoff_sequence
        and event.event_type == "tool_progress"
        and str(event.payload.get("task_id") or "") == invocation_scope.task_id
        and str(event.payload.get("stage") or "") == "started"
    )
    if (
        invocation_scope.max_tool_calls is not None
        and started_calls >= invocation_scope.max_tool_calls
    ):
        raise MCPToolError(
            f"task {invocation_scope.task_id!r} exhausted its handoff budget of "
            f"{invocation_scope.max_tool_calls} tool calls",
            code=MCPErrorCode.RESOURCE_LIMIT,
        )

    _remaining_handoff_seconds(invocation_scope)


def _remaining_handoff_seconds(invocation_scope: _InvocationScope) -> float | None:
    """Return the live handoff allowance, failing once its deadline is reached."""

    if invocation_scope.timeout_seconds is None or invocation_scope.handoff_created_at is None:
        return None
    created = _parse_utc_timestamp(invocation_scope.handoff_created_at)
    elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    remaining = invocation_scope.timeout_seconds - elapsed
    if remaining <= 0:
        raise _handoff_timeout_error(invocation_scope)
    return remaining


def _handoff_timeout_error(invocation_scope: _InvocationScope) -> MCPToolError:
    return MCPToolError(
        f"task {invocation_scope.task_id!r} exceeded its handoff timeout of "
        f"{invocation_scope.timeout_seconds:g}s",
        code=MCPErrorCode.TIMEOUT,
    )


def _required_positive_int_budget(budget: Mapping[str, Any], name: str) -> int:
    value = budget.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MCPToolError(
            f"catalog handoff budget requires positive integer {name}",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    return value


def _required_positive_float_budget(budget: Mapping[str, Any], name: str) -> float:
    value = budget.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise MCPToolError(
            f"catalog handoff budget requires positive finite {name}",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    return float(value)


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise MCPToolError(
            "catalog handoff has an invalid created_at timestamp",
            code=MCPErrorCode.PERMISSION_DENIED,
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_control_plane_spec(spec: ToolSpec) -> bool:
    return (
        spec.group == "skills"
        or spec.mcp_name == "mcp_bootstrap"
        or spec.mcp_name.startswith("workflow_")
    )


def _run_write_lock(
    spec: ToolSpec,
    ctx: MCPAgentContext,
) -> asyncio.Lock | None:
    """Return an event-loop-local lock for one artifact-producing workflow run."""

    serializes_artifacts = (
        spec.write_scope == "session" and not spec.read_only and not _is_control_plane_spec(spec)
    ) or (spec.result_artifact_type is not None or spec.mcp_name in _ARTIFACT_CONTROL_PLANE_TOOLS)
    if not serializes_artifacts:
        return None
    layout = _active_output_layout(ctx)
    if layout is None:
        return None

    loop = asyncio.get_running_loop()
    key = (layout.session_id, layout.run_id)
    with _RUN_WRITE_LOCKS_GUARD:
        registry = _RUN_WRITE_LOCKS.setdefault(loop, {})
        lock = registry.get(key)
        if lock is None:
            lock = asyncio.Lock()
            registry[key] = lock
        return typing.cast(asyncio.Lock, lock)


@contextmanager
def _tool_write_scope(spec: ToolSpec, ctx: MCPAgentContext):
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


def _enforce_read_boundary(
    spec: ToolSpec,
    public_args: Mapping[str, Any],
    ctx: MCPAgentContext,
) -> _ReadBoundaryResult:
    """Resolve declared scientific file inputs to verified run artifacts."""

    copied = copy.deepcopy(dict(public_args))
    for field_name in spec.read_artifact_fields:
        candidate = copied.get(field_name)
        if candidate is None:
            continue
        copied[field_name] = _confine_registered_read_path(
            candidate,
            ctx=ctx,
            location=(field_name,),
            trusted_pickle=field_name in spec.trusted_pickle_fields,
        )
    copied, pandas_state = _enforce_pandas_read_boundary(spec, copied, ctx)
    copied, report_state = _enforce_report_read_boundary(spec, copied, ctx)
    copied, candidate_state = _enforce_candidate_reference_boundary(spec, copied, ctx)
    session_views = [
        state for state in (pandas_state, report_state, candidate_state) if state is not None
    ]
    if len(session_views) > 1:  # pragma: no cover - tool names are mutually exclusive
        raise MCPToolError(
            f"{spec.mcp_name} requested incompatible session read snapshots",
            code=MCPErrorCode.INTERNAL,
        )
    return _ReadBoundaryResult(
        arguments=copied,
        session_state=session_views[0] if session_views else None,
    )


def _enforce_write_boundary(
    spec: ToolSpec,
    public_args: Mapping[str, Any],
    ctx: MCPAgentContext,
) -> dict[str, Any]:
    """Confine client-selected write destinations to the active run/session.

    Toolkit methods continue to own scientific behavior and storage formats.
    The MCP boundary only recognizes destination-shaped arguments, including
    nested parameter bags, and rewrites safe relative paths into the active
    workflow root. Read paths and workflow control-plane paths are preserved.
    """

    copied = copy.deepcopy(dict(public_args))
    if spec.write_scope != "session" or spec.read_only or _is_control_plane_spec(spec):
        return copied

    _validate_derived_write_names(spec, copied)
    operation = _normalized_operation(public_args)
    writer_operation = _writer_operation(operation)
    _validate_open_ended_writer(spec, copied, operation=operation)
    return typing.cast(
        dict[str, Any],
        _rewrite_write_paths(
            copied,
            ctx=ctx,
            location=(),
            writer_operation=writer_operation,
            path_container=False,
            parameter_bag=False,
        ),
    )


def _validate_derived_write_names(spec: ToolSpec, arguments: Mapping[str, Any]) -> None:
    """Reject state keys that selected toolkit methods later embed in filenames."""

    for field_name in _DERIVED_WRITE_NAME_FIELDS.get(spec.mcp_name, ()):
        value = arguments.get(field_name)
        if not isinstance(value, str) or not _ARTIFACT_TYPE_RE.fullmatch(value):
            raise MCPToolError(
                f"{field_name} must be a safe identifier because {spec.mcp_name} "
                "uses it to derive a run artifact filename",
                code=MCPErrorCode.INVALID_INPUT,
            )


def _normalized_operation(arguments: Mapping[str, Any]) -> str | None:
    operation = arguments.get("operation")
    if not isinstance(operation, str):
        return None
    normalized = operation.strip().lower()
    if normalized.endswith("()"):
        normalized = normalized[:-2]
    normalized = normalized.rsplit(".", 1)[-1]
    return _PANDAS_WRITE_OPERATION_ALIASES.get(normalized, normalized)


def _writer_operation(operation: str | None) -> bool:
    """Return whether an open-ended operation name denotes a serialization sink."""

    if operation is None:
        return False
    return operation.startswith("to_") and operation not in {
        "to_dict",
        "to_numpy",
        "to_records",
    }


def _validate_open_ended_writer(
    spec: ToolSpec,
    arguments: Mapping[str, Any],
    *,
    operation: str | None,
) -> None:
    """Block pandas persistence modes that bypass the session storage adapter."""

    if spec.mcp_name == "pandas_create_dataframe":
        _validate_pandas_creation_function(arguments)
        return
    if spec.mcp_name != "pandas_run_operation" or operation is None:
        return
    if operation in _PANDAS_EXTERNAL_WRITE_OPERATIONS:
        raise MCPToolError(
            f"pandas operation {operation!r} writes outside artifact storage and "
            "is not available through MCP",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    if not _writer_operation(operation) or operation == "to_csv":
        return
    if _contains_write_destination(arguments):
        raise MCPToolError(
            f"pandas operation {operation!r} cannot persist through the session "
            "storage adapter; use to_csv for a run-scoped artifact",
            code=MCPErrorCode.PERMISSION_DENIED,
        )


def _validate_pandas_creation_function(arguments: Mapping[str, Any]) -> None:
    """Allow only DataFrame-producing pandas entry points without write side effects."""

    raw_function, function = _normalized_pandas_creation_function(arguments)
    if function not in _PANDAS_SAFE_CREATE_FUNCTIONS:
        raise MCPToolError(
            f"pandas creation function {raw_function!r} is not available through MCP; "
            "use an approved DataFrame constructor or registered CSV artifact",
            code=MCPErrorCode.PERMISSION_DENIED,
        )


def _normalized_pandas_creation_function(
    arguments: Mapping[str, Any],
) -> tuple[str, str]:
    raw_function = arguments.get("create_using_function")
    if not isinstance(raw_function, str) or not raw_function.strip():
        raise MCPToolError(
            "create_using_function must name an approved DataFrame creation function",
            code=MCPErrorCode.INVALID_INPUT,
        )
    function = raw_function.strip()
    if function.startswith("pd."):
        function = function[3:]
    return raw_function, function


def _enforce_pandas_read_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: MCPAgentContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Require pandas filesystem reads to use verified active-run artifacts."""

    if spec.mcp_name == "pandas_load_dataframe_from_session":
        from cs_copilot.tools.io.session_memory import resolve_loadable_session_data

        session_key = arguments.get("session_key")
        if not isinstance(session_key, str) or not session_key.strip():
            raise MCPToolError(
                "session_key must be a non-empty string",
                code=MCPErrorCode.INVALID_INPUT,
            )
        try:
            resolved = resolve_loadable_session_data(ctx.session_state, session_key)
        except (KeyError, TypeError, ValueError) as exc:
            raise MCPToolError(str(exc), code=MCPErrorCode.INVALID_INPUT) from exc
        resolved_key = str(resolved.get("session_key") or session_key)
        resolved_value = resolved.get("value")
        if resolved.get("kind") == "csv_path":
            pinned_value: Any = _confine_registered_read_path(
                resolved.get("value"),
                ctx=ctx,
                location=("session_state", resolved_key),
            )
        else:
            # A detached frame prevents a concurrent caller from mutating the
            # in-memory object after it has been selected for this invocation.
            pinned_value = resolved_value.copy(deep=True)
        session_snapshot = dict(ctx.session_state)
        # resolve_loadable_session_data checks exact top-level keys first, so a
        # dotted selected key can be pinned without rewriting its source tree.
        session_snapshot[resolved_key] = pinned_value
        arguments["session_key"] = resolved_key
        return arguments, session_snapshot

    if spec.mcp_name == "pandas_create_dataframe":
        _raw_function, function = _normalized_pandas_creation_function(arguments)
        arguments["create_using_function"] = function
        if function not in _PANDAS_FILE_CREATE_FUNCTIONS:
            return arguments, None
        raw_parameters = arguments.get("function_parameters")
        serialized = isinstance(raw_parameters, str)
        if raw_parameters is None:
            parameters: dict[str, Any] = {}
        elif isinstance(raw_parameters, Mapping):
            parameters = dict(raw_parameters)
        elif serialized:
            decoded = _decode_parameter_bag(raw_parameters)
            if not isinstance(decoded, Mapping):
                raise MCPToolError(
                    "function_parameters must decode to an object",
                    code=MCPErrorCode.INVALID_INPUT,
                )
            parameters = dict(decoded)
        else:
            raise MCPToolError(
                "function_parameters must be an object or encoded object",
                code=MCPErrorCode.INVALID_INPUT,
            )

        allowed_keys = set(_PANDAS_FILE_PARAMETER_KEYS[function])
        selected_keys = [key for key in parameters if _normalized_argument_key(key) in allowed_keys]
        if len(selected_keys) != 1:
            raise MCPToolError(
                f"pandas creation function {function!r} requires exactly one "
                "unambiguous registered artifact path",
                code=MCPErrorCode.INVALID_INPUT,
            )
        selected_key = selected_keys[0]
        parameters[selected_key] = _confine_registered_read_path(
            parameters[selected_key],
            ctx=ctx,
            location=("function_parameters", str(selected_key)),
        )
        arguments["function_parameters"] = (
            json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
            if serialized
            else parameters
        )
        return arguments, None

    read_field = {
        "pandas_run_operation": "dataframe_name",
        "pandas_normalize_for_analysis": "df_path",
    }.get(spec.mcp_name)
    if read_field is None:
        return arguments, None
    candidate = arguments.get(read_field)
    if isinstance(candidate, str) and _looks_like_pandas_file_source(candidate):
        arguments[read_field] = _confine_registered_read_path(
            candidate,
            ctx=ctx,
            location=(read_field,),
        )
    return arguments, None


def _enforce_report_read_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: MCPAgentContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Verify every explicit or session-resolved rich-report figure path."""

    if spec.mcp_name != "report_save_rich":
        return arguments, None

    session_snapshot: _SessionStateReadSnapshot | None = None

    def report_session_snapshot() -> _SessionStateReadSnapshot:
        nonlocal session_snapshot
        if session_snapshot is not None:
            return session_snapshot
        live_memory = ctx.session_state.get("session_objects")
        if not isinstance(live_memory, dict):
            live_memory = {}
            ctx.session_state["session_objects"] = live_memory
        live_figures = live_memory.get("figures")
        frozen_figures = (
            copy.deepcopy(dict(live_figures)) if isinstance(live_figures, Mapping) else {}
        )
        memory_snapshot = _SessionStateReadSnapshot(
            live_memory,
            frozen={"figures": frozen_figures},
        )
        session_snapshot = _SessionStateReadSnapshot(
            ctx.session_state,
            frozen={"session_objects": memory_snapshot},
        )
        return session_snapshot

    def normalize_figures(value: Any, *, location: tuple[str, ...]) -> Any:
        if value is None:
            return None
        figures = value if isinstance(value, (list, tuple)) else [value]
        normalized = [
            normalize_figure(figure, location=(*location, str(index)))
            for index, figure in enumerate(figures)
        ]
        return normalized if isinstance(value, (list, tuple)) else normalized[0]

    def normalize_figure(value: Any, *, location: tuple[str, ...]) -> Any:
        if isinstance(value, str):
            return _confine_registered_read_path(value, ctx=ctx, location=location)
        if not isinstance(value, Mapping):
            return value
        figure = copy.deepcopy(dict(value))
        for key in _REPORT_FIGURE_PATH_KEYS:
            candidate = figure.get(key)
            if candidate:
                figure[key] = _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, key),
                )
        normalize_metadata_paths(figure, location=location)
        for metadata_key in ("figure_metadata", "metadata"):
            metadata = figure.get(metadata_key)
            if isinstance(metadata, Mapping):
                metadata_copy = copy.deepcopy(dict(metadata))
                normalize_metadata_paths(
                    metadata_copy,
                    location=(*location, metadata_key),
                )
                figure[metadata_key] = metadata_copy
        validate_session_figure(figure, location=location)
        return figure

    def normalize_metadata_paths(
        metadata: dict[str, Any],
        *,
        location: tuple[str, ...],
    ) -> None:
        paths = metadata.get("paths")
        if not isinstance(paths, Mapping):
            return
        normalized_paths = {}
        for key, candidate in paths.items():
            normalized_paths[key] = (
                _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, "paths", str(key)),
                )
                if candidate
                else candidate
            )
        metadata["paths"] = normalized_paths

    def validate_session_figure(
        figure: Mapping[str, Any],
        *,
        location: tuple[str, ...],
    ) -> None:
        figure_id = figure.get("figure_id")
        if figure_id is None:
            figure_id = figure.get("session_figure_id")
        if figure_id is None or not str(figure_id).strip():
            return
        from cs_copilot.tools.io.figure_metadata import session_figure_metadata

        snapshot = report_session_snapshot()
        metadata = session_figure_metadata(snapshot, figure_id)
        if not metadata:
            return
        paths = metadata.get("paths")
        if not isinstance(paths, Mapping):
            return
        normalized_paths: dict[str, Any] = {}
        for key, candidate in paths.items():
            normalized_paths[str(key)] = (
                _confine_registered_read_path(
                    candidate,
                    ctx=ctx,
                    location=(*location, "session_figure", str(figure_id), str(key)),
                )
                if candidate
                else candidate
            )
        memory = snapshot.get("session_objects")
        figures = memory.get("figures") if isinstance(memory, Mapping) else None
        record = figures.get(str(figure_id)) if isinstance(figures, dict) else None
        if isinstance(record, Mapping):
            pinned_record = copy.deepcopy(dict(record))
            pinned_record["paths"] = normalized_paths
            figures[str(figure_id)] = pinned_record

    if "figures" in arguments:
        arguments["figures"] = normalize_figures(
            arguments.get("figures"),
            location=("figures",),
        )

    sections = arguments.get("sections")
    if isinstance(sections, (list, tuple)):
        normalized_sections = []
        for index, section in enumerate(sections):
            if not isinstance(section, Mapping):
                normalized_sections.append(section)
                continue
            section_copy = copy.deepcopy(dict(section))
            if "figures" in section_copy:
                section_copy["figures"] = normalize_figures(
                    section_copy.get("figures"),
                    location=("sections", str(index), "figures"),
                )
            normalized_sections.append(section_copy)
        arguments["sections"] = normalized_sections
    return arguments, session_snapshot


def _enforce_candidate_reference_boundary(
    spec: ToolSpec,
    arguments: dict[str, Any],
    ctx: MCPAgentContext,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Pin the exact candidate artifact that the selected toolkit will consume."""

    guarded_tools = {
        "peptide_load_design_candidates",
        "session_load_candidate_set_artifact",
        "session_materialize_candidate_set_dataset",
    }
    if spec.mcp_name not in guarded_tools:
        return arguments, None
    default_reference = {
        "peptide_load_design_candidates": "designed_peptides",
        "session_load_candidate_set_artifact": "top candidates",
        "session_materialize_candidate_set_dataset": "generated compounds",
    }[spec.mcp_name]
    reference = arguments.get("reference", default_reference)
    if not isinstance(reference, str) or not reference.strip():
        raise MCPToolError(
            "reference must be a non-empty string",
            code=MCPErrorCode.INVALID_INPUT,
        )
    reference = reference.strip()
    direct_path = _looks_like_candidate_artifact_path(reference)
    selected: tuple[str, tuple[str, ...]] | None = (
        (reference, ("reference",)) if direct_path else None
    )
    candidate_set: Mapping[str, Any] | None = None
    pointer = ctx.session_state.get(reference)
    if selected is None and spec.mcp_name == "peptide_load_design_candidates":
        selected = _first_candidate_path(
            pointer,
            keys=("artifact_rel_path", "artifact_path"),
            location=("session_state", reference),
        )

    if selected is None and spec.mcp_name == "session_load_candidate_set_artifact":
        selected = _first_candidate_path(
            pointer,
            keys=("artifact_rel_path", "artifact_path"),
            location=("session_state", reference),
        )

    if spec.mcp_name.startswith("session_") and selected is None:
        from cs_copilot.tools.io.session_memory import (
            get_session_object,
            resolve_candidate_set,
        )

        if (
            spec.mcp_name == "session_materialize_candidate_set_dataset"
            and isinstance(pointer, Mapping)
            and pointer.get("candidate_set_id")
        ):
            candidate_set = get_session_object(
                ctx.session_state,
                str(pointer["candidate_set_id"]),
            )
        if candidate_set is None:
            resolved = resolve_candidate_set(ctx.session_state, reference)
            candidate_set = resolved.get("candidate_set")
        if isinstance(candidate_set, Mapping):
            if (
                spec.mcp_name == "session_materialize_candidate_set_dataset"
                and arguments.get("top_n") is None
            ):
                selected = _first_candidate_path(
                    candidate_set,
                    keys=("csv_path",),
                    location=("session_candidate_set", reference),
                )
            if selected is None:
                selected = _first_candidate_path(
                    candidate_set,
                    keys=("artifact_rel_path", "artifact_path"),
                    location=("session_candidate_set", reference),
                )

    if (
        spec.mcp_name
        in {
            "peptide_load_design_candidates",
            "session_load_candidate_set_artifact",
        }
        and selected is None
    ):
        raise MCPToolError(
            f"candidate reference {reference!r} does not resolve to a registered run artifact",
            code=MCPErrorCode.INVALID_INPUT,
        )

    if selected is None:
        return arguments, None

    canonical = _confine_registered_read_path(
        selected[0],
        ctx=ctx,
        location=selected[1],
    )
    if not direct_path and spec.mcp_name in {
        "peptide_load_design_candidates",
        "session_load_candidate_set_artifact",
    }:
        pinned_pointer = copy.deepcopy(dict(pointer)) if isinstance(pointer, Mapping) else {}
        for path_key in ("artifact_rel_path", "artifact_path", "csv_path"):
            pinned_pointer.pop(path_key, None)
        pinned_pointer["artifact_rel_path"] = canonical
        if isinstance(candidate_set, Mapping) and candidate_set.get("id"):
            pinned_pointer.setdefault("candidate_set_id", str(candidate_set["id"]))
        session_snapshot = dict(ctx.session_state)
        session_snapshot[reference] = pinned_pointer
        return arguments, session_snapshot

    arguments["reference"] = canonical
    return arguments, None


def _first_candidate_path(
    source: Any,
    *,
    keys: tuple[str, ...],
    location: tuple[str, ...],
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(source, Mapping):
        return None
    for key in keys:
        candidate = source.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate, (*location, key)
    return None


def _looks_like_candidate_artifact_path(value: str) -> bool:
    candidate = value.strip().lower().split("?", 1)[0]
    return _looks_like_pandas_file_source(candidate) or candidate.endswith((".json", ".json.gz"))


def _looks_like_pandas_file_source(value: str) -> bool:
    candidate = value.strip().lower()
    return (
        "/" in candidate
        or "\\" in candidate
        or bool(urlsplit(candidate).scheme)
        or candidate.endswith(_PANDAS_LOADABLE_SUFFIXES)
    )


def _confine_registered_read_path(
    value: Any,
    *,
    ctx: MCPAgentContext,
    location: tuple[str, ...],
    trusted_pickle: bool = False,
) -> str:
    label = _format_argument_location(location)
    if not isinstance(value, str) or not value.strip():
        raise MCPToolError(
            f"pandas read source {label} must be a non-empty path string",
            code=MCPErrorCode.INVALID_INPUT,
        )
    candidate = value.strip()
    layout = _active_output_layout(ctx)
    run_context = getattr(ctx, "run_context", None)
    if run_context is not None and hasattr(run_context, "refresh"):
        run_context.refresh()
    run = getattr(run_context, "run", None)
    if layout is None or run_context is None or run is None:
        raise MCPToolError(
            "pandas filesystem reads require an active workflow run",
            code=MCPErrorCode.PERMISSION_DENIED,
        )

    record = next(
        (
            artifact
            for artifact in run.artifacts.values()
            if _matches_registered_read_alias(
                candidate,
                run_relative=artifact.relative_path,
                run_scoped=layout.artifact_rel_path(artifact.relative_path),
            )
        ),
        None,
    )
    if record is None:
        raise MCPToolError(
            f"read source {label} is not a registered artifact in the active run",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    workflow_contract = run.workflow_contract if isinstance(run.workflow_contract, Mapping) else {}
    task_contracts = workflow_contract.get("tasks")
    if isinstance(task_contracts, list) and task_contracts:
        state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
        task_id = _optional_str(state.get("active_task_id"))
        task = run.tasks.get(task_id) if task_id is not None else None
        if task is None or getattr(task.status, "value", str(task.status)) != "running":
            raise MCPToolError(
                "catalog workflow artifact reads require an active RUNNING task",
                code=MCPErrorCode.PERMISSION_DENIED,
            )
        allowed_artifact_ids = set(task.input_artifact_ids)
        allowed_artifact_ids.update(
            artifact.artifact_id
            for artifact in run.artifacts.values()
            if artifact.producer_task_id == task.task_id
        )
        if record.artifact_id not in allowed_artifact_ids:
            raise MCPToolError(
                f"read source {label} was not handed off to active task " f"{task.task_id!r}",
                code=MCPErrorCode.PERMISSION_DENIED,
            )
    if trusted_pickle and (
        getattr(record.trust, "value", str(record.trust)) != "internal"
        or record.producer_tool != "gtm_save_model_and_data"
        or record.artifact_type != "gtm_model_path"
        or not record.relative_path.lower().endswith(".pkl.gz")
    ):
        raise MCPToolError(
            f"read source {label} is executable serialized model content; only "
            "an internal gtm_model_path produced by gtm_save_model_and_data is accepted",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    run_context.verify_artifact(record.artifact_id)
    return layout.artifact_rel_path(record.relative_path)


def _matches_registered_read_alias(
    candidate: str,
    *,
    run_relative: str,
    run_scoped: str,
) -> bool:
    """Match public, storage-expanded, or absolute forms of one artifact path."""

    from cs_copilot.storage import S3

    expanded = S3.path(run_scoped)
    if candidate in {run_relative, run_scoped, expanded}:
        return True

    candidate_url = urlsplit(candidate)
    expanded_url = urlsplit(expanded)
    if candidate_url.scheme or expanded_url.scheme:
        if (
            candidate_url.scheme.lower() == "s3"
            and expanded_url.scheme.lower() == "s3"
            and not candidate_url.query
            and not candidate_url.fragment
        ):
            return candidate_url.netloc == expanded_url.netloc and unquote(
                candidate_url.path
            ) == unquote(expanded_url.path)
        if candidate_url.scheme.lower() != "file" or expanded_url.scheme:
            return False
        candidate_path = Path(unquote(candidate_url.path))
    else:
        candidate_path = Path(candidate)

    if not candidate_path.is_absolute() and ".." in candidate_path.parts:
        return False
    try:
        return candidate_path.resolve(strict=False) == Path(expanded).resolve(strict=False)
    except (OSError, RuntimeError):
        return False


def _contains_write_destination(value: Any, *, parameter_bag: bool = False) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = _normalized_argument_key(key)
            if normalized_key in _WRITE_PATH_KEYS:
                return child is not None
            if parameter_bag and normalized_key in _GENERIC_PATH_KEYS:
                return child is not None
            if _contains_write_destination(
                child,
                parameter_bag=(parameter_bag or normalized_key in _PARAMETER_BAG_KEYS),
            ):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_write_destination(child, parameter_bag=parameter_bag) for child in value
        )
    if parameter_bag and isinstance(value, str) and value.lstrip().startswith(("{", "[")):
        decoded = _decode_parameter_bag(value)
        if decoded is None:
            return False
        return _contains_write_destination(decoded, parameter_bag=True)
    return False


def _rewrite_write_paths(
    value: Any,
    *,
    ctx: MCPAgentContext,
    location: tuple[str, ...],
    writer_operation: bool,
    path_container: bool,
    parameter_bag: bool,
) -> Any:
    if isinstance(value, Mapping):
        rewritten: dict[Any, Any] = {}
        for key, child in value.items():
            normalized_key = _normalized_argument_key(key)
            child_location = (*location, str(key))
            strong_sink = normalized_key in _WRITE_PATH_KEYS
            generic_sink = normalized_key in _GENERIC_PATH_KEYS and (
                path_container or (parameter_bag and writer_operation)
            )
            child_container = path_container or normalized_key in _WRITE_PATH_CONTAINER_KEYS
            child_parameter_bag = parameter_bag or normalized_key in _PARAMETER_BAG_KEYS

            if strong_sink or generic_sink:
                rewritten[key] = _rewrite_path_sink(
                    child,
                    ctx=ctx,
                    location=child_location,
                )
                continue

            if (
                child_parameter_bag
                and isinstance(child, str)
                and child.lstrip().startswith(("{", "["))
            ):
                rewritten[key] = _rewrite_json_parameter_bag(
                    child,
                    ctx=ctx,
                    location=child_location,
                    writer_operation=writer_operation,
                )
                continue

            rewritten[key] = _rewrite_write_paths(
                child,
                ctx=ctx,
                location=child_location,
                writer_operation=writer_operation,
                path_container=child_container,
                parameter_bag=child_parameter_bag,
            )
        return rewritten

    if isinstance(value, list):
        return [
            _rewrite_write_paths(
                child,
                ctx=ctx,
                location=(*location, str(index)),
                writer_operation=writer_operation,
                path_container=path_container,
                parameter_bag=parameter_bag,
            )
            for index, child in enumerate(value)
        ]

    if isinstance(value, tuple):
        return tuple(
            _rewrite_write_paths(
                child,
                ctx=ctx,
                location=(*location, str(index)),
                writer_operation=writer_operation,
                path_container=path_container,
                parameter_bag=parameter_bag,
            )
            for index, child in enumerate(value)
        )

    if path_container and isinstance(value, str):
        return _confine_write_path(value, ctx=ctx, location=location)
    return value


def _rewrite_json_parameter_bag(
    value: str,
    *,
    ctx: MCPAgentContext,
    location: tuple[str, ...],
    writer_operation: bool,
) -> str:
    decoded = _decode_parameter_bag(value)
    if decoded is None:
        return value
    rewritten = _rewrite_write_paths(
        decoded,
        ctx=ctx,
        location=location,
        writer_operation=writer_operation,
        path_container=False,
        parameter_bag=True,
    )
    return json.dumps(rewritten, ensure_ascii=False, separators=(",", ":"))


def _decode_parameter_bag(value: str) -> Any | None:
    """Decode JSON or Python-literal tool parameter bags without executing code."""

    for parser in (json.loads, ast.literal_eval):
        try:
            decoded = parser(value)
        except (TypeError, ValueError, SyntaxError):
            continue
        if isinstance(decoded, (Mapping, list, tuple)):
            return decoded
        return None
    return None


def _rewrite_path_sink(
    value: Any,
    *,
    ctx: MCPAgentContext,
    location: tuple[str, ...],
) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        return _rewrite_write_paths(
            value,
            ctx=ctx,
            location=location,
            writer_operation=True,
            path_container=True,
            parameter_bag=False,
        )
    if not isinstance(value, str):
        raise MCPToolError(
            f"write destination {_format_argument_location(location)} must be a path string",
            code=MCPErrorCode.INVALID_INPUT,
        )
    return _confine_write_path(value, ctx=ctx, location=location)


def _confine_write_path(
    value: str,
    *,
    ctx: MCPAgentContext,
    location: tuple[str, ...],
) -> str:
    from cs_copilot.storage import S3
    from cs_copilot.storage.layout import normalize_run_relative_path

    candidate = value.strip()
    label = _format_argument_location(location)
    if not candidate:
        raise MCPToolError(
            f"write destination {label} cannot be empty",
            code=MCPErrorCode.INVALID_INPUT,
        )
    if "\x00" in candidate or "\\" in candidate:
        raise _write_boundary_error(label)
    if _ENCODED_PATH_SEPARATOR_RE.search(candidate):
        raise _write_boundary_error(label)

    layout = _active_output_layout(ctx)
    run_prefix = layout.run_root if layout is not None else ""
    parsed = urlsplit(candidate)
    if parsed.scheme:
        if parsed.scheme.lower() != "s3":
            raise _write_boundary_error(label)
        _validate_scoped_s3_destination(
            candidate,
            run_prefix=run_prefix,
            label=label,
        )
        return candidate

    if candidate.startswith(("/", "file://")) or PurePosixPath(candidate).is_absolute():
        raise _write_boundary_error(label)

    if layout is not None:
        try:
            relative = normalize_run_relative_path(layout.run_id, candidate)
        except ValueError as exc:
            raise _write_boundary_error(label) from exc
        confined = layout.artifact_rel_path(relative)
    else:
        confined = _normalize_session_relative_path(candidate, label=label)

    _validate_local_destination(
        S3,
        confined=confined,
        run_prefix=run_prefix,
        label=label,
    )
    return confined


def _active_output_layout(ctx: MCPAgentContext):
    from cs_copilot.storage.layout import OutputLayout

    run_context = getattr(ctx, "run_context", None)
    layout = getattr(run_context, "layout", None)
    if isinstance(layout, OutputLayout):
        _validate_storage_session(layout.session_id)
        return layout

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    output_context = state.get("output_context")
    if not isinstance(output_context, Mapping):
        return None
    required = ("session_id", "run_id", "workflow_slug")
    if not all(output_context.get(field) for field in required):
        return None
    try:
        layout = OutputLayout(
            session_id=str(output_context["session_id"]),
            run_id=str(output_context["run_id"]),
            workflow_slug=str(output_context["workflow_slug"]),
        )
    except ValueError as exc:
        raise MCPToolError(
            f"active output context is invalid: {exc}",
            code=MCPErrorCode.PERMISSION_DENIED,
        ) from exc
    _validate_storage_session(layout.session_id)
    return layout


def _validate_storage_session(session_id: str) -> None:
    from cs_copilot.storage import S3

    active_session = PurePosixPath(S3.current_prefix().strip("/")).name
    if active_session != session_id:
        raise MCPToolError(
            "active storage session does not match the workflow run",
            code=MCPErrorCode.PERMISSION_DENIED,
        )


def _normalize_session_relative_path(candidate: str, *, label: str) -> str:
    path = PurePosixPath(candidate)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise _write_boundary_error(label)
    if not path.parts:
        raise MCPToolError(
            f"write destination {label} cannot be empty",
            code=MCPErrorCode.INVALID_INPUT,
        )
    return path.as_posix()


def _validate_scoped_s3_destination(
    candidate: str,
    *,
    run_prefix: str,
    label: str,
) -> None:
    from cs_copilot.storage import S3

    parsed = urlsplit(candidate)
    if parsed.query or parsed.fragment or not parsed.netloc:
        raise _write_boundary_error(label)
    decoded_path = unquote(parsed.path)
    if "\\" in decoded_path:
        raise _write_boundary_error(label)
    path_parts = PurePosixPath(decoded_path).parts
    if any(part in {"", ".", ".."} for part in path_parts):
        raise _write_boundary_error(label)

    expected = urlsplit(S3.path(run_prefix))
    if expected.scheme.lower() != "s3" or not expected.netloc:
        raise _write_boundary_error(label)
    expected_path = expected.path.rstrip("/")
    if (
        parsed.netloc != expected.netloc
        or not decoded_path.startswith(f"{expected_path}/")
        or decoded_path == expected_path
    ):
        raise _write_boundary_error(label)


def _validate_local_destination(
    storage: Any,
    *,
    confined: str,
    run_prefix: str,
    label: str,
) -> None:
    storage_root = storage.path("")
    destination = storage.path(confined)
    if urlsplit(storage_root).scheme == "s3":
        return
    if urlsplit(destination).scheme:
        raise _write_boundary_error(label)

    session_root = Path(storage_root).resolve(strict=False)
    boundary_root = Path(storage.path(run_prefix)).resolve(strict=False)
    destination_path = Path(destination).resolve(strict=False)
    try:
        boundary_root.relative_to(session_root)
        destination_path.relative_to(boundary_root)
    except ValueError as exc:
        raise _write_boundary_error(label) from exc


def _normalized_argument_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _format_argument_location(location: tuple[str, ...]) -> str:
    return ".".join(location) or "<argument>"


def _write_boundary_error(label: str) -> MCPToolError:
    return MCPToolError(
        f"write destination {label} must remain inside the active workflow run/session; "
        "absolute paths, file:// URLs, traversal, and foreign S3 locations are not allowed",
        code=MCPErrorCode.PERMISSION_DENIED,
    )


def _retry_delay(spec: ToolSpec, attempt: int) -> float:
    multiplier = 2 ** min(max(0, int(attempt) - 1), 10)
    return min(float(spec.retry_backoff_s) * multiplier, 30.0)


def _validate_idempotency_key(value: Any, spec: ToolSpec) -> str | None:
    if value is None:
        return None
    if not spec.idempotent:
        raise MCPToolError(
            "idempotency_key is only supported for idempotent tools",
            code=MCPErrorCode.INVALID_INPUT,
        )
    if not isinstance(value, str) or not value.strip():
        raise MCPToolError(
            "idempotency_key must be a non-empty string",
            code=MCPErrorCode.INVALID_INPUT,
        )
    normalized = value.strip()
    if len(normalized) > 256:
        raise MCPToolError(
            "idempotency_key cannot exceed 256 characters",
            code=MCPErrorCode.INVALID_INPUT,
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


def _idempotency_cache(ctx: MCPAgentContext) -> OrderedDict:
    cache = getattr(ctx, "_mcp_idempotency_cache", None)
    if not isinstance(cache, OrderedDict):
        cache = OrderedDict()
        ctx._mcp_idempotency_cache = cache
    return cache


def _idempotency_digests(ctx: MCPAgentContext) -> OrderedDict:
    digests = getattr(ctx, "_mcp_idempotency_digests", None)
    if not isinstance(digests, OrderedDict):
        digests = OrderedDict()
        ctx._mcp_idempotency_digests = digests
    return digests


def _idempotency_inflight(
    ctx: MCPAgentContext,
) -> dict[tuple[str, str, str, str, str], _IdempotencyReservation]:
    inflight = getattr(ctx, "_mcp_idempotency_inflight", None)
    if not isinstance(inflight, dict):
        inflight = {}
        ctx._mcp_idempotency_inflight = inflight
    return inflight


def _idempotency_mutex(ctx: MCPAgentContext) -> threading.RLock:
    mutex = getattr(ctx, "_mcp_idempotency_mutex", None)
    if mutex is None:
        mutex = threading.RLock()
        ctx._mcp_idempotency_mutex = mutex
    return mutex


def _reserve_idempotency(
    ctx: MCPAgentContext,
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
            raise MCPToolError(
                "idempotency_key was already used with different arguments",
                code=MCPErrorCode.INVALID_INPUT,
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
                raise MCPToolError(
                    "idempotency_key was already used with different arguments",
                    code=MCPErrorCode.INVALID_INPUT,
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
        raise MCPToolError(
            "idempotency reservation is missing its in-flight result",
            code=MCPErrorCode.INTERNAL,
        )
    return await asyncio.shield(asyncio.wrap_future(reservation.future))


def _finish_idempotency(
    ctx: MCPAgentContext,
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
    ctx: MCPAgentContext,
    reservation: _IdempotencyReservation,
) -> None:
    """Release waiters when an owning MCP task is externally cancelled."""

    with _idempotency_mutex(ctx):
        inflight = _idempotency_inflight(ctx)
        current = inflight.get(reservation.identity)
        if current is not reservation:
            return
        inflight.pop(reservation.identity, None)
        future = reservation.future
        if future is not None and not future.done():
            future.set_exception(
                MCPToolError(
                    "the owning idempotent call was cancelled before publishing a result",
                    code=MCPErrorCode.TRANSIENT_EXTERNAL,
                    retryable=True,
                )
            )


def _trim_idempotency_digests(ctx: MCPAgentContext) -> None:
    digests = _idempotency_digests(ctx)
    inflight = _idempotency_inflight(ctx)
    while len(digests) > MAX_IDEMPOTENCY_ENTRIES:
        removable = next((identity for identity in digests if identity not in inflight), None)
        if removable is None:
            return
        digests.pop(removable, None)


def _cache_lookup_unlocked(
    ctx: MCPAgentContext,
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
        raise MCPToolError(
            "idempotency_key was already used with different arguments",
            code=MCPErrorCode.INVALID_INPUT,
        )
    return copy.deepcopy(entry["envelope"])


def _cache_store(
    ctx: MCPAgentContext,
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


def _cached_envelope(
    envelope: Mapping[str, Any],
    *,
    duration_ms: float,
    trace: Mapping[str, str | None],
) -> dict[str, Any]:
    cached = copy.deepcopy(dict(envelope))
    metrics = dict(cached.get("metrics") or {})
    metrics.update(
        {
            "duration_ms": round(float(duration_ms), 3),
            "cached": True,
            "attempts": 0,
            "retries": 0,
        }
    )
    cached["metrics"] = metrics
    cached["trace"] = dict(trace)
    return cached


def _register_result_artifacts(
    spec: ToolSpec,
    value: Any,
    ctx: MCPAgentContext,
    *,
    active_task_id: str | None,
    invocation_span_id: str | None,
    publication_leases: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    from cs_copilot.workflows import ArtifactIntegrityError

    artifact_ids = _artifact_ids(value)
    warnings: list[str] = []
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if run_context is None or run is None:
        return artifact_ids, warnings

    existing_by_path = {
        record.relative_path: record.artifact_id for record in run.artifacts.values()
    }
    if active_task_id not in run.tasks:
        active_task_id = None
    required_output_types = _required_task_output_types(run, active_task_id)
    owned_publications = dict(publication_leases)

    if spec.result_artifact_type is not None:
        materialized_publications: dict[str, dict[str, Any]] = {}
        try:
            relative, materialized_publications = _materialize_result_json(
                run_context,
                artifact_type=spec.result_artifact_type,
                value=value,
            )
            owned_publications.update(materialized_publications)
            artifact_id = _register_result_path(
                spec,
                run_context,
                existing_by_path,
                relative=relative,
                artifact_type=spec.result_artifact_type,
                mime_type="application/json",
                active_task_id=active_task_id,
                result_field="structured_result",
                invocation_span_id=invocation_span_id,
                publication_leases=owned_publications,
            )
            artifact_ids.append(artifact_id)
        except ArtifactIntegrityError:
            _rollback_unregistered_publications(ctx, materialized_publications)
            raise
        except Exception as exc:  # noqa: BLE001
            _rollback_unregistered_publications(ctx, materialized_publications)
            if spec.result_artifact_type in required_output_types:
                raise MCPToolError(
                    f"required artifact {spec.result_artifact_type!r} could not "
                    f"be registered for task {active_task_id!r}: {exc}",
                    code=MCPErrorCode.INTERNAL,
                ) from exc
            warning = (
                f"Could not materialize structured result as "
                f"{spec.result_artifact_type!r}: {exc}"
            )
            warnings.append(warning)
            logger.warning("%s: %s", spec.mcp_name, warning)

    for field_name, candidate in _result_paths(value):
        relative = _run_relative_result_path(run_context, candidate)
        if relative is None:
            continue
        artifact_type = _infer_artifact_type(spec, field_name, relative)
        try:
            artifact_id = _register_result_path(
                spec,
                run_context,
                existing_by_path,
                relative=relative,
                artifact_type=artifact_type,
                mime_type=_infer_mime_type(relative),
                active_task_id=active_task_id,
                result_field=field_name,
                invocation_span_id=invocation_span_id,
                publication_leases=owned_publications,
            )
            artifact_ids.append(artifact_id)
        except ArtifactIntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001
            if artifact_type in required_output_types:
                raise MCPToolError(
                    f"required artifact {artifact_type!r} at {relative!r} could "
                    f"not be registered for task {active_task_id!r}: {exc}",
                    code=MCPErrorCode.INTERNAL,
                ) from exc
            warning = (
                f"Could not register run-scoped result path {relative!r} " f"as an artifact: {exc}"
            )
            warnings.append(warning)
            logger.warning("%s: %s", spec.mcp_name, warning)

    return (
        list(dict.fromkeys(artifact_ids)),
        list(dict.fromkeys(warnings)),
    )


def _rollback_unregistered_publications(
    ctx: MCPAgentContext,
    publications: Mapping[str, Mapping[str, Any]],
) -> None:
    """Release only invocation-owned bytes that have no durable artifact event."""

    if not publications:
        return
    from cs_copilot.storage import S3

    run_context = getattr(ctx, "run_context", None)
    if run_context is None:
        return
    try:
        run = run_context.refresh()
        registered = {
            run_context.layout.artifact_rel_path(record.relative_path)
            for record in run.artifacts.values()
        }
    except Exception:
        logger.warning(
            "Could not establish authoritative artifact state; publication " "rollback was skipped",
            exc_info=True,
        )
        return
    releasable = {
        path: metadata for path, metadata in publications.items() if path not in registered
    }
    try:
        S3.rollback_promoted_publications(releasable)
    except Exception:
        logger.warning(
            "Could not release unregistered invocation publications",
            exc_info=True,
        )


def _register_result_path(
    spec: ToolSpec,
    run_context: Any,
    existing_by_path: dict[str, str],
    *,
    relative: str,
    artifact_type: str,
    mime_type: str,
    active_task_id: str | None,
    result_field: str,
    invocation_span_id: str | None,
    publication_leases: Mapping[str, Mapping[str, Any]],
) -> str:
    existing_id = existing_by_path.get(relative)
    if existing_id is not None:
        existing = run_context.verify_artifact(existing_id)
        expected_trust = "external" if spec.requires_network else "internal"
        existing_trust = getattr(existing.trust, "value", str(existing.trust))
        existing_provenance = (
            existing.provenance if isinstance(existing.provenance, Mapping) else {}
        )
        if (
            existing.artifact_type != artifact_type
            or existing.mime_type != mime_type
            or existing.producer_task_id != active_task_id
            or existing.producer_tool != spec.mcp_name
            or existing_trust != expected_trust
            or existing_provenance.get("registration") != "automatic"
            or existing_provenance.get("result_field") != result_field
        ):
            raise ValueError(
                f"existing artifact {existing_id!r} at {relative!r} does not "
                "match this tool result's type, producer, trust, or provenance"
            )
        return existing_id
    storage_key = run_context.layout.artifact_rel_path(relative)
    if storage_key not in publication_leases:
        from cs_copilot.workflows import ArtifactIntegrityError

        raise ArtifactIntegrityError(
            f"unregistered result path {relative!r} was not published by this " "tool invocation"
        )
    record = run_context.register_artifact(
        relative,
        artifact_type=artifact_type,
        mime_type=mime_type,
        producer_task_id=active_task_id,
        active_task_id=active_task_id,
        producer_tool=spec.mcp_name,
        provenance={
            "registration": "automatic",
            "result_field": result_field,
            "invocation_span_id": invocation_span_id,
        },
        trust="external" if spec.requires_network else "internal",
    )
    existing_by_path[record.relative_path] = record.artifact_id
    return record.artifact_id


def _required_task_output_types(run: Any, task_id: str | None) -> frozenset[str]:
    """Return the pinned output contracts for one active catalog task."""

    if task_id is None:
        return frozenset()
    contract = getattr(run, "workflow_contract", None)
    if not isinstance(contract, Mapping):
        return frozenset()
    tasks = contract.get("tasks")
    if not isinstance(tasks, list):
        return frozenset()
    globally_required = {
        str(item.get("name"))
        for item in contract.get("output_artifacts", ())
        if isinstance(item, Mapping) and item.get("name") and item.get("required", True)
    }
    for task in tasks:
        if not isinstance(task, Mapping) or str(task.get("task_id") or "") != task_id:
            continue
        outputs = task.get("output_artifacts")
        if not isinstance(outputs, (list, tuple)):
            return frozenset()
        return frozenset(
            str(item) for item in outputs if str(item).strip() and str(item) in globally_required
        )
    return frozenset()


def _materialize_result_json(
    run_context: Any,
    *,
    artifact_type: str,
    value: Any,
) -> tuple[str, dict[str, dict[str, Any]]]:
    from cs_copilot.storage import S3

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    encoded = f"{serialized}\n".encode("utf-8")
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    relative = f"artifacts/contracts/{artifact_type}-{digest[:16]}.json"
    storage_path = run_context.layout.artifact_rel_path(relative)
    created = False
    try:
        with S3.open_atomic(storage_path, "x") as handle:
            handle.write(serialized)
            handle.write("\n")
        created = True
    except (FileExistsError, PermissionError):
        # The deterministic content-addressed path may already have been
        # materialized by an idempotent call. Reuse it only when the pinned
        # bytes are exactly the result being returned.
        from cs_copilot.workflows import ArtifactIntegrityError

        try:
            with S3._open_verified_snapshot(
                storage_path,
                "rb",
                expected_sha256=hashlib.sha256(encoded).hexdigest(),
                expected_size=len(encoded),
            ):
                pass
        except Exception as integrity_exc:
            raise ArtifactIntegrityError(
                f"result artifact path {relative!r} already exists with "
                "different or unverifiable content"
            ) from integrity_exc
    publications = (
        {
            storage_path: {
                "staged_path": storage_path,
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "size_bytes": len(encoded),
            }
        }
        if created
        else {}
    )
    return relative, publications


def _result_paths(value: Any, *, depth: int = 0):
    if depth > 6:
        return
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if _is_path_result_field(key):
                if isinstance(item, (str, Path)):
                    if str(item).strip():
                        yield key, str(item)
                elif isinstance(item, (list, tuple, set)):
                    for candidate in item:
                        if isinstance(candidate, (str, Path)) and str(candidate).strip():
                            yield key, str(candidate)
            if isinstance(item, (Mapping, list, tuple, set)):
                yield from _result_paths(item, depth=depth + 1)
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            if isinstance(item, (Mapping, list, tuple, set)):
                yield from _result_paths(item, depth=depth + 1)
    elif isinstance(value, str):
        labeled_paths: set[str] = set()
        for matched in _KNOWN_LABELED_BACKTICK_RE.finditer(value):
            candidate = matched.group("path").strip()
            labeled_paths.add(candidate)
            yield _labeled_path_field(matched.group("label")), candidate
        for candidate in _BACKTICK_PATH_RE.findall(value):
            normalized = candidate.strip()
            if normalized and normalized not in labeled_paths:
                yield "backticked_path", normalized
        saved_paths = _GTM_SAVED_PATHS_RE.fullmatch(value)
        if saved_paths is not None:
            yield "dataset_path", saved_paths.group("dataset").strip()
            yield "gtm_path", saved_paths.group("gtm").strip()


def _labeled_path_field(label: str) -> str:
    normalized = label.strip().lower()
    if normalized.startswith("clean dataset"):
        return "clean_dataset_path"
    known = {
        "raw dataset": "raw_dataset_path",
        "descriptor parquet": "descriptor_parquet_path",
        "standardization report": "standardization_report_path",
        "filtered rows": "filtered_rows_path",
        "html": "html_path",
        "pdf": "pdf_path",
        "markdown": "markdown_path",
    }
    return known.get(normalized, "backticked_path")


def _is_path_result_field(field_name: str) -> bool:
    normalized = field_name.strip().lower()
    return normalized in {"path", "paths", "file", "files"} or normalized.endswith(
        ("_path", "_paths")
    )


def _run_relative_result_path(run_context: Any, candidate: str) -> str | None:
    from cs_copilot.storage import S3, normalize_run_relative_path

    run_id = run_context.layout.run_id
    value = str(candidate).strip()
    run_root = S3.path(run_context.layout.run_root).rstrip("/")
    comparable_value = value[7:] if value.startswith("file://") else value
    comparable_root = run_root[7:] if run_root.startswith("file://") else run_root
    prefix = f"{comparable_root}/"
    if comparable_value.startswith(prefix):
        try:
            return normalize_run_relative_path(run_id, comparable_value[len(prefix) :])
        except ValueError:
            return None
    try:
        return normalize_run_relative_path(run_id, value)
    except ValueError:
        return None


def _infer_artifact_type(
    spec: ToolSpec,
    field_name: str,
    relative_path: str,
) -> str:
    normalized = field_name.strip().lower()
    suffix = Path(relative_path).suffix.lower()
    if spec.mcp_name == "gtm_save_model_and_data":
        contract_types = {
            "dataset_path": "projected_dataset_path",
            "gtm_path": "gtm_model_path",
        }
        if normalized in contract_types:
            return contract_types[normalized]
    if spec.mcp_name == "gtm_create_activity_landscapes":
        if suffix == ".csv":
            return "activity_landscape_csv"
        if suffix in {".html", ".png", ".jpg", ".jpeg", ".svg", ".webp"}:
            return "activity_plot_path"
    if spec.mcp_name == "gtm_save_density_plot" and suffix in {
        ".html",
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
        ".webp",
    }:
        return "density_plot_path"
    if spec.mcp_name == "report_save_rich":
        report_types = {
            ".html": "html_report_path",
            ".pdf": "pdf_report_path",
            ".md": "markdown_report_path",
        }
        if suffix in report_types:
            return report_types[suffix]
    if spec.mcp_name == "report_save_markdown" and suffix == ".md":
        return "markdown_report_path"
    if normalized.endswith("_paths"):
        normalized = f"{normalized[:-6]}_path"
    generic = {
        "path",
        "paths",
        "file",
        "files",
        "relative_path",
        "artifact_path",
        "artifact_rel_path",
    }
    if normalized and normalized not in generic:
        return normalized
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".webp"}:
        return "visualization"
    if suffix in {".csv", ".tsv", ".parquet", ".feather"}:
        return "dataset"
    if suffix in {".html", ".md", ".pdf", ".docx"}:
        return "report"
    if suffix in {".pkl", ".pickle", ".joblib", ".onnx", ".pt", ".pth"}:
        return "model"
    return "artifact"


def _infer_mime_type(relative_path: str) -> str:
    suffix = Path(relative_path).suffix.lower()
    overrides = {
        ".jsonl": "application/x-ndjson",
        ".parquet": "application/vnd.apache.parquet",
        ".pkl": "application/x-python-pickle",
        ".pickle": "application/x-python-pickle",
    }
    return (
        overrides.get(suffix)
        or mimetypes.guess_type(relative_path)[0]
        or ("application/octet-stream")
    )


def _artifact_ids(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    candidates: list[Any] = []
    plural = value.get("artifact_ids")
    if isinstance(plural, (list, tuple, set)):
        candidates.extend(plural)
    singular = value.get("artifact_id")
    if singular:
        candidates.append(singular)
    for key, item in value.items():
        if str(key).endswith("_artifact_id") and item:
            candidates.append(item)
    return list(dict.fromkeys(str(item) for item in candidates if str(item).strip()))


def _warnings(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    raw = value.get("warnings")
    if isinstance(raw, (list, tuple, set)):
        warnings = [str(item) for item in raw if str(item).strip()]
    elif raw:
        warnings = [str(raw)]
    else:
        warning = value.get("warning")
        warnings = [str(warning)] if warning else []
    return list(dict.fromkeys(warnings))


def _cache_hit(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return bool(value.get("cached") or value.get("cache_hit") or value.get("_cache_hit"))


def _optional_str(value: Any) -> str | None:
    return str(value) if value not in (None, "") else None
