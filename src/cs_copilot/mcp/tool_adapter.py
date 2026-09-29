"""Adapt cs_copilot toolkit methods into bounded, typed MCP v2 tools.

The runtime-neutral execution pipeline (authorization, read and write
boundaries, idempotency, artifact registration, durable tool events, and result
envelopes) lives in :mod:`cs_copilot.execution`, where the in-process runtime
can share it. This module owns what is specific to MCP: the public tool
signature, the ``idempotency_key`` parameter, injection of the MCP agent shim,
and subprocess worker dispatch. Scientific logic remains in the toolkit
methods themselves.
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import replace
from typing import Any, Callable, Dict, Mapping, Optional

from cs_copilot.execution.envelopes import RESULT_SCHEMA_VERSION
from cs_copilot.execution.idempotency import MAX_IDEMPOTENCY_ENTRIES
from cs_copilot.execution.kernel import AttemptContext, ExecutedCall
from cs_copilot.execution.runner import execute_async, in_process_async_executor
from cs_copilot.execution.scope import _handoff_timeout_error
from cs_copilot.execution.spec import (
    DEFAULT_MAX_OUTPUT_BYTES,
    INJECTED_PARAMS,
    ToolSpec,
    _public_parameters,
)

from .context import MCPAgentContext
from .errors import MCPErrorCode, MCPToolError

__all__ = [
    "DEFAULT_MAX_OUTPUT_BYTES",
    "INJECTED_PARAMS",
    "MAX_IDEMPOTENCY_ENTRIES",
    "RESULT_SCHEMA_VERSION",
    "ToolSpec",
    "build_tool",
]


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

    if spec.run_in_worker_process:
        # Worker jobs rebuild their own context in the child process.
        inject = None
        executor = _worker_executor(spec, ctx)
    else:
        inject = _mcp_injector(sig, ctx)
        executor = in_process_async_executor(spec, bound_method)

    async def _invoke(**kwargs: Any) -> dict[str, Any]:
        outcome = await execute_async(
            spec,
            ctx,
            kwargs,
            executor=executor,
            inject=inject,
        )
        return outcome.envelope

    _invoke.__name__ = spec.mcp_name
    _invoke.__qualname__ = spec.mcp_name
    _invoke.__doc__ = spec.summary or (bound_method.__doc__ or "").strip()
    _invoke.__signature__ = public_signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = public_annotations
    _invoke.__wrapped__ = bound_method  # type: ignore[attr-defined]
    return _invoke


def _mcp_injector(sig: inspect.Signature, ctx: MCPAgentContext):
    """Inject the MCP agent shim and session state the toolkit method declares."""

    def inject(call_kwargs: Dict[str, Any], session_view: Optional[Dict[str, Any]]) -> None:
        if "agent" in sig.parameters:
            if session_view is None:
                call_kwargs["agent"] = ctx
            else:
                invocation_agent = copy.copy(ctx)
                invocation_agent.session_state = session_view
                call_kwargs["agent"] = invocation_agent
        if "session_state" in sig.parameters:
            call_kwargs["session_state"] = (
                session_view if session_view is not None else ctx.session_state
            )

    return inject


def _worker_executor(spec: ToolSpec, ctx: MCPAgentContext):
    """Run ``spec`` in a subprocess worker whose commit the parent accepts."""

    async def execute(call_kwargs: Mapping[str, Any], attempt: AttemptContext) -> ExecutedCall:
        from .jobs import DeferredToolJob, run_tool_job

        # The worker runner owns process-group termination and uses
        # ``worker_timeout_s``. Cancellation is delayed until the cleanup
        # thread has reaped that process, so no mutating job is orphaned.
        effective_spec = spec
        handoff_limited = False
        if attempt.handoff_remaining_s is not None and (
            spec.worker_timeout_s is None or attempt.handoff_remaining_s <= spec.worker_timeout_s
        ):
            effective_spec = replace(spec, worker_timeout_s=attempt.handoff_remaining_s)
            handoff_limited = True
        try:
            result = await _run_deferred_worker_to_completion(
                run_tool_job,
                effective_spec,
                call_kwargs,
                ctx,
                defer_commit=True,
            )
        except MCPToolError as exc:
            if handoff_limited and exc.code == MCPErrorCode.TIMEOUT.value:
                raise _handoff_timeout_error(attempt.scope) from exc
            raise
        if not isinstance(result, DeferredToolJob):
            raise MCPToolError(
                "worker result was not deferred for parent acceptance",
                code=MCPErrorCode.INTERNAL,
            )
        return ExecutedCall(result.result, deferred=result)

    return execute


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
