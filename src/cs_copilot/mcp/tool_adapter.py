"""Adapter that wraps cs_copilot toolkit methods as FastMCP tools.

The adapter is the only place that knows about:

* injecting the shared :class:`~cs_copilot.mcp.context.MCPAgentContext`
  in place of ``agent`` / ``session_state`` parameters,
* merging the per-tool ``forces`` overrides (e.g. disabling the ChEMBL
  LLM-as-judge in MCP mode),
* dispatching synchronous toolkit calls onto a worker thread so the FastMCP
  stdio loop never stalls on long-running RDKit / GTM / ChEMBL work,
* coercing return values that are not natively JSON-serialisable.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
import typing
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Mapping

from .context import MCPAgentContext
from .errors import MCPToolError

logger = logging.getLogger(__name__)

# Parameter names that the adapter injects from the agent context rather than
# exposing on the public MCP schema.
INJECTED_PARAMS = ("agent", "session_state")

_DATAFRAME_PREVIEW_ROWS = 200


@dataclass(frozen=True)
class ToolSpec:
    """Declarative description of one MCP tool."""

    mcp_name: str
    toolkit_factory: Callable[[], Any]
    method: str
    summary: str
    group: str | None = None
    forces: Mapping[str, Any] = field(default_factory=dict)
    read_only: bool = False
    destructive: bool = False
    open_world: bool = False


def _coerce_return_value(value: Any) -> Any:
    """Best-effort JSON-friendly coercion of toolkit return values."""

    # Pandas is an optional but ubiquitous dependency in this project; importing
    # it eagerly is fine because every toolkit already pulls it in.
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
                "or read the session resource for the full content."
            ),
        }
    if isinstance(value, pd.Series):
        return value.to_dict()
    return value


def _resolve_annotations(method: Callable[..., Any]) -> Dict[str, Any]:
    """Resolve string annotations (PEP 563) to real types using the method's globals."""

    target = getattr(method, "__func__", method)
    try:
        return typing.get_type_hints(target, include_extras=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Falling back to raw annotations for %s: %s", method, exc)
        return getattr(target, "__annotations__", {}) or {}


def _public_parameters(method: Callable[..., Any], forces: Mapping[str, Any]):
    """Return the signature parameters exposed to the MCP client.

    String-style annotations are resolved into real types so that downstream
    pydantic schema builders (FastMCP's tool manager) can introspect them.
    """

    sig = inspect.signature(method)
    resolved = _resolve_annotations(method)
    hidden = set(INJECTED_PARAMS) | set(forces.keys())
    kept = []
    for name, param in sig.parameters.items():
        if name == "self" or name in hidden:
            continue
        annotation = resolved.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = param.annotation
        kept.append(param.replace(annotation=annotation))
    return sig, kept, resolved


def build_tool(
    spec: ToolSpec,
    instance: Any,
    ctx: MCPAgentContext,
) -> Callable[..., Any]:
    """Return an async callable suitable for FastMCP tool registration.

    Parameters
    ----------
    spec:
        The :class:`ToolSpec` describing the tool.
    instance:
        Toolkit instance whose method should be invoked.
    ctx:
        Shared :class:`MCPAgentContext` whose ``session_state`` is injected
        into every call.
    """

    bound_method = getattr(instance, spec.method)
    sig, public_params, resolved = _public_parameters(bound_method, spec.forces)

    # Resolve the return annotation in the same namespace as the parameters.
    return_annotation = resolved.get("return", sig.return_annotation)

    # Build a synthetic signature for FastMCP schema generation.
    public_signature = inspect.Signature(
        parameters=public_params, return_annotation=return_annotation
    )
    public_annotations: Dict[str, Any] = {
        param.name: param.annotation
        for param in public_params
        if param.annotation is not inspect.Parameter.empty
    }
    if return_annotation is not inspect.Parameter.empty:
        public_annotations["return"] = return_annotation

    async def _invoke(**kwargs: Any) -> Any:
        from .manifests import record_tool_call

        started = perf_counter()
        call_kwargs: Dict[str, Any] = dict(kwargs)
        if "agent" in sig.parameters:
            call_kwargs["agent"] = ctx
        if "session_state" in sig.parameters:
            call_kwargs["session_state"] = ctx.session_state
        for key, value in spec.forces.items():
            call_kwargs[key] = value

        try:
            if inspect.iscoroutinefunction(bound_method):
                result = await bound_method(**call_kwargs)
            else:
                result = await asyncio.to_thread(bound_method, **call_kwargs)
            coerced = _coerce_return_value(result)
        except MCPToolError as exc:
            duration_ms = (perf_counter() - started) * 1000
            record_tool_call(
                ctx=ctx,
                tool_name=spec.mcp_name,
                public_args=kwargs,
                forced_args=spec.forces,
                status="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            raise
        except Exception as exc:  # noqa: BLE001 — convert to protocol error
            duration_ms = (perf_counter() - started) * 1000
            record_tool_call(
                ctx=ctx,
                tool_name=spec.mcp_name,
                public_args=kwargs,
                forced_args=spec.forces,
                status="error",
                duration_ms=duration_ms,
                error=str(exc),
            )
            logger.error(
                "MCP tool %s failed: %s\n%s",
                spec.mcp_name,
                exc,
                traceback.format_exc(),
            )
            raise MCPToolError(f"{spec.mcp_name} failed: {exc}") from exc

        duration_ms = (perf_counter() - started) * 1000
        record_tool_call(
            ctx=ctx,
            tool_name=spec.mcp_name,
            public_args=kwargs,
            forced_args=spec.forces,
            status="success",
            duration_ms=duration_ms,
            result=coerced,
        )
        return coerced

    _invoke.__name__ = spec.mcp_name
    _invoke.__qualname__ = spec.mcp_name
    _invoke.__doc__ = spec.summary or (bound_method.__doc__ or "").strip()
    _invoke.__signature__ = public_signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = public_annotations
    _invoke.__wrapped__ = bound_method  # type: ignore[attr-defined]
    return _invoke
