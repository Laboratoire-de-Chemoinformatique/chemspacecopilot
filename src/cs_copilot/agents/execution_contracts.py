"""Map in-process Agno tool functions to execution-kernel contracts.

Every MCP tool spec names the in-process implementations of its operation in
``ToolSpec.agno_bindings`` (``module:Class.method`` or ``module:function``).
This module indexes those bindings by the function object they resolve to, so
an Agno toolkit call is recorded under the same tool identity and execution
contract as its MCP twin. Functions without an MCP twin get a conservative
synthesized contract named ``agno.<Owner>.<function>``.

The MCP tool *catalog* (``cs_copilot.mcp.tools_registry``) is imported lazily;
the MCP server module and the ``mcp`` SDK never are.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from typing import Any, Callable, Mapping

from cs_copilot import capabilities
from cs_copilot.execution.spec import ToolSpec

logger = logging.getLogger(__name__)

FunctionKey = tuple[str, str]

_INDEX: dict[FunctionKey, ToolSpec] | None = None
_DEFAULTS: dict[FunctionKey, ToolSpec] = {}
_LOCK = threading.Lock()


def function_key(function: Callable[..., Any]) -> FunctionKey:
    """Identify a callable by its defining module and qualified name."""

    target = inspect.unwrap(getattr(function, "__func__", function))
    target = getattr(target, "__func__", target)
    return str(target.__module__), str(target.__qualname__)


def binding_target(binding: str) -> Callable[..., Any]:
    """Import the function a ``module:Class.method`` / ``module:function`` names."""

    module_name, _, attribute = binding.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def contract_index() -> Mapping[FunctionKey, ToolSpec]:
    """Return (building once) the function-to-contract index of the MCP catalog."""

    global _INDEX
    with _LOCK:
        if _INDEX is None:
            from cs_copilot.mcp.tools_registry import all_specs

            index: dict[FunctionKey, ToolSpec] = {}
            for spec in all_specs():
                for binding in spec.agno_bindings:
                    key = function_key(binding_target(binding))
                    claimed = index.get(key)
                    if claimed is not None and claimed.mcp_name != spec.mcp_name:
                        raise ValueError(
                            f"{'.'.join(key)} is bound to both {claimed.mcp_name!r} "
                            f"and {spec.mcp_name!r}"
                        )
                    index[key] = spec
            _INDEX = index
        return _INDEX


def resolve_contract(
    function: Callable[..., Any],
    *,
    owner: Any = None,
) -> tuple[ToolSpec, str]:
    """Return ``(contract, source)`` for an Agno tool function.

    ``source`` is ``"spec"`` for a function bound to an MCP tool spec and
    ``"default"`` for a synthesized contract. ``owner`` is the toolkit instance
    (or ``None`` for a bare function tool).
    """

    key = function_key(function)
    spec = contract_index().get(key)
    if spec is not None:
        return spec, "spec"
    with _LOCK:
        default = _DEFAULTS.get(key)
        if default is None:
            default = _default_contract(function, owner)
            _DEFAULTS[key] = default
    return default, "default"


def _default_contract(function: Callable[..., Any], owner: Any) -> ToolSpec:
    target = inspect.unwrap(getattr(function, "__func__", function))
    owner_type = type(owner) if owner is not None else None
    owner_name = (
        owner_type.__name__ if owner_type is not None else target.__module__.rsplit(".")[-1]
    )
    group = None
    for cls in owner_type.__mro__ if owner_type is not None else ():
        group = capabilities.group_for_toolkit(cls.__name__)
        if group is not None:
            break
    doc = inspect.getdoc(target) or ""
    return ToolSpec(
        mcp_name=f"agno.{owner_name}.{target.__name__}",
        toolkit_factory=owner_type or (lambda: None),
        method=target.__name__,
        summary=doc.splitlines()[0] if doc else f"{owner_name}.{target.__name__}",
        group=group,
        # Unknown operations are treated as writers so observe mode audits them.
        write_scope="session",
    )


def reset_contract_index() -> None:
    """Forget the cached index (tests and catalog reloads)."""

    global _INDEX
    with _LOCK:
        _INDEX = None
        _DEFAULTS.clear()
