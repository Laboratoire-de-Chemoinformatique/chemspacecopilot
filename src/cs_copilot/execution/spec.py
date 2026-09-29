"""Declarative execution contract for one tool, shared by every runtime."""

from __future__ import annotations

import inspect
import logging
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

logger = logging.getLogger(__name__)
INJECTED_PARAMS = ("agent", "session_state")
DEFAULT_MAX_OUTPUT_BYTES = 1_000_000
_RISK_LEVELS = frozenset({"low", "medium", "high"})
_WRITE_SCOPES = frozenset({"none", "session", "external"})
_ARTIFACT_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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
    # In-process implementations of this operation, as "module:Class.method" or
    # "module:function". Specs whose factory builds a toolkit (not an MCP
    # facade) are bound to ``<toolkit>.<method>`` automatically.
    agno_bindings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        bindings = tuple(dict.fromkeys(str(item).strip() for item in self.agno_bindings))
        if any(
            item.count(":") != 1 or item.startswith(":") or item.endswith(":") for item in bindings
        ):
            raise ValueError(
                f"{self.mcp_name}: agno_bindings entries must be 'module:Class.method' "
                "or 'module:function'"
            )
        object.__setattr__(self, "agno_bindings", bindings)
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


def _is_control_plane_spec(spec: ToolSpec) -> bool:
    return (
        spec.group == "skills"
        or spec.mcp_name == "mcp_bootstrap"
        or spec.mcp_name.startswith("workflow_")
    )
