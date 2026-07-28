"""Normalized error contracts for the cs_copilot MCP server."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MCPErrorCode(str, Enum):
    """Stable v2 error taxonomy returned to MCP clients."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    TRANSIENT_EXTERNAL = "transient_external"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SCIENTIFIC_VALIDATION = "scientific_validation"
    INTERNAL = "internal"


@dataclass(frozen=True)
class NormalizedMCPError:
    """Serializable form of an MCP tool failure."""

    code: str
    message: str
    retryable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class MCPToolError(RuntimeError):
    """Tool failure carrying stable machine-readable semantics.

    The positional-message-only constructor remains supported for toolkit and
    test compatibility. Such errors are conservatively classified as
    ``internal`` and non-retryable until a caller supplies explicit metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        code: MCPErrorCode | str = MCPErrorCode.INTERNAL,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = _code_value(code)
        self.retryable = bool(retryable)


def normalize_error(
    exc: BaseException,
    *,
    tool_name: str | None = None,
    idempotent: bool = False,
) -> NormalizedMCPError:
    """Map arbitrary Python failures onto the public v2 error taxonomy."""

    workflow_code = _workflow_error_code(exc)
    if isinstance(exc, MCPToolError):
        code = exc.code
        message = str(exc)
        retryable = exc.retryable
    elif workflow_code is not None:
        code = workflow_code
        message = str(exc)
        retryable = False
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        code = MCPErrorCode.TIMEOUT.value
        message = str(exc) or "Tool execution timed out."
        retryable = True
    elif isinstance(exc, PermissionError):
        code = MCPErrorCode.PERMISSION_DENIED.value
        message = str(exc) or "Tool execution was not permitted."
        retryable = False
    elif isinstance(exc, (ValueError, TypeError, KeyError)):
        code = MCPErrorCode.INVALID_INPUT.value
        message = str(exc) or "Tool input was invalid."
        retryable = False
    elif isinstance(exc, ConnectionError):
        code = MCPErrorCode.TRANSIENT_EXTERNAL.value
        message = str(exc) or "An external service was temporarily unavailable."
        retryable = True
    elif isinstance(exc, MemoryError):
        code = MCPErrorCode.RESOURCE_LIMIT.value
        message = str(exc) or "Tool execution exceeded an available resource."
        retryable = False
    else:
        code = MCPErrorCode.INTERNAL.value
        message = str(exc) or type(exc).__name__
        retryable = False

    retryable = bool(retryable and idempotent)
    if tool_name and not message.startswith(f"{tool_name} "):
        message = f"{tool_name} failed: {message}"
    return NormalizedMCPError(code=code, message=message, retryable=retryable)


def _workflow_error_code(exc: BaseException) -> str | None:
    """Classify runtime contract failures without coupling runtime to MCP."""

    try:
        from cs_copilot.workflows.runtime import (
            ArtifactIntegrityError,
            InvalidTransitionError,
        )
    except ImportError:  # pragma: no cover - runtime is part of this package
        return None
    if isinstance(exc, InvalidTransitionError):
        return MCPErrorCode.INVALID_INPUT.value
    if isinstance(exc, ArtifactIntegrityError):
        return MCPErrorCode.SCIENTIFIC_VALIDATION.value
    return None


def _code_value(code: MCPErrorCode | str) -> str:
    value = code.value if isinstance(code, MCPErrorCode) else str(code)
    allowed = {item.value for item in MCPErrorCode}
    if value not in allowed:
        raise ValueError(f"Unknown MCP error code: {value!r}")
    return value
