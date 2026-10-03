"""Stable error taxonomy and normalization for tool invocations.

The same seven codes classify workflow-runtime failures and tool result
envelopes in every runtime (external MCP clients and the in-process Agno
team). This module only depends on the standard library so that the workflow
runtime can share the taxonomy without importing the execution kernel.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ToolErrorCode(str, Enum):
    """Stable error taxonomy shared by workflow and tool envelopes."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    TRANSIENT_EXTERNAL = "transient_external"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SCIENTIFIC_VALIDATION = "scientific_validation"
    INTERNAL = "internal"


@dataclass(frozen=True)
class NormalizedToolError:
    """Serializable form of a tool failure."""

    code: str
    message: str
    retryable: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


class ToolExecutionError(RuntimeError):
    """Tool failure carrying stable machine-readable semantics.

    The positional-message-only constructor remains supported for toolkit and
    test compatibility. Such errors are conservatively classified as
    ``internal`` and non-retryable until a caller supplies explicit metadata.
    """

    def __init__(
        self,
        message: str,
        *,
        code: ToolErrorCode | str = ToolErrorCode.INTERNAL,
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
) -> NormalizedToolError:
    """Map arbitrary Python failures onto the public v2 error taxonomy."""

    workflow_code = _workflow_error_code(exc)
    if isinstance(exc, ToolExecutionError):
        code = exc.code
        message = str(exc)
        retryable = exc.retryable
    elif workflow_code is not None:
        code = workflow_code
        message = str(exc)
        retryable = False
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        code = ToolErrorCode.TIMEOUT.value
        message = str(exc) or "Tool execution timed out."
        retryable = True
    elif isinstance(exc, PermissionError):
        code = ToolErrorCode.PERMISSION_DENIED.value
        message = str(exc) or "Tool execution was not permitted."
        retryable = False
    elif isinstance(exc, (ValueError, TypeError, KeyError)):
        code = ToolErrorCode.INVALID_INPUT.value
        message = str(exc) or "Tool input was invalid."
        retryable = False
    elif isinstance(exc, ConnectionError):
        code = ToolErrorCode.TRANSIENT_EXTERNAL.value
        message = str(exc) or "An external service was temporarily unavailable."
        retryable = True
    elif isinstance(exc, MemoryError):
        code = ToolErrorCode.RESOURCE_LIMIT.value
        message = str(exc) or "Tool execution exceeded an available resource."
        retryable = False
    else:
        code = ToolErrorCode.INTERNAL.value
        message = str(exc) or type(exc).__name__
        retryable = False

    retryable = bool(retryable and idempotent)
    if tool_name and not message.startswith(f"{tool_name} "):
        message = f"{tool_name} failed: {message}"
    return NormalizedToolError(code=code, message=message, retryable=retryable)


def _workflow_error_code(exc: BaseException) -> str | None:
    """Classify runtime contract failures without coupling the runtime to tools."""

    try:
        from cs_copilot.workflows.runtime import (
            ArtifactIntegrityError,
            InvalidTransitionError,
        )
    except ImportError:  # pragma: no cover - runtime is part of this package
        return None
    if isinstance(exc, InvalidTransitionError):
        return ToolErrorCode.INVALID_INPUT.value
    if isinstance(exc, ArtifactIntegrityError):
        return ToolErrorCode.SCIENTIFIC_VALIDATION.value
    return None


def _code_value(code: ToolErrorCode | str) -> str:
    value = code.value if isinstance(code, ToolErrorCode) else str(code)
    allowed = {item.value for item in ToolErrorCode}
    if value not in allowed:
        raise ValueError(f"Unknown tool error code: {value!r}")
    return value
