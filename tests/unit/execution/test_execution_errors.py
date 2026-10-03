"""One error taxonomy shared by the workflow runtime and every tool runtime."""

from __future__ import annotations

import pytest

import cs_copilot.workflows as workflows
from cs_copilot.execution.errors import (
    NormalizedToolError,
    ToolErrorCode,
    ToolExecutionError,
    normalize_error,
)
from cs_copilot.mcp.errors import MCPErrorCode, MCPToolError, NormalizedMCPError


def test_mcp_and_workflow_names_are_the_same_objects():
    assert MCPErrorCode is ToolErrorCode
    assert workflows.ToolErrorCode is ToolErrorCode
    assert MCPToolError is ToolExecutionError
    assert NormalizedMCPError is NormalizedToolError


@pytest.mark.parametrize(
    ("exc", "code", "retryable"),
    [
        (ToolExecutionError("x", code="timeout", retryable=True), "timeout", True),
        (TimeoutError(), "timeout", True),
        (PermissionError("no"), "permission_denied", False),
        (ValueError("bad"), "invalid_input", False),
        (KeyError("k"), "invalid_input", False),
        (ConnectionError("down"), "transient_external", True),
        (MemoryError(), "resource_limit", False),
        (RuntimeError("other"), "internal", False),
        (workflows.InvalidTransitionError("t"), "invalid_input", False),
        (workflows.ArtifactIntegrityError("a"), "scientific_validation", False),
    ],
)
def test_normalize_error_taxonomy(exc, code, retryable):
    idempotent = normalize_error(exc, tool_name="demo_tool", idempotent=True)
    assert idempotent.code == code
    assert idempotent.retryable is retryable
    assert idempotent.message.startswith("demo_tool failed: ")
    assert normalize_error(exc, idempotent=False).retryable is False


def test_unknown_codes_are_rejected():
    with pytest.raises(ValueError, match="Unknown tool error code"):
        ToolExecutionError("x", code="not-a-code")
