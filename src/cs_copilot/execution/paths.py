"""Argument-location and parameter-bag helpers shared by the read and write boundaries."""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Mapping

from .errors import ToolErrorCode, ToolExecutionError


def _normalized_pandas_creation_function(
    arguments: Mapping[str, Any],
) -> tuple[str, str]:
    raw_function = arguments.get("create_using_function")
    if not isinstance(raw_function, str) or not raw_function.strip():
        raise ToolExecutionError(
            "create_using_function must name an approved DataFrame creation function",
            code=ToolErrorCode.INVALID_INPUT,
        )
    function = raw_function.strip()
    if function.startswith("pd."):
        function = function[3:]
    return raw_function, function


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


def _normalized_argument_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _format_argument_location(location: tuple[str, ...]) -> str:
    return ".".join(location) or "<argument>"
