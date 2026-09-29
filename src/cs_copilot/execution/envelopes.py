"""Result coercion, output limits, and the schema-v2 result envelope."""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from .errors import ToolErrorCode, ToolExecutionError
from .spec import ToolSpec

RESULT_SCHEMA_VERSION = 2
_DATAFRAME_PREVIEW_ROWS = 200


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


def _enforce_output_limit(spec: ToolSpec, value: Any) -> int:
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(
            f"returned a value that is not JSON serializable: {exc}",
            code=ToolErrorCode.INTERNAL,
        ) from exc
    output_bytes = len(encoded)
    if spec.max_output_bytes is not None and output_bytes > spec.max_output_bytes:
        raise ToolExecutionError(
            f"output was {output_bytes} bytes; limit is {spec.max_output_bytes} bytes. "
            "Persist the full result as a run artifact and return its artifact id.",
            code=ToolErrorCode.RESOURCE_LIMIT,
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
