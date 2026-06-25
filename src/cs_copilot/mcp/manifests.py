"""Run-manifest recording for MCP tool calls."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Mapping

from cs_copilot.storage import OUTPUT_CONTEXT_KEY, S3

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
_SECRET_RE = re.compile(r"(token|secret|password|api[_-]?key|authorization)", re.I)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def record_tool_call(
    *,
    ctx: Any,
    tool_name: str,
    public_args: Mapping[str, Any],
    forced_args: Mapping[str, Any],
    status: str,
    duration_ms: float,
    result: Any = None,
    error: str | None = None,
) -> str | None:
    """Persist a compact JSON manifest for one MCP tool call.

    The MCP bootstrap initializes ``session_state['output_context']``. Calls
    made by isolated unit tests without that context are ignored.
    """

    session_state = getattr(ctx, "session_state", None)
    if not isinstance(session_state, dict):
        return None
    output_context = session_state.get(OUTPUT_CONTEXT_KEY)
    if not isinstance(output_context, dict):
        return None

    workflow_id = str(output_context.get("workflow_id") or "workflow")
    now = datetime.now(timezone.utc)
    rel_path = _manifest_rel_path(workflow_id, tool_name, now)
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "runtime": "mcp",
        "timestamp": now.isoformat(timespec="milliseconds"),
        "session_prefix": S3.current_prefix(),
        "workflow_id": workflow_id,
        "tool_name": tool_name,
        "status": status,
        "duration_ms": round(float(duration_ms), 3),
        "public_args": _redact(public_args),
        "forced_args": _redact(forced_args),
        "output_summary": _summarize(result),
    }
    if error:
        payload["error"] = _short_text(error, max_chars=1000)

    try:
        with S3.open(rel_path, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to write MCP run manifest for %s: %s", tool_name, exc)
        return None
    return S3.path(rel_path)


def _manifest_rel_path(workflow_id: str, tool_name: str, now: datetime) -> str:
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    safe_tool = _SAFE_NAME_RE.sub("_", tool_name).strip("._-") or "tool"
    filename = f"{stamp}_{safe_tool}_{uuid.uuid4().hex[:8]}.json"
    return PurePosixPath("workflows", workflow_id, "manifests", "mcp", filename).as_posix()


def _redact(value: Any) -> Any:
    return _json_safe(value, redact=True)


def _summarize(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "none"}
    if isinstance(value, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": [str(key) for key in list(value)[:20]],
        }
        for key in ("status", "row_count", "artifact_path", "path", "csv_path"):
            if key in value:
                summary[key] = _json_safe(value[key])
        return summary
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        return {"type": type(value).__name__, "length": len(values)}
    if isinstance(value, str):
        return {"type": "str", "preview": _short_text(value)}
    return {"type": type(value).__name__, "preview": _short_text(value)}


def _json_safe(value: Any, *, redact: bool = False, key: str | None = None, depth: int = 0) -> Any:
    if redact and key and _SECRET_RE.search(key):
        return "<redacted>"
    if depth > 4:
        return _short_text(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else None
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for index, (item_key, item_value) in enumerate(value.items()):
            if index >= 50:
                out["_truncated"] = True
                break
            str_key = str(item_key)
            out[str_key] = _json_safe(
                item_value,
                redact=redact,
                key=str_key,
                depth=depth + 1,
            )
        return out
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        out = [_json_safe(item, redact=redact, depth=depth + 1) for item in values[:50]]
        if len(values) > 50:
            out.append({"_truncated": len(values) - 50})
        return out
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return _short_text(value)


def _short_text(value: Any, *, max_chars: int = 240) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3]}..."
