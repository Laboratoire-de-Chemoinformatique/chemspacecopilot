"""Normalize Agno run outputs and tool executions without parsing logs."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .models import ToolCallRecord

_SENSITIVE_KEY = re.compile(
    r"(api[_-]?key|authorization|credential|password|secret|access[_-]?token)",
    re.IGNORECASE,
)
_MAX_PREVIEW_CHARS = 1000
_MAX_COLLECTION_ITEMS = 50


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        output: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                output["_truncated"] = True
                break
            key_text = str(key)
            output[key_text] = (
                "[REDACTED]"
                if _SENSITIVE_KEY.search(key_text)
                else _json_safe(item, depth=depth + 1)
            )
        return output
    if isinstance(value, (list, tuple, set)):
        values = list(value)
        output = [_json_safe(item, depth=depth + 1) for item in values[:_MAX_COLLECTION_ITEMS]]
        if len(values) > _MAX_COLLECTION_ITEMS:
            output.append({"_truncated": len(values) - _MAX_COLLECTION_ITEMS})
        return output
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(exclude_none=True), depth=depth + 1)
        except Exception:
            pass
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(), depth=depth + 1)
        except Exception:
            pass
    return str(value)


def _result_details(result: Any) -> Tuple[Optional[str], Optional[str]]:
    if result is None:
        return None, None
    if isinstance(result, str):
        text = result
    else:
        try:
            text = json.dumps(_json_safe(result), sort_keys=True, default=str)
        except Exception:
            text = str(result)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    preview = text if len(text) <= _MAX_PREVIEW_CHARS else f"{text[:_MAX_PREVIEW_CHARS]}…"
    return preview, digest


def _metric_value(metrics: Any, name: str, default: int | float = 0) -> int | float:
    if metrics is None:
        return default
    value = getattr(metrics, name, None)
    if value is None and isinstance(metrics, dict):
        value = metrics.get(name)
    if value is None:
        return default
    try:
        return float(value) if isinstance(default, float) else int(value)
    except (TypeError, ValueError):
        return default


def _iter_outputs(root: Any) -> Iterable[Tuple[Any, Optional[str]]]:
    """Yield a run output tree once, including team members."""
    stack: List[Tuple[Any, Optional[str]]] = [(root, getattr(root, "team_name", None))]
    seen: Set[int] = set()
    while stack:
        output, parent_name = stack.pop()
        output_id = id(output)
        if output_id in seen:
            continue
        seen.add(output_id)
        name = (
            getattr(output, "agent_name", None) or getattr(output, "team_name", None) or parent_name
        )
        yield output, name
        members = getattr(output, "member_responses", None) or []
        for member in reversed(members):
            stack.append((member, name))


def _iter_tools(output: Any) -> Iterable[Any]:
    tools = getattr(output, "tools", None)
    if tools:
        yield from tools
        return
    if isinstance(output, dict):
        yield from output.get("tools") or output.get("tool_calls") or []


def _tool_attr(tool: Any, name: str, default: Any = None) -> Any:
    if isinstance(tool, dict):
        return tool.get(name, default)
    return getattr(tool, name, default)


def _optional_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def normalize_agno_output(
    run_output: Any,
    *,
    pricing: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Return aggregate model metrics and structured tool-call records.

    Team-level Agno metrics contain coordinator calls. Member metrics live on
    ``member_responses`` and are added recursively, which avoids treating only
    the coordinator as the cost of a multi-agent run.
    """
    totals: Dict[str, int | float] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "llm_duration_seconds": 0.0,
        "model_call_count": 0,
    }
    tool_records: List[ToolCallRecord] = []
    models: List[Dict[str, Optional[str]]] = []

    for output, agent_name in _iter_outputs(run_output):
        metrics = getattr(output, "metrics", None)
        token_total = int(_metric_value(metrics, "total_tokens"))
        if metrics is not None:
            totals["input_tokens"] += int(_metric_value(metrics, "input_tokens"))
            totals["output_tokens"] += int(_metric_value(metrics, "output_tokens"))
            totals["total_tokens"] += token_total
            totals["reasoning_tokens"] += int(_metric_value(metrics, "reasoning_tokens"))
            totals["cache_read_tokens"] += int(_metric_value(metrics, "cache_read_tokens"))
            totals["cache_write_tokens"] += int(_metric_value(metrics, "cache_write_tokens"))
            totals["llm_duration_seconds"] += float(_metric_value(metrics, "duration", 0.0))
            if token_total or getattr(output, "model", None):
                totals["model_call_count"] += 1

        model = getattr(output, "model", None)
        provider = getattr(output, "model_provider", None)
        if model or provider:
            model_item = {"model_id": model, "model_provider": provider}
            if model_item not in models:
                models.append(model_item)

        for tool in _iter_tools(output):
            result_preview, result_sha256 = _result_details(_tool_attr(tool, "result"))
            tool_metrics = _tool_attr(tool, "metrics")
            tool_records.append(
                ToolCallRecord(
                    sequence=len(tool_records),
                    agent_name=agent_name,
                    tool_name=str(_tool_attr(tool, "tool_name", "unknown") or "unknown"),
                    tool_args=_json_safe(_tool_attr(tool, "tool_args", {}) or {}),
                    created_at=_optional_int(_tool_attr(tool, "created_at")),
                    duration_seconds=(
                        float(_metric_value(tool_metrics, "duration", 0.0))
                        if tool_metrics is not None
                        else None
                    ),
                    error=bool(_tool_attr(tool, "tool_call_error", False)),
                    result_preview=result_preview,
                    result_sha256=result_sha256,
                    child_run_id=_tool_attr(tool, "child_run_id"),
                )
            )

    tool_records.sort(
        key=lambda record: (
            record.created_at is None,
            record.created_at or 0,
            record.sequence,
        )
    )
    for sequence, record in enumerate(tool_records):
        record.sequence = sequence

    pricing = pricing or {}
    input_rate = float(pricing.get("input_per_million", 0) or 0)
    output_rate = float(pricing.get("output_per_million", 0) or 0)
    estimated_cost = None
    if input_rate or output_rate:
        estimated_cost = (
            int(totals["input_tokens"]) * input_rate + int(totals["output_tokens"]) * output_rate
        ) / 1_000_000

    return {
        **totals,
        "llm_duration_seconds": round(float(totals["llm_duration_seconds"]), 6),
        "estimated_cost": estimated_cost,
        "models": models,
        "tool_calls": [record.to_dict() for record in tool_records],
        "tool_call_count": len(tool_records),
        "failed_tool_call_count": sum(record.error for record in tool_records),
    }
