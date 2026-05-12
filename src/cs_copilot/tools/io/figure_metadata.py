#!/usr/bin/env python
# coding: utf-8
"""Shared helpers for reportable figure metadata."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from .session_memory import register_session_object

FIGURE_METADATA_VERSION = 1
REPORT_ROLE_INLINE_STATIC = "inline_static"
REPORT_ROLE_INTERACTIVE_ONLY = "interactive_only"
REPORT_ROLE_EXCLUDE = "exclude"
REPORTABLE_FIGURE_ROLES = {REPORT_ROLE_INLINE_STATIC}


def _clean_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _clean_list(value: Any) -> list[str]:
    cleaned = []
    for item in _as_list(value):
        text = _clean_text(item)
        if text:
            cleaned.append(text)
    return cleaned


def normalize_figure_metadata(metadata: Any) -> dict[str, Any]:
    """Return a compact, normalized figure metadata dictionary."""
    if not isinstance(metadata, dict):
        return {}
    meaningful_keys = (
        "figure_kind",
        "renderer",
        "report_role",
        "title_subject",
        "paths",
        "color_encoding",
        "overlays",
        "node_labels",
        "caption_facts",
    )
    if not any(metadata.get(key) for key in meaningful_keys):
        return {}

    normalized = deepcopy(metadata)
    normalized["schema_version"] = int(normalized.get("schema_version") or FIGURE_METADATA_VERSION)
    normalized["figure_kind"] = _clean_text(normalized.get("figure_kind"))
    normalized["renderer"] = _clean_text(normalized.get("renderer"))
    normalized["report_role"] = (
        _clean_text(normalized.get("report_role")) or REPORT_ROLE_INTERACTIVE_ONLY
    )
    normalized["title_subject"] = _clean_text(normalized.get("title_subject"))
    normalized["caption_facts"] = _clean_list(normalized.get("caption_facts"))

    paths = normalized.get("paths")
    if not isinstance(paths, dict):
        paths = {}
    normalized["paths"] = {str(key): _clean_text(value) for key, value in paths.items() if value}

    color_encoding = normalized.get("color_encoding")
    if isinstance(color_encoding, dict):
        normalized["color_encoding"] = _normalize_color_encoding(color_encoding)
    elif isinstance(color_encoding, list):
        normalized["color_encoding"] = [
            _normalize_color_encoding(item) for item in color_encoding if isinstance(item, dict)
        ]
    else:
        normalized["color_encoding"] = []

    normalized["overlays"] = [
        _normalize_overlay(item)
        for item in _as_list(normalized.get("overlays"))
        if isinstance(item, dict)
    ]
    normalized["node_labels"] = _clean_list(normalized.get("node_labels"))
    return normalized


def _normalize_color_encoding(encoding: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(encoding)
    for key in (
        "role",
        "field",
        "encoded_variable",
        "palette",
        "legend_title",
        "low_value_meaning",
        "high_value_meaning",
        "legend_text",
    ):
        if key in normalized:
            normalized[key] = _clean_text(normalized[key])
    return normalized


def _normalize_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(overlay)
    for key in ("role", "color", "symbol", "meaning", "legend_text"):
        if key in normalized:
            normalized[key] = _clean_text(normalized[key])
    return normalized


def build_figure_metadata(
    *,
    figure_kind: str,
    renderer: str,
    report_role: str,
    title_subject: str,
    paths: Optional[dict[str, Any]] = None,
    color_encoding: Optional[dict[str, Any] | list[dict[str, Any]]] = None,
    overlays: Optional[list[dict[str, Any]]] = None,
    node_labels: Optional[list[Any]] = None,
    caption_facts: Optional[list[str]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build normalized figure metadata for session memory and reports."""
    payload: Dict[str, Any] = {
        "schema_version": FIGURE_METADATA_VERSION,
        "figure_kind": figure_kind,
        "renderer": renderer,
        "report_role": report_role,
        "title_subject": title_subject,
        "paths": paths or {},
        "color_encoding": color_encoding or [],
        "overlays": overlays or [],
        "node_labels": node_labels or [],
        "caption_facts": caption_facts or [],
    }
    payload.update({key: value for key, value in extra.items() if value is not None})
    return normalize_figure_metadata(payload)


def figure_caption_facts(metadata: Any) -> list[str]:
    """Return deterministic caption facts from normalized figure metadata."""
    normalized = normalize_figure_metadata(metadata)
    facts = list(normalized.get("caption_facts") or [])
    for encoding in _as_list(normalized.get("color_encoding")):
        if not isinstance(encoding, dict):
            continue
        legend_text = _clean_text(encoding.get("legend_text"))
        if legend_text:
            facts.append(legend_text)
            continue

        encoded_variable = _clean_text(encoding.get("encoded_variable"))
        palette = _clean_text(encoding.get("palette"))
        high = _clean_text(encoding.get("high_value_meaning"))
        low = _clean_text(encoding.get("low_value_meaning"))
        if encoded_variable:
            if palette:
                facts.append(
                    f"Cell color uses the {palette} colorscale to encode {encoded_variable}."
                )
            else:
                facts.append(f"Cell color encodes {encoded_variable}.")
        if high or low:
            value_parts = []
            if high:
                value_parts.append(f"higher legend values indicate {high}")
            if low:
                value_parts.append(f"lower legend values indicate {low}")
            facts.append("; ".join(value_parts).capitalize() + ".")

    for overlay in normalized.get("overlays") or []:
        if not isinstance(overlay, dict):
            continue
        legend_text = _clean_text(overlay.get("legend_text"))
        if legend_text:
            facts.append(legend_text)
            continue
        color = _clean_text(overlay.get("color"))
        meaning = _clean_text(overlay.get("meaning"))
        symbol = _clean_text(overlay.get("symbol")) or "markers"
        if color and meaning:
            facts.append(f"{color.capitalize()} {symbol} show {meaning}.")
        elif meaning:
            facts.append(f"Overlay markers show {meaning}.")

    node_labels = normalized.get("node_labels") or []
    if node_labels:
        facts.append(f"Labeled GTM nodes: {', '.join(str(node) for node in node_labels)}.")

    deduplicated = []
    seen = set()
    for fact in facts:
        text = " ".join(str(fact).split()).strip()
        if not text:
            continue
        key = text.lower()
        if key not in seen:
            seen.add(key)
            deduplicated.append(text)
    return deduplicated


def report_image_path(metadata: Any) -> str:
    """Return the static image path if this metadata represents an inline figure."""
    normalized = normalize_figure_metadata(metadata)
    if normalized.get("report_role") not in REPORTABLE_FIGURE_ROLES:
        return ""
    paths = normalized.get("paths") or {}
    return _clean_text(paths.get("png_path") or paths.get("image_path") or paths.get("path"))


def session_figure_metadata(
    session_state: Optional[dict[str, Any]],
    figure_id: Any,
) -> dict[str, Any]:
    """Resolve a registered figure metadata record from session state."""
    if not session_state or not figure_id:
        return {}
    memory = session_state.get("session_objects")
    if not isinstance(memory, dict):
        return {}
    figures = memory.get("figures")
    if not isinstance(figures, dict):
        return {}
    record = figures.get(str(figure_id))
    if not isinstance(record, dict):
        return {}
    return normalize_figure_metadata(record)


def register_figure_metadata(
    session_state: dict[str, Any],
    metadata: dict[str, Any],
    *,
    label: Optional[str] = None,
    source_agent: Optional[str] = None,
    source_tool: Optional[str] = None,
    set_current: bool = True,
) -> str:
    """Register normalized figure metadata in shared session memory."""
    normalized = normalize_figure_metadata(metadata)
    return register_session_object(
        session_state,
        "figure",
        normalized,
        label=label or normalized.get("title_subject") or normalized.get("figure_kind"),
        source_agent=source_agent,
        source_tool=source_tool,
        set_current=set_current,
    )
