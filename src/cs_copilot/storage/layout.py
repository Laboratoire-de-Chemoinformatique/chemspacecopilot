#!/usr/bin/env python
# coding: utf-8
"""Session workflow output layout helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Optional

from .client import S3

OUTPUT_CONTEXT_KEY = "output_context"
LAYOUT_VERSION = 3
WORKFLOWS_DIR = "workflows"

_DEFAULT_WORKFLOW_SLUG = "workflow"
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_.-]+")


class OutputOperation(str, Enum):
    """Top-level operation folders inside a workflow."""

    CHEMICAL_SPACE = "01_chemical_space"
    ANALOG_GENERATION = "02_analog_generation"
    RETROSYNTHESIS = "03_retrosynthesis"
    REPORTS = "reports"


@dataclass(frozen=True)
class OutputLayout:
    """Structured output path builder for one workflow."""

    workflow_id: str

    def rel_path(self, operation: OutputOperation, *parts: str) -> str:
        cleaned_parts = [sanitize_path_part(part) for part in parts if str(part).strip()]
        path = PurePosixPath(WORKFLOWS_DIR, self.workflow_id, operation.value, *cleaned_parts)
        return path.as_posix()

    @property
    def manifest_rel_path(self) -> str:
        return PurePosixPath(WORKFLOWS_DIR, self.workflow_id, "manifest.json").as_posix()


def sanitize_path_part(value: Any, *, default: str = "artifact") -> str:
    """Return a safe single path component while preserving useful suffix dots."""

    text = str(value or "").replace("\\", "/").strip().strip("/")
    text = PurePosixPath(text).name if "/" in text else text
    text = _SAFE_PART_RE.sub("_", text).strip("._-")
    return text or default


def sanitize_workflow_slug(value: Any) -> str:
    return sanitize_path_part(value, default=_DEFAULT_WORKFLOW_SLUG).lower()


def is_explicit_storage_path(path: str) -> bool:
    return isinstance(path, str) and (
        path.startswith("s3://") or path.startswith("file://") or path.startswith("/")
    )


def is_workflow_scoped_path(path: str) -> bool:
    return isinstance(path, str) and path.strip("/").startswith(f"{WORKFLOWS_DIR}/")


def ensure_output_context(
    session_state: Optional[dict[str, Any]] = None,
    *,
    workflow_slug: Optional[str] = None,
) -> dict[str, Any]:
    """Ensure a layout context exists and return it.

    One storage session gets one workflow root.  The workflow id is derived from
    the active S3/local session prefix, so tools that do not share the same
    in-memory session_state still write under the same root.
    """

    workflow_id = _session_workflow_id()
    context = {"layout_version": LAYOUT_VERSION, "workflow_id": workflow_id}

    if isinstance(session_state, dict):
        existing = session_state.get(OUTPUT_CONTEXT_KEY)
        if (
            isinstance(existing, dict)
            and existing.get("layout_version") == LAYOUT_VERSION
            and existing.get("workflow_id") == workflow_id
        ):
            return existing
        session_state[OUTPUT_CONTEXT_KEY] = context

    return context


def current_output_layout(
    session_state: Optional[dict[str, Any]] = None,
    *,
    workflow_slug: Optional[str] = None,
) -> OutputLayout:
    context = ensure_output_context(session_state, workflow_slug=workflow_slug)
    return OutputLayout(str(context["workflow_id"]))


def operation_rel_path(
    operation: OutputOperation,
    *parts: str,
    session_state: Optional[dict[str, Any]] = None,
    workflow_slug: Optional[str] = None,
) -> str:
    return current_output_layout(session_state, workflow_slug=workflow_slug).rel_path(
        operation,
        *parts,
    )


def scoped_artifact_path(
    path: str,
    operation: OutputOperation,
    *folders: str,
    session_state: Optional[dict[str, Any]] = None,
    workflow_slug: Optional[str] = None,
) -> str:
    """Scope an ordinary relative artifact path into the workflow layout."""

    if is_explicit_storage_path(path) or is_workflow_scoped_path(path):
        return path
    filename = sanitize_path_part(path)
    return operation_rel_path(
        operation,
        *folders,
        filename,
        session_state=session_state,
        workflow_slug=workflow_slug,
    )


def _session_workflow_id() -> str:
    session_prefix = str(S3.current_prefix()).strip("/")
    session_name = PurePosixPath(session_prefix).name if session_prefix else _DEFAULT_WORKFLOW_SLUG
    return sanitize_path_part(session_name, default=_DEFAULT_WORKFLOW_SLUG)
