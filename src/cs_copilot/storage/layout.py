#!/usr/bin/env python
# coding: utf-8
"""Workflow-run output layout helpers.

The storage prefix identifies a chat/session. A session can contain multiple
workflow runs, so run identity is represented independently and persisted in
``session_state[OUTPUT_CONTEXT_KEY]`` (and a context variable for direct tool
calls that do not receive session state).
"""

from __future__ import annotations

import contextvars
import errno
import os
import re
import stat
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Optional

from .client import S3, _open_existing_directory_tree

OUTPUT_CONTEXT_KEY = "output_context"
LAYOUT_VERSION = 4
WORKFLOWS_DIR = "workflows"

_DEFAULT_WORKFLOW_SLUG = "workflow"
_SAFE_PART_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_RUN_CONTEXT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cs_copilot_run_context",
    default=None,
)


class OutputOperation(str, Enum):
    """Top-level operation folders inside a workflow run."""

    CHEMICAL_SPACE = "01_chemical_space"
    ANALOG_GENERATION = "02_analog_generation"
    RETROSYNTHESIS = "03_retrosynthesis"
    REPORTS = "reports"


@dataclass(frozen=True)
class OutputLayout:
    """Structured output path builder for one workflow run."""

    session_id: str
    run_id: str
    workflow_slug: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "session_id",
            validate_identifier(self.session_id, field="session_id"),
        )
        object.__setattr__(
            self,
            "run_id",
            validate_identifier(self.run_id, field="run_id"),
        )
        object.__setattr__(
            self,
            "workflow_slug",
            sanitize_workflow_slug(self.workflow_slug),
        )

    @property
    def run_root(self) -> str:
        return PurePosixPath(WORKFLOWS_DIR, self.run_id).as_posix()

    def rel_path(self, operation: OutputOperation, *parts: str) -> str:
        cleaned_parts = [sanitize_path_part(part) for part in parts if str(part).strip()]
        return PurePosixPath(self.run_root, operation.value, *cleaned_parts).as_posix()

    @property
    def manifest_rel_path(self) -> str:
        return PurePosixPath(self.run_root, "manifest.json").as_posix()

    @property
    def artifact_index_rel_path(self) -> str:
        return PurePosixPath(self.run_root, "artifacts", "index.json").as_posix()

    @property
    def events_rel_path(self) -> str:
        return PurePosixPath(self.run_root, "events").as_posix()

    def event_rel_path(self, event_id: str) -> str:
        return PurePosixPath(
            self.events_rel_path,
            f"{validate_identifier(event_id, field='event_id')}.jsonl",
        ).as_posix()

    def artifact_rel_path(self, path: str) -> str:
        """Resolve and validate a path relative to this run's root."""

        relative = normalize_run_relative_path(self.run_id, path)
        return PurePosixPath(self.run_root, relative).as_posix()


def open_local_run_artifact(layout: OutputLayout, relative_path: str) -> BinaryIO:
    """Open a local artifact for reading without following symlinks.

    The workflow root and every descendant component are opened by descriptor.
    Each child lookup is relative to its already-open parent, so replacing a
    directory between validation and the final open cannot redirect the read.
    This helper intentionally fails closed on platforms without ``openat`` and
    ``O_NOFOLLOW`` support.
    """

    relative = normalize_run_relative_path(layout.run_id, relative_path)
    if not _OPEN_SUPPORTS_DIR_FD or not _O_NOFOLLOW:
        raise RuntimeError(
            "secure local artifact reads require directory-relative no-follow support"
        )

    common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _O_NOFOLLOW
    directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0)
    current_fd: int | None = None
    file_fd: int | None = None
    try:
        current_fd = _open_existing_directory_tree(Path(S3.path(layout.run_root)).absolute())
        parts = PurePosixPath(relative).parts
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(parts[-1], common_flags, dir_fd=current_fd)
        metadata = os.fstat(file_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("artifact path must identify a regular, non-linked file")
        handle = os.fdopen(file_fd, "rb")
        file_fd = None
        return handle
    except OSError as exc:
        if isinstance(exc, PermissionError) or exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                f"artifact path resolves outside workflow run {layout.run_id!r} "
                "or contains a symlink"
            ) from exc
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if current_fd is not None:
            os.close(current_fd)


def sanitize_path_part(value: Any, *, default: str = "artifact") -> str:
    """Return a safe single path component while preserving useful suffix dots."""

    text = str(value or "").replace("\\", "/").strip().strip("/")
    text = PurePosixPath(text).name if "/" in text else text
    text = _SAFE_PART_RE.sub("_", text).strip("._-")
    return text or default


def sanitize_workflow_slug(value: Any) -> str:
    return sanitize_path_part(value, default=_DEFAULT_WORKFLOW_SLUG).lower()


def validate_identifier(value: Any, *, field: str = "identifier") -> str:
    """Validate an externally supplied identifier without silently rewriting it."""

    text = str(value or "").strip()
    if not _IDENTIFIER_RE.fullmatch(text):
        raise ValueError(
            f"{field} must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_' or '-' (maximum 128 characters)"
        )
    return text


def is_explicit_storage_path(path: str) -> bool:
    return isinstance(path, str) and (
        path.startswith("s3://") or path.startswith("file://") or path.startswith("/")
    )


def is_workflow_scoped_path(path: str) -> bool:
    return isinstance(path, str) and path.strip("/").startswith(f"{WORKFLOWS_DIR}/")


def normalize_run_relative_path(run_id: str, path: str) -> str:
    """Return a safe path relative to ``workflows/<run_id>``.

    Absolute paths, storage URLs, traversal components, paths belonging to
    another run, and runtime-control files are rejected. Artifact registration
    uses this function as its workflow-root containment boundary.
    """

    run_id = validate_identifier(run_id, field="run_id")
    if not isinstance(path, str) or not path.strip():
        raise ValueError("artifact path cannot be empty")
    value = path.strip()
    if is_explicit_storage_path(value) or "\\" in value:
        raise ValueError("artifact path must be relative to the workflow run")

    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact path must not be absolute or contain traversal components")

    run_prefix = PurePosixPath(WORKFLOWS_DIR, run_id)
    if candidate.parts and candidate.parts[0] == WORKFLOWS_DIR:
        try:
            candidate = candidate.relative_to(run_prefix)
        except ValueError as exc:
            raise ValueError(f"artifact path is outside workflow run {run_id!r}") from exc

    if not candidate.parts:
        raise ValueError("artifact path must identify a file")
    if candidate == PurePosixPath("manifest.json"):
        raise ValueError("manifest.json is reserved for the workflow runtime")
    if candidate.parts[0] in {".staging", "events"}:
        raise ValueError(f"{candidate.parts[0]}/ is reserved for the workflow runtime")
    if candidate == PurePosixPath("artifacts", "index.json"):
        raise ValueError("artifacts/index.json is reserved for the workflow runtime")
    return candidate.as_posix()


def ensure_output_context(
    session_state: Optional[dict[str, Any]] = None,
    *,
    workflow_slug: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    """Ensure and return explicit session/run/workflow output context."""

    resolved_session_id = validate_identifier(session_id or _session_id(), field="session_id")
    requested_slug = sanitize_workflow_slug(workflow_slug)

    candidates: list[dict[str, Any]] = []
    if isinstance(session_state, dict):
        existing = session_state.get(OUTPUT_CONTEXT_KEY)
        if isinstance(existing, dict):
            candidates.append(existing)
    contextual = _RUN_CONTEXT.get()
    if isinstance(contextual, dict):
        candidates.append(contextual)

    for existing in candidates:
        if not _valid_context(existing, session_id=resolved_session_id):
            continue
        if run_id is not None and existing["run_id"] != run_id:
            continue
        # Legacy toolkit helpers pass operation labels such as
        # ``chemical_space`` and ``reports`` here. Once a run is active those
        # labels must not fork the experiment; only an explicit run_id selects
        # a different run.
        context = dict(existing)
        _RUN_CONTEXT.set(context)
        if isinstance(session_state, dict):
            session_state[OUTPUT_CONTEXT_KEY] = context
        return context

    resolved_run_id = (
        validate_identifier(run_id, field="run_id")
        if run_id is not None
        else f"{requested_slug}-{uuid.uuid4().hex[:12]}"
    )
    context = {
        "layout_version": LAYOUT_VERSION,
        "session_id": resolved_session_id,
        "run_id": resolved_run_id,
        "workflow_slug": requested_slug,
    }
    _RUN_CONTEXT.set(context)
    if isinstance(session_state, dict):
        session_state[OUTPUT_CONTEXT_KEY] = context
    return context


def set_output_context(context: dict[str, Any]) -> dict[str, Any]:
    """Validate and activate a serialized output context."""

    required = {"layout_version", "session_id", "run_id", "workflow_slug"}
    missing = sorted(required - set(context))
    if missing:
        raise ValueError(f"output context is missing fields: {', '.join(missing)}")
    if context["layout_version"] != LAYOUT_VERSION:
        raise ValueError(
            f"unsupported output layout version {context['layout_version']!r}; "
            f"expected {LAYOUT_VERSION}"
        )
    normalized = {
        "layout_version": LAYOUT_VERSION,
        "session_id": validate_identifier(context["session_id"], field="session_id"),
        "run_id": validate_identifier(context["run_id"], field="run_id"),
        "workflow_slug": sanitize_workflow_slug(context["workflow_slug"]),
    }
    _RUN_CONTEXT.set(normalized)
    return normalized


def current_output_layout(
    session_state: Optional[dict[str, Any]] = None,
    *,
    workflow_slug: Optional[str] = None,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> OutputLayout:
    context = ensure_output_context(
        session_state,
        workflow_slug=workflow_slug,
        run_id=run_id,
        session_id=session_id,
    )
    return OutputLayout(
        session_id=str(context["session_id"]),
        run_id=str(context["run_id"]),
        workflow_slug=str(context["workflow_slug"]),
    )


def operation_rel_path(
    operation: OutputOperation,
    *parts: str,
    session_state: Optional[dict[str, Any]] = None,
    workflow_slug: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    return current_output_layout(
        session_state,
        workflow_slug=workflow_slug,
        run_id=run_id,
    ).rel_path(operation, *parts)


def scoped_artifact_path(
    path: str,
    operation: OutputOperation,
    *folders: str,
    session_state: Optional[dict[str, Any]] = None,
    workflow_slug: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Scope an ordinary relative artifact path into the workflow-run layout."""

    if is_explicit_storage_path(path) or is_workflow_scoped_path(path):
        return path
    filename = sanitize_path_part(path)
    return operation_rel_path(
        operation,
        *folders,
        filename,
        session_state=session_state,
        workflow_slug=workflow_slug,
        run_id=run_id,
    )


def _session_id() -> str:
    session_prefix = str(S3.current_prefix()).strip("/")
    session_name = PurePosixPath(session_prefix).name if session_prefix else "session"
    return sanitize_path_part(session_name, default="session")


def _valid_context(context: dict[str, Any], *, session_id: str) -> bool:
    required = {"layout_version", "session_id", "run_id", "workflow_slug"}
    if not required.issubset(context):
        return False
    try:
        return (
            context.get("layout_version") == LAYOUT_VERSION
            and validate_identifier(context.get("session_id"), field="session_id") == session_id
            and bool(validate_identifier(context.get("run_id"), field="run_id"))
            and bool(sanitize_workflow_slug(context.get("workflow_slug")))
        )
    except ValueError:
        return False
