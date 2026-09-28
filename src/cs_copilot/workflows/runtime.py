"""Versioned workflow-run state, events, and artifact persistence.

Events are authoritative. Every event is stored in its own immutable JSONL
object so local filesystems and object stores have the same append semantics.
``manifest.json`` and ``artifacts/index.json`` are replaceable snapshots that
can always be rebuilt by replaying the event objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import fsspec

from cs_copilot.storage import (
    LAYOUT_VERSION,
    OUTPUT_CONTEXT_KEY,
    S3,
    OutputLayout,
    ensure_output_context,
    get_s3_config,
    is_s3_enabled,
    normalize_run_relative_path,
    open_local_run_artifact,
    sanitize_workflow_slug,
    validate_identifier,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
_FORBIDDEN_HANDOFF_FIELDS = frozenset(
    {
        "messages",
        "history",
        "conversation_history",
        "private_reasoning",
        "chain_of_thought",
        "scratchpad",
        "agent_scratch",
    }
)
_RUN_LOCKS_GUARD = threading.Lock()
_RUN_LOCKS: dict[tuple[str, str], threading.RLock] = {}


def _reject_private_handoff_context(
    value: Any,
    *,
    path: str = "handoff",
    depth: int = 0,
) -> None:
    """Reject private/history channels at any depth in structured handoff data."""

    if depth > 32:
        raise ValueError("handoff JSON nesting exceeds 32 levels")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_HANDOFF_FIELDS:
                raise ValueError(f"{path} contains forbidden private/history field {str(key)!r}")
            _reject_private_handoff_context(
                nested,
                path=f"{path}.{key}",
                depth=depth + 1,
            )
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_private_handoff_context(
                nested,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )


class RunStatus(str, Enum):
    """Lifecycle states for a workflow run."""

    SUBMITTED = "submitted"
    PLANNING = "planning"
    RUNNING = "running"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(str, Enum):
    """Lifecycle states for one workflow task."""

    PENDING = "pending"
    RUNNING = "running"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class ToolErrorCode(str, Enum):
    """Stable error taxonomy shared by workflow and tool envelopes."""

    INVALID_INPUT = "invalid_input"
    PERMISSION_DENIED = "permission_denied"
    TRANSIENT_EXTERNAL = "transient_external"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    SCIENTIFIC_VALIDATION = "scientific_validation"
    INTERNAL = "internal"


class ArtifactTrust(str, Enum):
    """Trust classification for artifact content."""

    INTERNAL = "internal"
    EXTERNAL = "external"
    UNTRUSTED = "untrusted"


_TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PARTIAL,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
    }
)


_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.SUBMITTED: frozenset(
        {
            RunStatus.PLANNING,
            RunStatus.RUNNING,
            RunStatus.INPUT_REQUIRED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.PLANNING: frozenset(
        {RunStatus.RUNNING, RunStatus.INPUT_REQUIRED, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.INPUT_REQUIRED,
            RunStatus.COMPLETED,
            RunStatus.PARTIAL,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
    ),
    RunStatus.INPUT_REQUIRED: frozenset(
        {RunStatus.PLANNING, RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}
    ),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.PARTIAL: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

_TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.SKIPPED}),
    TaskStatus.RUNNING: frozenset(
        {
            TaskStatus.INPUT_REQUIRED,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        }
    ),
    TaskStatus.INPUT_REQUIRED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED}
    ),
    TaskStatus.FAILED: frozenset({TaskStatus.RUNNING}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.SKIPPED: frozenset(),
}


class WorkflowRuntimeError(RuntimeError):
    """Base error for workflow runtime contract failures."""


class InvalidTransitionError(WorkflowRuntimeError):
    """Raised when a run or task lifecycle transition is illegal."""


class ArtifactIntegrityError(WorkflowRuntimeError):
    """Raised when an artifact is missing or its checksum does not match."""


class EventReplayError(WorkflowRuntimeError):
    """Raised when an event stream is malformed or cannot be replayed."""


class _EventConflictError(WorkflowRuntimeError):
    """Raised when another writer reserved the next event sequence first."""


@dataclass(frozen=True)
class ToolError:
    """Normalized tool/workflow error suitable for persistence and retry policy."""

    code: ToolErrorCode
    message: str
    retryable: bool = False
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", ToolErrorCode(self.code))
        if not str(self.message).strip():
            raise ValueError("ToolError.message cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "details": _json_safe(self.details),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ToolError":
        return cls(
            code=ToolErrorCode(str(value["code"])),
            message=str(value["message"]),
            retryable=bool(value.get("retryable", False)),
            details=dict(value.get("details") or {}),
        )


@dataclass
class TaskRecord:
    """Durable state for one role-assigned workflow task."""

    task_id: str
    role: str
    profile: str
    step: str
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    input_artifact_ids: list[str] = field(default_factory=list)
    output_artifact_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    error: ToolError | None = None

    def __post_init__(self) -> None:
        self.task_id = validate_identifier(self.task_id, field="task_id")
        self.status = TaskStatus(self.status)
        if self.attempts < 0:
            raise ValueError("task attempts cannot be negative")
        if not self.role.strip() or not self.profile.strip() or not self.step.strip():
            raise ValueError("task role, profile, and step are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "profile": self.profile,
            "step": self.step,
            "status": self.status.value,
            "attempts": self.attempts,
            "input_artifact_ids": list(self.input_artifact_ids),
            "output_artifact_ids": list(self.output_artifact_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskRecord":
        error = value.get("error")
        return cls(
            task_id=str(value["task_id"]),
            role=str(value["role"]),
            profile=str(value["profile"]),
            step=str(value["step"]),
            status=TaskStatus(str(value.get("status", TaskStatus.PENDING.value))),
            attempts=int(value.get("attempts", 0)),
            input_artifact_ids=[str(item) for item in value.get("input_artifact_ids", ())],
            output_artifact_ids=[str(item) for item in value.get("output_artifact_ids", ())],
            created_at=str(value.get("created_at") or _utc_now()),
            updated_at=str(value.get("updated_at") or _utc_now()),
            error=ToolError.from_dict(error) if isinstance(error, Mapping) else None,
        )


@dataclass(frozen=True)
class HandoffEnvelope:
    """Minimal, replayable context passed between workflow roles."""

    handoff_id: str
    run_id: str
    workflow_slug: str
    task_id: str
    sender_role: str
    receiver_role: str
    objective: str
    trace_id: str
    span_id: str
    task_attempt: int | None = None
    parent_span_id: str | None = None
    constraints: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    input_artifact_ids: tuple[str, ...] = ()
    expected_output_artifacts: tuple[str, ...] = ()
    expected_output_schema: Mapping[str, Any] = field(default_factory=dict)
    context_summary: str | None = None
    budget: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _utc_now())
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "handoff_id", validate_identifier(self.handoff_id, field="handoff_id")
        )
        object.__setattr__(self, "run_id", validate_identifier(self.run_id, field="run_id"))
        object.__setattr__(self, "task_id", validate_identifier(self.task_id, field="task_id"))
        object.__setattr__(self, "workflow_slug", sanitize_workflow_slug(self.workflow_slug))
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported handoff schema version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if self.task_attempt is not None and (
            isinstance(self.task_attempt, bool)
            or not isinstance(self.task_attempt, int)
            or self.task_attempt < 0
        ):
            raise ValueError("handoff task_attempt must be a non-negative integer")
        budget = self.budget.as_dict() if hasattr(self.budget, "as_dict") else self.budget
        if not isinstance(budget, Mapping):
            raise TypeError("handoff budget must be a mapping or expose as_dict()")
        object.__setattr__(self, "budget", dict(budget))
        output_schema = (
            self.expected_output_schema.as_dict()
            if hasattr(self.expected_output_schema, "as_dict")
            else self.expected_output_schema
        )
        if not isinstance(output_schema, Mapping):
            raise TypeError("handoff expected_output_schema must be a mapping")
        object.__setattr__(self, "expected_output_schema", dict(output_schema))
        _reject_private_handoff_context(
            {
                "expected_output_schema": output_schema,
                "budget": budget,
            }
        )
        for budget_name in ("max_tokens", "max_tool_calls", "timeout_seconds"):
            budget_value = budget.get(budget_name)
            if budget_value is not None and (
                isinstance(budget_value, bool)
                or not isinstance(budget_value, (int, float))
                or budget_value <= 0
            ):
                raise ValueError(f"handoff budget {budget_name} must be a positive number")
        for field_name in (
            "constraints",
            "required_capabilities",
            "acceptance_criteria",
            "input_artifact_ids",
            "expected_output_artifacts",
        ):
            object.__setattr__(self, field_name, _string_tuple(getattr(self, field_name)))
        required = (
            self.sender_role,
            self.receiver_role,
            self.objective,
            self.trace_id,
            self.span_id,
        )
        if any(not value.strip() for value in required):
            raise ValueError(
                "handoff sender, receiver, objective, trace_id, and span_id are required"
            )
        if self.sender_role == self.receiver_role:
            raise ValueError("handoff sender_role and receiver_role must differ")

    def to_dict(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "run_id": self.run_id,
            "workflow_slug": self.workflow_slug,
            "task_id": self.task_id,
            "sender_role": self.sender_role,
            "receiver_role": self.receiver_role,
            "objective": self.objective,
            "task_attempt": self.task_attempt,
            "constraints": list(self.constraints),
            "required_capabilities": list(self.required_capabilities),
            "acceptance_criteria": list(self.acceptance_criteria),
            "input_artifact_ids": list(self.input_artifact_ids),
            "expected_output_artifacts": list(self.expected_output_artifacts),
            "expected_output_schema": _json_safe(self.expected_output_schema),
            "context_summary": self.context_summary,
            "budget": _json_safe(self.budget),
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "created_at": self.created_at,
            "schema_version": self.schema_version,
        }

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        workflow_slug: str,
        task_id: str,
        sender_role: str,
        receiver_role: str,
        objective: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        **kwargs: Any,
    ) -> "HandoffEnvelope":
        """Create a handoff with generated stable and trace identifiers."""

        return cls(
            handoff_id=f"handoff-{uuid.uuid4().hex[:12]}",
            run_id=run_id,
            workflow_slug=workflow_slug,
            task_id=task_id,
            sender_role=sender_role,
            receiver_role=receiver_role,
            objective=objective,
            trace_id=trace_id or uuid.uuid4().hex,
            span_id=uuid.uuid4().hex,
            parent_span_id=parent_span_id,
            **kwargs,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HandoffEnvelope":
        """Validate an external handoff and stamp its creation time locally."""

        _reject_private_handoff_context(value)
        return cls._from_mapping_fields(value, created_at=_utc_now())

    @classmethod
    def _from_mapping_fields(
        cls,
        value: Mapping[str, Any],
        *,
        created_at: str,
    ) -> "HandoffEnvelope":
        return cls(
            handoff_id=str(value.get("handoff_id") or f"handoff-{uuid.uuid4().hex[:12]}"),
            run_id=str(value["run_id"]),
            workflow_slug=str(value["workflow_slug"]),
            task_id=str(value["task_id"]),
            sender_role=str(value["sender_role"]),
            receiver_role=str(value["receiver_role"]),
            objective=str(value["objective"]),
            task_attempt=(
                int(value["task_attempt"]) if value.get("task_attempt") is not None else None
            ),
            constraints=tuple(str(item) for item in value.get("constraints", ())),
            required_capabilities=tuple(
                str(item) for item in value.get("required_capabilities", ())
            ),
            acceptance_criteria=tuple(str(item) for item in value.get("acceptance_criteria", ())),
            input_artifact_ids=tuple(
                str(item) for item in value.get("input_artifact_ids", value.get("artifact_ids", ()))
            ),
            expected_output_artifacts=tuple(
                str(item) for item in value.get("expected_output_artifacts", ())
            ),
            expected_output_schema=dict(value.get("expected_output_schema") or {}),
            context_summary=(
                str(value["context_summary"]) if value.get("context_summary") else None
            ),
            budget=dict(value.get("budget") or {}),
            trace_id=str(value.get("trace_id") or ""),
            span_id=str(value.get("span_id") or ""),
            parent_span_id=(str(value["parent_span_id"]) if value.get("parent_span_id") else None),
            created_at=created_at,
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HandoffEnvelope":
        """Restore a persisted handoff, including its authoritative timestamp."""

        _reject_private_handoff_context(value)
        return cls._from_mapping_fields(
            value,
            created_at=str(value.get("created_at") or _utc_now()),
        )


@dataclass(frozen=True)
class ArtifactRecord:
    """Immutable identity and provenance for one workflow-scoped artifact."""

    artifact_id: str
    run_id: str
    artifact_type: str
    mime_type: str
    relative_path: str
    sha256: str
    size_bytes: int
    producer_task_id: str | None = None
    producer_tool: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    trust: ArtifactTrust = ArtifactTrust.INTERNAL
    created_at: str = field(default_factory=lambda: _utc_now())

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "artifact_id", validate_identifier(self.artifact_id, field="artifact_id")
        )
        object.__setattr__(self, "run_id", validate_identifier(self.run_id, field="run_id"))
        object.__setattr__(
            self,
            "relative_path",
            normalize_run_relative_path(self.run_id, self.relative_path),
        )
        object.__setattr__(self, "trust", ArtifactTrust(self.trust))
        if not self.artifact_type.strip() or not self.mime_type.strip():
            raise ValueError("artifact_type and mime_type are required")
        if self.size_bytes < 0:
            raise ValueError("artifact size cannot be negative")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("artifact sha256 must be a lowercase hexadecimal digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "artifact_type": self.artifact_type,
            "mime_type": self.mime_type,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "producer_task_id": self.producer_task_id,
            "producer_tool": self.producer_tool,
            "provenance": _json_safe(self.provenance),
            "trust": self.trust.value,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            artifact_id=str(value["artifact_id"]),
            run_id=str(value["run_id"]),
            artifact_type=str(value["artifact_type"]),
            mime_type=str(value["mime_type"]),
            relative_path=str(value["relative_path"]),
            sha256=str(value["sha256"]),
            size_bytes=int(value["size_bytes"]),
            producer_task_id=(
                str(value["producer_task_id"]) if value.get("producer_task_id") else None
            ),
            producer_tool=str(value["producer_tool"]) if value.get("producer_tool") else None,
            provenance=dict(value.get("provenance") or {}),
            trust=ArtifactTrust(str(value.get("trust", ArtifactTrust.INTERNAL.value))),
            created_at=str(value.get("created_at") or _utc_now()),
        )


@dataclass
class WorkflowRun:
    """Current state reconstructed from a workflow run's event stream."""

    session_id: str
    run_id: str
    workflow_slug: str
    trace_id: str
    workflow_contract: dict[str, Any] = field(default_factory=dict)
    status: RunStatus = RunStatus.SUBMITTED
    constraints: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    workflow_inputs: dict[str, str] = field(default_factory=dict)
    tasks: dict[str, TaskRecord] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    handoffs: list[HandoffEnvelope] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    error: ToolError | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        self.session_id = validate_identifier(self.session_id, field="session_id")
        self.run_id = validate_identifier(self.run_id, field="run_id")
        self.workflow_slug = sanitize_workflow_slug(self.workflow_slug)
        self.status = RunStatus(self.status)
        self.workflow_contract = dict(self.workflow_contract or {})
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported workflow schema version {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if not self.trace_id.strip():
            raise ValueError("trace_id cannot be empty")
        if self.workflow_contract:
            _validate_workflow_contract_snapshot(
                self.workflow_contract,
                expected_slug=self.workflow_slug,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "workflow_slug": self.workflow_slug,
            "trace_id": self.trace_id,
            "workflow_contract": _json_safe(self.workflow_contract),
            "status": self.status.value,
            "constraints": _json_safe(self.constraints),
            "budget": _json_safe(self.budget),
            "workflow_inputs": dict(sorted(self.workflow_inputs.items())),
            "tasks": [self.tasks[key].to_dict() for key in sorted(self.tasks)],
            "artifacts": [self.artifacts[key].to_dict() for key in sorted(self.artifacts)],
            "handoffs": [handoff.to_dict() for handoff in self.handoffs],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowRun":
        tasks = [TaskRecord.from_dict(item) for item in value.get("tasks", ())]
        artifacts = [ArtifactRecord.from_dict(item) for item in value.get("artifacts", ())]
        error = value.get("error")
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            session_id=str(value["session_id"]),
            run_id=str(value["run_id"]),
            workflow_slug=str(value["workflow_slug"]),
            trace_id=str(value["trace_id"]),
            workflow_contract=dict(value.get("workflow_contract") or {}),
            status=RunStatus(str(value.get("status", RunStatus.SUBMITTED.value))),
            constraints=dict(value.get("constraints") or {}),
            budget=dict(value.get("budget") or {}),
            workflow_inputs={
                str(name): validate_identifier(artifact_id, field="workflow input artifact_id")
                for name, artifact_id in dict(value.get("workflow_inputs") or {}).items()
            },
            tasks={task.task_id: task for task in tasks},
            artifacts={artifact.artifact_id: artifact for artifact in artifacts},
            handoffs=[HandoffEnvelope.from_dict(item) for item in value.get("handoffs", ())],
            created_at=str(value.get("created_at") or _utc_now()),
            updated_at=str(value.get("updated_at") or _utc_now()),
            error=ToolError.from_dict(error) if isinstance(error, Mapping) else None,
        )


@dataclass(frozen=True)
class WorkflowEvent:
    """One immutable workflow event object."""

    event_id: str
    sequence: int
    run_id: str
    event_type: str
    timestamp: str
    payload: Mapping[str, Any]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "payload": _json_safe(self.payload),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkflowEvent":
        schema_version = int(value.get("schema_version", 0))
        if schema_version != SCHEMA_VERSION:
            raise EventReplayError(
                f"unsupported event schema version {schema_version}; expected {SCHEMA_VERSION}"
            )
        return cls(
            schema_version=schema_version,
            event_id=validate_identifier(value["event_id"], field="event_id"),
            sequence=int(value["sequence"]),
            run_id=validate_identifier(value["run_id"], field="run_id"),
            event_type=str(value["event_type"]),
            timestamp=str(value["timestamp"]),
            payload=dict(value.get("payload") or {}),
        )


class RunContext:
    """Event-sourced workflow runtime bound to one session-scoped run."""

    def __init__(
        self,
        *,
        layout: OutputLayout,
        run: WorkflowRun | None = None,
        events: Sequence[WorkflowEvent] = (),
    ) -> None:
        self.layout = layout
        self.run = run
        self.events = list(events)
        self._event_lock = threading.RLock()

    @classmethod
    def create(
        cls,
        workflow_slug: str,
        *,
        session_state: dict[str, Any] | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        constraints: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> "RunContext":
        slug = sanitize_workflow_slug(workflow_slug)
        if session_id is not None:
            _activate_session(session_id)
        selected_run_id = run_id or f"{slug}-{uuid.uuid4().hex[:12]}"
        output_context = ensure_output_context(
            session_state,
            workflow_slug=slug,
            run_id=selected_run_id,
            session_id=session_id,
        )
        run = WorkflowRun(
            session_id=str(output_context["session_id"]),
            run_id=str(output_context["run_id"]),
            workflow_slug=str(output_context["workflow_slug"]),
            trace_id=trace_id or uuid.uuid4().hex,
            workflow_contract=_workflow_contract_snapshot(slug),
            constraints=dict(constraints or {}),
            budget=dict(budget or {}),
        )
        context = cls(
            layout=OutputLayout(run.session_id, run.run_id, run.workflow_slug),
        )
        context.append_event("run_created", {"run": run.to_dict()})
        context.bind_session_state(session_state)
        return context

    @classmethod
    def load(
        cls,
        run_id: str,
        *,
        session_id: str | None = None,
        verify_artifacts: bool = False,
    ) -> "RunContext":
        run_id = validate_identifier(run_id, field="run_id")
        resolved_session = validate_identifier(
            session_id or _current_session_id(), field="session_id"
        )
        _activate_session(resolved_session)
        provisional = OutputLayout(resolved_session, run_id, "workflow")
        events = _read_events(provisional)
        run = _replay(events)
        if run.session_id != resolved_session:
            raise EventReplayError(
                f"run {run_id!r} belongs to session {run.session_id!r}, not {resolved_session!r}"
            )
        context = cls(
            layout=OutputLayout(run.session_id, run.run_id, run.workflow_slug),
            run=run,
            events=events,
        )
        if verify_artifacts:
            context.verify_artifacts()
        return context

    @classmethod
    def resume(cls, run_id: str, *, session_id: str | None = None) -> "RunContext":
        """Load a run and verify all registered inputs before more work is scheduled."""

        return cls.load(run_id, session_id=session_id, verify_artifacts=True)

    @classmethod
    def from_session_state(
        cls,
        session_state: Mapping[str, Any],
        *,
        verify_artifacts: bool = False,
    ) -> "RunContext":
        """Load the run named by canonical ``session_state`` output context."""

        output_context = session_state.get(OUTPUT_CONTEXT_KEY)
        if not isinstance(output_context, Mapping):
            raise ValueError(f"session_state[{OUTPUT_CONTEXT_KEY!r}] is missing")
        return cls.load(
            str(output_context["run_id"]),
            session_id=str(output_context["session_id"]),
            verify_artifacts=verify_artifacts,
        )

    def bind_session_state(
        self,
        session_state: dict[str, Any] | None,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
    ) -> dict[str, Any]:
        """Expose serializable run identity and trace context to agent/tool code."""

        run = self._require_run()
        output_context = {
            "layout_version": LAYOUT_VERSION,
            "session_id": run.session_id,
            "run_id": run.run_id,
            "workflow_slug": run.workflow_slug,
            "trace_id": run.trace_id,
            "span_id": span_id or run.trace_id,
            "parent_span_id": parent_span_id,
        }
        if isinstance(session_state, dict):
            session_state[OUTPUT_CONTEXT_KEY] = output_context
        return output_context

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        precondition: (
            Callable[
                [WorkflowRun | None, Sequence[WorkflowEvent]],
                None,
            ]
            | None
        ) = None,
    ) -> WorkflowEvent:
        """Validate, persist, and apply one immutable event.

        Snapshots are refreshed on a best-effort basis after the immutable
        event commits. A replaceable snapshot failure must never report the
        committed event as failed.
        """

        if not event_type.strip():
            raise ValueError("event_type cannot be empty")
        with self._event_lock, _run_lock(self.layout):
            for attempt in range(3):
                self._synchronize_events(pending_event_type=event_type)
                if precondition is not None:
                    precondition(self.run, self.events)
                sequence = len(self.events) + 1
                event = WorkflowEvent(
                    # A deterministic name reserves the sequence atomically.
                    # The run id scopes event identity, so a UUID is unnecessary.
                    event_id=f"{sequence:08d}",
                    sequence=sequence,
                    run_id=self.layout.run_id,
                    event_type=event_type,
                    timestamp=_utc_now(),
                    payload=dict(payload or {}),
                )
                # Apply to a detached copy so a failed immutable write cannot
                # mutate the caller-visible snapshot.
                base = WorkflowRun.from_dict(self.run.to_dict()) if self.run is not None else None
                candidate = _apply_event(base, event)
                try:
                    _write_event(self.layout, event)
                except _EventConflictError as exc:
                    if attempt == 2:
                        raise WorkflowRuntimeError(
                            f"could not reserve the next event for run "
                            f"{self.layout.run_id!r} after concurrent writes"
                        ) from exc
                    continue
                self.events.append(event)
                self.run = candidate
                try:
                    self._write_snapshots()
                except Exception:
                    logger.warning(
                        "Workflow event %s committed for run %s, but derived "
                        "snapshot refresh failed",
                        event.event_id,
                        self.layout.run_id,
                        exc_info=True,
                    )
                return event
        raise AssertionError("unreachable event append state")  # pragma: no cover

    def _synchronize_events(self, *, pending_event_type: str) -> None:
        """Refresh a stale writer and reject replaced or duplicate run streams."""

        paths = _list_event_paths(self.layout)
        if not paths:
            if self.events or self.run is not None:
                raise EventReplayError(f"event stream for run {self.layout.run_id!r} disappeared")
            return

        authoritative = _read_events(self.layout)
        if not self.events:
            if pending_event_type == "run_created":
                raise WorkflowRuntimeError(f"workflow run {self.layout.run_id!r} already exists")
            if self.run is None:
                raise EventReplayError(
                    f"run {self.layout.run_id!r} has events but no local run state"
                )
        else:
            known_ids = [event.event_id for event in self.events]
            authoritative_prefix = [event.event_id for event in authoritative[: len(known_ids)]]
            if authoritative_prefix != known_ids:
                raise EventReplayError(f"event stream for run {self.layout.run_id!r} was replaced")
        self.events = authoritative
        self.run = _replay(authoritative)

    def refresh(self, *, verify_artifacts: bool = False) -> WorkflowRun:
        """Refresh this context from its authoritative event stream."""

        with _run_lock(self.layout):
            self._synchronize_events(pending_event_type="refresh")
            run = self._require_run()
            if verify_artifacts:
                self.verify_artifacts()
            return run

    def pending_tool_invocations(
        self,
        *,
        task_id: str | None = None,
        domain_only: bool = False,
    ) -> tuple[str, ...]:
        """Return authoritative MCP calls that have not reached a terminal event."""

        self.refresh()
        return tuple(
            _pending_tool_invocations(
                self.events,
                task_id=task_id,
                authoritative_run=self.run,
                domain_only=domain_only,
            )
        )

    def abandon_tool_invocation(
        self,
        span_id: str,
        *,
        reason: str,
    ) -> WorkflowEvent:
        """Reconcile one confirmed-orphaned MCP invocation after a crash."""

        selected_span = validate_identifier(span_id, field="span_id")
        selected_reason = str(reason).strip()
        if not selected_reason:
            raise ValueError("orphaned invocation recovery requires a reason")
        if len(selected_reason) > 1000:
            raise ValueError("orphaned invocation recovery reason exceeds 1000 characters")
        self.refresh()
        started = next(
            (
                event.payload
                for event in reversed(self.events)
                if event.event_type == "tool_progress"
                and event.payload.get("stage") == "started"
                and str(event.payload.get("span_id") or "") == selected_span
            ),
            None,
        )
        if started is None or not _is_pending_tool_span(
            self.events,
            selected_span,
            authoritative_run=self.run,
        ):
            raise InvalidTransitionError(f"MCP invocation {selected_span!r} is not pending")
        tool_name = str(started.get("tool_name") or "")
        if not tool_name or tool_name.startswith("workflow_"):
            raise InvalidTransitionError("only orphaned domain-tool invocations may be reconciled")
        payload = {
            key: started.get(key)
            for key in (
                "runtime",
                "session_id",
                "run_id",
                "workflow_slug",
                "trace_id",
                "span_id",
                "parent_span_id",
                "tool_name",
                "task_id",
                "role",
                "profile",
                "attempt",
                "max_attempts",
                "task_attempt",
                "handoff_id",
            )
        }
        payload.update(
            {
                "stage": "abandoned",
                "cached": False,
                "message": selected_reason,
                "recovery": {
                    "confirmed_not_running": True,
                    "reason": selected_reason,
                },
            }
        )

        def require_still_orphaned(
            authoritative_run: WorkflowRun | None,
            events: Sequence[WorkflowEvent],
        ) -> None:
            if not _is_pending_tool_span(
                events,
                selected_span,
                authoritative_run=authoritative_run,
            ):
                raise InvalidTransitionError(
                    f"MCP invocation {selected_span!r} is no longer pending"
                )

        return self.append_event(
            "tool_progress",
            payload,
            precondition=require_still_orphaned,
        )

    def transition_run(
        self,
        status: RunStatus | str,
        *,
        reason: str | None = None,
        error: ToolError | None = None,
    ) -> WorkflowRun:
        target = RunStatus(status)
        if target is RunStatus.FAILED and error is None:
            raise ValueError("failed run transition requires a structured error")
        if error is not None and target is not RunStatus.FAILED:
            raise ValueError("run errors may only accompany a failed transition")
        run = self._require_run()
        if target is RunStatus.COMPLETED and run.status is RunStatus.RUNNING:
            self.verify_artifacts()
            reasons = _completion_gaps(run)
            if reasons:
                raise InvalidTransitionError(
                    "cannot mark workflow completed before its contracts are "
                    f"satisfied: {'; '.join(reasons)}"
                )
        payload: dict[str, Any] = {"status": target.value}
        if reason:
            payload["reason"] = reason
        if error:
            payload["error"] = error.to_dict()

        def reject_inflight_calls(
            authoritative_run: WorkflowRun | None,
            events: Sequence[WorkflowEvent],
        ) -> None:
            pending = _pending_tool_invocations(
                events,
                authoritative_run=authoritative_run,
                domain_only=True,
            )
            if pending:
                raise InvalidTransitionError(
                    "workflow status cannot change while MCP tool calls are in "
                    f"flight: {', '.join(pending)}"
                )

        self.append_event(
            "run_status_changed",
            payload,
            precondition=reject_inflight_calls,
        )
        return self._require_run()

    def add_task(self, task: TaskRecord) -> TaskRecord:
        run = self._require_run()
        _require_mutable_run(run, operation="add a task")
        if task.task_id in run.tasks:
            raise ValueError(f"duplicate task_id: {task.task_id}")
        if task.status is not TaskStatus.PENDING:
            raise ValueError("new tasks must start in pending status")
        if task.attempts != 0 or task.output_artifact_ids or task.error is not None:
            raise ValueError("new tasks cannot contain attempts, outputs, or an error")
        contract = _run_task_contract(run, task.task_id)
        if run.workflow_contract.get("tasks") and contract is None:
            raise ValueError(
                f"task {task.task_id!r} is not declared by workflow " f"{run.workflow_slug!r}"
            )
        if contract is not None:
            required_role = str(contract["role"])
            required_profile = str(contract["profile"])
            if task.role != required_role:
                raise ValueError(
                    f"task {task.task_id!r} requires role {required_role!r}, " f"not {task.role!r}"
                )
            if task.profile != required_profile:
                raise ValueError(
                    f"task {task.task_id!r} requires profile {required_profile!r}, "
                    f"not {task.profile!r}"
                )
        unknown_inputs = sorted(set(task.input_artifact_ids) - set(run.artifacts))
        if unknown_inputs:
            raise ValueError(
                f"task references unknown input artifacts: {', '.join(unknown_inputs)}"
            )
        self.append_event("task_created", {"task": task.to_dict()})
        return self._require_run().tasks[task.task_id]

    def transition_task(
        self,
        task_id: str,
        status: TaskStatus | str,
        *,
        error: ToolError | None = None,
    ) -> TaskRecord:
        task_id = validate_identifier(task_id, field="task_id")
        target = TaskStatus(status)
        if target is TaskStatus.FAILED and error is None:
            raise ValueError("failed task transition requires a structured error")
        if error is not None and target is not TaskStatus.FAILED:
            raise ValueError("task errors may only accompany a failed transition")
        run = self._require_run()
        _require_mutable_run(run, operation="transition a task")
        if target is TaskStatus.RUNNING:
            if run.status is not RunStatus.RUNNING:
                raise InvalidTransitionError(
                    f"task {task_id!r} cannot run while workflow status is " f"{run.status.value!r}"
                )
            contract = _run_task_contract(run, task_id)
            if contract is not None:
                incomplete = [
                    dependency
                    for dependency in contract.get("depends_on", ())
                    if dependency not in run.tasks
                    or run.tasks[dependency].status is not TaskStatus.COMPLETED
                ]
                if incomplete:
                    raise InvalidTransitionError(
                        f"task {task_id!r} cannot run before dependencies complete: "
                        + ", ".join(incomplete)
                    )
                input_gaps = _task_input_gaps(run, task_id)
                if input_gaps:
                    raise InvalidTransitionError(
                        f"task {task_id!r} cannot run before required input "
                        f"artifacts are registered: {'; '.join(input_gaps)}"
                    )
                task = run.tasks.get(task_id)
                handoff_gap = _task_handoff_gap(run, task) if task is not None else None
                if handoff_gap:
                    raise InvalidTransitionError(handoff_gap)
            self.verify_task_inputs(task_id)
        if target is TaskStatus.COMPLETED:
            required_outputs = _run_required_task_outputs(run, task_id)
            produced = {
                artifact.artifact_type
                for artifact in run.artifacts.values()
                if artifact.producer_task_id == task_id
            }
            missing_outputs = sorted(set(required_outputs) - produced)
            if missing_outputs:
                raise InvalidTransitionError(
                    f"task {task_id!r} cannot complete before required artifacts "
                    f"are registered: {', '.join(missing_outputs)}"
                )
        payload: dict[str, Any] = {"task_id": task_id, "status": target.value}
        if error:
            payload["error"] = error.to_dict()

        def reject_inflight_calls(
            authoritative_run: WorkflowRun | None,
            events: Sequence[WorkflowEvent],
        ) -> None:
            authoritative_task = (
                authoritative_run.tasks.get(task_id) if authoritative_run is not None else None
            )
            selected_attempt = (
                int(authoritative_task.attempts) + (1 if target is TaskStatus.RUNNING else 0)
                if authoritative_task is not None
                else None
            )
            pending = _pending_tool_invocations(
                events,
                task_id=task_id,
                task_attempt=selected_attempt,
                domain_only=True,
            )
            if pending:
                raise InvalidTransitionError(
                    f"task {task_id!r} cannot change status while MCP tool calls "
                    f"are in flight: {', '.join(pending)}"
                )

        self.append_event(
            "task_status_changed",
            payload,
            precondition=reject_inflight_calls,
        )
        return self._require_run().tasks[task_id]

    def record_handoff(
        self,
        envelope: Any,
        *,
        input_artifact_contracts: Sequence[str] | None = None,
    ) -> HandoffEnvelope:
        """Persist a runtime or duck-typed agent handoff without importing agents."""

        if isinstance(envelope, HandoffEnvelope):
            handoff = envelope
        else:
            if hasattr(envelope, "to_dict"):
                value = envelope.to_dict()
            elif hasattr(envelope, "as_dict"):
                value = envelope.as_dict()
            elif isinstance(envelope, Mapping):
                value = envelope
            else:
                raise TypeError("handoff must be a mapping or expose to_dict()/as_dict()")
            handoff = HandoffEnvelope.from_dict(value)
        handoff = replace(handoff, created_at=_utc_now())
        run = self._require_run()
        _require_mutable_run(run, operation="record a handoff")
        if any(item.handoff_id == handoff.handoff_id for item in run.handoffs):
            raise ValueError(f"duplicate handoff_id: {handoff.handoff_id}")
        handoff = _validate_and_normalize_handoff(
            run,
            handoff,
            input_artifact_contracts=input_artifact_contracts,
        )
        for artifact_id in handoff.input_artifact_ids:
            self.verify_artifact(artifact_id)
        self.append_event("handoff_recorded", {"handoff": handoff.to_dict()})
        return self._require_run().handoffs[-1]

    def record_progress(
        self,
        *,
        task_id: str,
        current: int | float,
        total: int | float | None = None,
        message: str | None = None,
    ) -> WorkflowEvent:
        run = self._require_run()
        _require_mutable_run(run, operation="record task progress")
        selected_task_id = validate_identifier(task_id, field="task_id")
        if selected_task_id not in run.tasks:
            raise ValueError(f"progress references unknown task {selected_task_id!r}")
        if (
            isinstance(current, bool)
            or not isinstance(current, (int, float))
            or not math.isfinite(current)
            or current < 0
        ):
            raise ValueError("task progress current must be a finite non-negative number")
        if total is not None and (
            isinstance(total, bool)
            or not isinstance(total, (int, float))
            or not math.isfinite(total)
            or total <= 0
        ):
            raise ValueError("task progress total must be a finite positive number")
        if total is not None and current > total:
            raise ValueError("task progress current cannot exceed total")
        payload: dict[str, Any] = {
            "task_id": selected_task_id,
            "current": current,
        }
        if total is not None:
            payload["total"] = total
        if message:
            payload["message"] = message
        return self.append_event("task_progress", payload)

    def register_artifact(
        self,
        path: str,
        *,
        artifact_type: str,
        mime_type: str,
        artifact_id: str | None = None,
        producer_task_id: str | None = None,
        active_task_id: str | None = None,
        producer_tool: str | None = None,
        provenance: Mapping[str, Any] | None = None,
        trust: ArtifactTrust | str | None = None,
    ) -> ArtifactRecord:
        """Checksum and register a file that is contained by this run's root."""

        run = self._require_run()
        _require_mutable_run(run, operation="register an artifact")
        if active_task_id is not None:
            active_task_id = validate_identifier(active_task_id, field="active_task_id")
        if producer_task_id is not None:
            producer_task_id = validate_identifier(
                producer_task_id,
                field="producer_task_id",
            )
            try:
                producer_task = run.tasks[producer_task_id]
            except KeyError as exc:
                raise ValueError(f"artifact references unknown task {producer_task_id!r}") from exc
            if _run_task_contract(run, producer_task_id) is not None:
                if active_task_id != producer_task_id:
                    raise ValueError(
                        f"producer_task_id {producer_task_id!r} does not match the "
                        f"active task {active_task_id!r}"
                    )
                if producer_task.status is not TaskStatus.RUNNING:
                    raise ValueError(
                        f"producer task {producer_task_id!r} must be an active "
                        "RUNNING catalog task"
                    )
        relative = _normalize_artifact_input_path(self.layout, path)
        digest, size = _digest_artifact(self.layout, relative)
        existing = next(
            (artifact for artifact in run.artifacts.values() if artifact.relative_path == relative),
            None,
        )
        if existing is not None:
            if digest != existing.sha256 or size != existing.size_bytes:
                raise ArtifactIntegrityError(
                    f"registered artifact path {relative!r} changed after registration"
                )
            if artifact_id is not None and artifact_id != existing.artifact_id:
                raise ValueError(
                    f"artifact path {relative!r} is already registered as "
                    f"{existing.artifact_id!r}"
                )
            if artifact_type != existing.artifact_type or mime_type != existing.mime_type:
                raise ValueError(
                    f"artifact path {relative!r} is already registered with "
                    "different type or MIME metadata"
                )
            if producer_task_id is not None and producer_task_id != existing.producer_task_id:
                raise ValueError(
                    f"artifact path {relative!r} is already registered to a "
                    "different producer task"
                )
            if producer_tool is not None and producer_tool != existing.producer_tool:
                raise ValueError(
                    f"artifact path {relative!r} is already registered to a "
                    "different producer tool"
                )
            if provenance is not None and _json_safe(provenance) != _json_safe(existing.provenance):
                raise ValueError(
                    f"artifact path {relative!r} is already registered with " "different provenance"
                )
            if trust is not None and ArtifactTrust(trust) is not existing.trust:
                raise ValueError(
                    f"artifact path {relative!r} is already registered with "
                    "a different trust classification"
                )
            return existing
        selected_id = artifact_id or f"artifact-{uuid.uuid4().hex[:12]}"
        record = ArtifactRecord(
            artifact_id=selected_id,
            run_id=run.run_id,
            artifact_type=artifact_type,
            mime_type=mime_type,
            relative_path=relative,
            sha256=digest,
            size_bytes=size,
            producer_task_id=producer_task_id,
            producer_tool=producer_tool,
            provenance=dict(provenance or {}),
            trust=ArtifactTrust(trust or ArtifactTrust.INTERNAL),
        )
        if record.artifact_id in run.artifacts:
            raise ValueError(f"duplicate artifact_id: {record.artifact_id}")
        existing_input_id = run.workflow_inputs.get(record.artifact_type)
        if existing_input_id is not None and existing_input_id != record.artifact_id:
            raise ValueError(
                f"workflow input contract {record.artifact_type!r} is already bound "
                f"to artifact {existing_input_id!r}"
            )
        if producer_task_id:
            task_contract = _run_task_contract(run, producer_task_id)
            if task_contract is not None and artifact_type not in set(
                task_contract.get("output_artifacts", ())
            ):
                raise ValueError(
                    f"task {producer_task_id!r} is not contracted to produce "
                    f"artifact type {artifact_type!r}"
                )

        def authorize_registration(
            authoritative_run: WorkflowRun | None,
            events: Sequence[WorkflowEvent],
        ) -> None:
            _authorize_artifact_registration(
                authoritative_run,
                events,
                record=record,
                active_task_id=active_task_id,
            )

        self.append_event(
            "artifact_registered",
            {"artifact": record.to_dict()},
            precondition=authorize_registration,
        )
        return self._require_run().artifacts[record.artifact_id]

    def verify_artifact(self, artifact_id: str) -> ArtifactRecord:
        run = self._require_run()
        try:
            record = run.artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"unknown artifact_id: {artifact_id}") from exc
        try:
            digest, size = _digest_artifact(self.layout, record.relative_path)
        except FileNotFoundError as exc:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id!r} is missing: {record.relative_path}"
            ) from exc
        if digest != record.sha256 or size != record.size_bytes:
            raise ArtifactIntegrityError(
                f"artifact {artifact_id!r} failed checksum/size verification"
            )
        return record

    def verify_artifacts(self) -> list[ArtifactRecord]:
        return [self.verify_artifact(key) for key in sorted(self._require_run().artifacts)]

    def verify_task_inputs(self, task_id: str) -> list[ArtifactRecord]:
        """Verify the selected input artifacts for one task before execution."""

        selected_task_id = validate_identifier(task_id, field="task_id")
        run = self._require_run()
        try:
            task = run.tasks[selected_task_id]
        except KeyError as exc:
            raise ValueError(f"unknown task_id: {selected_task_id}") from exc
        input_gaps = _task_input_gaps(run, selected_task_id)
        if input_gaps:
            raise InvalidTransitionError(
                f"task {selected_task_id!r} has invalid required input "
                f"artifacts: {'; '.join(input_gaps)}"
            )
        return [
            self.verify_artifact(artifact_id)
            for artifact_id in dict.fromkeys(task.input_artifact_ids)
        ]

    def rebuild_snapshots(self) -> WorkflowRun:
        """Replay immutable events and replace both derived snapshots."""

        events = _read_events(self.layout)
        run = _replay(events)
        self.events = events
        self.run = run
        self.layout = OutputLayout(run.session_id, run.run_id, run.workflow_slug)
        self._write_snapshots()
        return run

    def complete(
        self,
        *,
        required_artifact_types: Sequence[str] | None = None,
        required_task_ids: Sequence[str] | None = None,
    ) -> WorkflowRun:
        """Complete the run only after its catalog contracts are satisfied.

        Catalog requirements are authoritative and explicit requirements can
        only make them stricter. Ad-hoc runs require every registered task and
        any explicitly requested artifacts to be complete.
        """

        run = self._require_run()
        if run.status in {RunStatus.COMPLETED, RunStatus.PARTIAL}:
            return run
        if run.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
            raise InvalidTransitionError(
                f"cannot complete terminal run in status {run.status.value!r}"
            )
        if run.status is not RunStatus.RUNNING:
            run = self.transition_run(
                RunStatus.RUNNING,
                reason="completion validation requested",
            )
        self.verify_artifacts()
        reasons = _completion_gaps(
            run,
            required_artifact_types=required_artifact_types,
            required_task_ids=required_task_ids,
        )
        if reasons:
            return self.transition_run(RunStatus.PARTIAL, reason="; ".join(reasons))
        return self.transition_run(RunStatus.COMPLETED)

    def manifest_payload(self) -> dict[str, Any]:
        run = self._require_run()
        payload = run.to_dict()
        payload["event_count"] = len(self.events)
        payload["last_event_id"] = self.events[-1].event_id if self.events else None
        return payload

    def artifact_index_payload(self) -> dict[str, Any]:
        run = self._require_run()
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": run.session_id,
            "run_id": run.run_id,
            "workflow_slug": run.workflow_slug,
            "artifacts": [run.artifacts[key].to_dict() for key in sorted(run.artifacts)],
        }

    def _write_snapshots(self) -> None:
        _write_json(self.layout.manifest_rel_path, self.manifest_payload())
        _write_json(self.layout.artifact_index_rel_path, self.artifact_index_payload())

    def _require_run(self) -> WorkflowRun:
        if self.run is None:
            raise WorkflowRuntimeError("workflow run has not been initialized")
        return self.run


# Descriptive alias used by callers that prefer a runtime-oriented name.
WorkflowRuntime = RunContext


class RunStore:
    """Read-only adapter used by trajectory replay and external evaluators."""

    def __init__(self, *, session_id: str | None = None) -> None:
        self.session_id = session_id

    def events(self, run_id: str) -> list[dict[str, Any]]:
        context = RunContext.load(run_id, session_id=self.session_id)
        return [event.to_dict() for event in context.events]

    def rebuild(self, run_id: str) -> WorkflowRun:
        context = RunContext.load(run_id, session_id=self.session_id)
        return context.rebuild_snapshots()


def _apply_event(run: WorkflowRun | None, event: WorkflowEvent) -> WorkflowRun:
    if event.event_type == "run_created":
        if run is not None:
            raise EventReplayError("run_created may only be the first event")
        value = event.payload.get("run")
        if not isinstance(value, Mapping):
            raise EventReplayError("run_created is missing its run payload")
        created = WorkflowRun.from_dict(value)
        if created.run_id != event.run_id:
            raise EventReplayError("run_created run_id does not match event run_id")
        if (
            created.status is not RunStatus.SUBMITTED
            or created.workflow_inputs
            or created.tasks
            or created.artifacts
            or created.handoffs
            or created.error is not None
        ):
            raise EventReplayError("run_created must contain a pristine submitted run")
        return created

    if run is None:
        raise EventReplayError("the first workflow event must be run_created")
    if event.run_id != run.run_id:
        raise EventReplayError("event run_id does not match the workflow run")

    if event.event_type == "run_status_changed":
        target = RunStatus(str(event.payload["status"]))
        _validate_transition(run.status, target, _RUN_TRANSITIONS, subject="run")
        error = event.payload.get("error")
        if target is RunStatus.FAILED and not isinstance(error, Mapping):
            raise EventReplayError("failed run status event requires a structured error")
        if target is not RunStatus.FAILED and error is not None:
            raise EventReplayError("run error is only valid for failed status")
        if target is RunStatus.COMPLETED:
            reasons = _completion_gaps(run)
            if reasons:
                raise InvalidTransitionError(
                    "cannot mark workflow completed before its contracts are "
                    f"satisfied: {'; '.join(reasons)}"
                )
        run.status = target
        run.error = ToolError.from_dict(error) if isinstance(error, Mapping) else None
        run.updated_at = event.timestamp
        return run

    if event.event_type == "task_created":
        _require_mutable_run(run, operation="add a task")
        value = event.payload.get("task")
        if not isinstance(value, Mapping):
            raise EventReplayError("task_created is missing its task payload")
        task = TaskRecord.from_dict(value)
        if task.task_id in run.tasks:
            raise EventReplayError(f"duplicate task_id in event stream: {task.task_id}")
        if (
            task.status is not TaskStatus.PENDING
            or task.attempts != 0
            or task.output_artifact_ids
            or task.error is not None
        ):
            raise EventReplayError("task_created must contain a pristine pending task")
        contract = _run_task_contract(run, task.task_id)
        if run.workflow_contract.get("tasks") and contract is None:
            raise EventReplayError(
                f"task {task.task_id!r} is not declared by workflow " f"{run.workflow_slug!r}"
            )
        if contract is not None and (
            task.role != str(contract["role"]) or task.profile != str(contract["profile"])
        ):
            raise EventReplayError(
                f"task {task.task_id!r} role/profile does not match its workflow contract"
            )
        unknown_inputs = sorted(set(task.input_artifact_ids) - set(run.artifacts))
        if unknown_inputs:
            raise EventReplayError(
                f"task references unknown input artifacts: {', '.join(unknown_inputs)}"
            )
        run.tasks[task.task_id] = task
        run.updated_at = event.timestamp
        return run

    if event.event_type == "task_status_changed":
        _require_mutable_run(run, operation="transition a task")
        task_id = str(event.payload["task_id"])
        try:
            task = run.tasks[task_id]
        except KeyError as exc:
            raise EventReplayError(f"status event references unknown task {task_id!r}") from exc
        target = TaskStatus(str(event.payload["status"]))
        _validate_transition(task.status, target, _TASK_TRANSITIONS, subject=f"task {task_id}")
        error = event.payload.get("error")
        if target is TaskStatus.FAILED and not isinstance(error, Mapping):
            raise EventReplayError("failed task status event requires a structured error")
        if target is not TaskStatus.FAILED and error is not None:
            raise EventReplayError("task error is only valid for failed status")
        if target is TaskStatus.RUNNING:
            if run.status is not RunStatus.RUNNING:
                raise InvalidTransitionError(
                    f"task {task_id!r} cannot run while workflow status is " f"{run.status.value!r}"
                )
            contract = _run_task_contract(run, task_id)
            if contract is not None:
                incomplete = [
                    dependency
                    for dependency in contract.get("depends_on", ())
                    if dependency not in run.tasks
                    or run.tasks[dependency].status is not TaskStatus.COMPLETED
                ]
                if incomplete:
                    raise InvalidTransitionError(
                        f"task {task_id!r} cannot run before dependencies complete: "
                        + ", ".join(incomplete)
                    )
                input_gaps = _task_input_gaps(run, task_id)
                if input_gaps:
                    raise InvalidTransitionError(
                        f"task {task_id!r} cannot run before required input "
                        f"artifacts are registered: {'; '.join(input_gaps)}"
                    )
                handoff_gap = _task_handoff_gap(run, task)
                if handoff_gap:
                    raise InvalidTransitionError(handoff_gap)
        if target is TaskStatus.COMPLETED:
            required_outputs = _run_required_task_outputs(run, task_id)
            produced = {
                artifact.artifact_type
                for artifact in run.artifacts.values()
                if artifact.producer_task_id == task_id
            }
            missing_outputs = sorted(set(required_outputs) - produced)
            if missing_outputs:
                raise InvalidTransitionError(
                    f"task {task_id!r} cannot complete before required artifacts "
                    f"are registered: {', '.join(missing_outputs)}"
                )
        if target is TaskStatus.RUNNING:
            task.attempts += 1
        task.status = target
        task.error = ToolError.from_dict(error) if isinstance(error, Mapping) else None
        task.updated_at = event.timestamp
        run.updated_at = event.timestamp
        return run

    if event.event_type == "handoff_recorded":
        _require_mutable_run(run, operation="record a handoff")
        value = event.payload.get("handoff")
        if not isinstance(value, Mapping):
            raise EventReplayError("handoff_recorded is missing its handoff payload")
        handoff = HandoffEnvelope.from_dict(value)
        if any(item.handoff_id == handoff.handoff_id for item in run.handoffs):
            raise EventReplayError(f"duplicate handoff_id in event stream: {handoff.handoff_id}")
        try:
            normalized = _validate_and_normalize_handoff(run, handoff)
        except InvalidTransitionError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise EventReplayError(str(exc)) from exc
        if (
            handoff.task_attempt != normalized.task_attempt
            or handoff.input_artifact_ids != normalized.input_artifact_ids
        ):
            raise EventReplayError(
                f"handoff {handoff.handoff_id!r} is not the canonical validated "
                "task contract recorded by the runtime"
            )
        task = run.tasks[handoff.task_id]
        task.input_artifact_ids = list(handoff.input_artifact_ids)
        task.updated_at = event.timestamp
        run.handoffs.append(normalized)
        run.updated_at = event.timestamp
        return run

    if event.event_type == "artifact_registered":
        _require_mutable_run(run, operation="register an artifact")
        value = event.payload.get("artifact")
        if not isinstance(value, Mapping):
            raise EventReplayError("artifact_registered is missing its artifact payload")
        artifact = ArtifactRecord.from_dict(value)
        if artifact.run_id != run.run_id:
            raise EventReplayError("artifact run_id does not match its workflow run")
        if artifact.artifact_id in run.artifacts:
            raise EventReplayError(f"duplicate artifact_id in event stream: {artifact.artifact_id}")
        if any(
            existing.relative_path == artifact.relative_path for existing in run.artifacts.values()
        ):
            raise EventReplayError(
                f"duplicate artifact path in event stream: {artifact.relative_path}"
            )
        existing_input_id = run.workflow_inputs.get(artifact.artifact_type)
        if existing_input_id is not None and existing_input_id != artifact.artifact_id:
            raise EventReplayError(
                f"workflow input contract {artifact.artifact_type!r} is already bound "
                f"to artifact {existing_input_id!r}"
            )
        if artifact.producer_task_id:
            try:
                task = run.tasks[artifact.producer_task_id]
            except KeyError as exc:
                raise EventReplayError(
                    f"artifact references unknown task {artifact.producer_task_id!r}"
                ) from exc
            task_contract = _run_task_contract(run, artifact.producer_task_id)
            if task_contract is not None and artifact.artifact_type not in set(
                task_contract.get("output_artifacts", ())
            ):
                raise EventReplayError(
                    f"task {artifact.producer_task_id!r} is not contracted to "
                    f"produce artifact type {artifact.artifact_type!r}"
                )
            if task_contract is not None and task.status is not TaskStatus.RUNNING:
                raise EventReplayError(
                    f"catalog task {artifact.producer_task_id!r} registered output "
                    f"while its status was {task.status.value!r}, not 'running'"
                )
        run.artifacts[artifact.artifact_id] = artifact
        if artifact.artifact_type in _run_workflow_input_contracts(run):
            run.workflow_inputs[artifact.artifact_type] = artifact.artifact_id
        if artifact.producer_task_id:
            task = run.tasks[artifact.producer_task_id]
            if artifact.artifact_id not in task.output_artifact_ids:
                task.output_artifact_ids.append(artifact.artifact_id)
                task.updated_at = event.timestamp
        run.updated_at = event.timestamp
        return run

    # Progress, tool-call, approval, and future observational events are still
    # durable even when they do not alter the current snapshot.
    run.updated_at = event.timestamp
    return run


def _validate_transition(
    current: Enum,
    target: Enum,
    transitions: Mapping[Enum, frozenset[Enum]],
    *,
    subject: str,
) -> None:
    if target not in transitions[current]:
        raise InvalidTransitionError(
            f"illegal {subject} transition: {current.value} -> {target.value}"
        )


def _write_event(layout: OutputLayout, event: WorkflowEvent) -> None:
    path = layout.event_rel_path(event.event_id)
    try:
        with S3.open_atomic(path, "x") as handle:
            json.dump(event.to_dict(), handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
    except FileExistsError as exc:
        raise _EventConflictError(f"event path already exists: {path}") from exc
    except PermissionError as exc:
        if "destination already exists" in str(exc):
            raise _EventConflictError(f"event path already exists: {path}") from exc
        raise


def _write_json(path: str, payload: Mapping[str, Any]) -> None:
    with S3.open(path, "w") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_events(layout: OutputLayout) -> list[WorkflowEvent]:
    event_paths = _list_event_paths(layout)
    if not event_paths:
        raise EventReplayError(f"workflow run {layout.run_id!r} has no events")
    events: list[WorkflowEvent] = []
    for path in event_paths:
        with S3.open(path, "r") as handle:
            lines = [line for line in handle.read().splitlines() if line.strip()]
        if len(lines) != 1:
            raise EventReplayError(f"event segment must contain exactly one JSON line: {path}")
        try:
            raw = json.loads(lines[0])
        except json.JSONDecodeError as exc:
            raise EventReplayError(f"invalid JSON event segment: {path}") from exc
        if not isinstance(raw, Mapping):
            raise EventReplayError(f"event segment must contain a JSON object: {path}")
        try:
            event = WorkflowEvent.from_dict(raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise EventReplayError(f"invalid workflow event record: {path}") from exc
        expected_name = f"{event.event_id}.jsonl"
        if PurePosixPath(path).name != expected_name:
            raise EventReplayError(f"event id does not match segment name: {path}")
        events.append(event)

    events.sort(key=lambda item: item.sequence)
    sequences = [event.sequence for event in events]
    if sequences != list(range(1, len(events) + 1)):
        raise EventReplayError(f"event sequence is not contiguous: {sequences}")
    if len({event.event_id for event in events}) != len(events):
        raise EventReplayError("event stream contains duplicate event ids")
    return events


def _replay(events: Sequence[WorkflowEvent]) -> WorkflowRun:
    run: WorkflowRun | None = None
    for event in events:
        try:
            run = _apply_event(run, event)
        except EventReplayError:
            raise
        except (KeyError, TypeError, ValueError, WorkflowRuntimeError) as exc:
            raise EventReplayError(
                f"could not apply event {event.event_id!r} ({event.event_type})"
            ) from exc
    if run is None:
        raise EventReplayError("workflow event stream is empty")
    return run


def _list_event_paths(layout: OutputLayout) -> list[str]:
    glob_url = S3.path(f"{layout.events_rel_path}/*.jsonl")
    options: dict[str, Any] = {}
    if is_s3_enabled():
        options = get_s3_config().to_storage_options()
    fs, fs_path = fsspec.core.url_to_fs(glob_url, **options)
    matches = fs.glob(fs_path)
    paths: list[str] = []
    for match in matches:
        name = PurePosixPath(str(match)).name
        if not name.endswith(".jsonl"):
            continue
        event_id = name[: -len(".jsonl")]
        paths.append(layout.event_rel_path(event_id))
    return sorted(paths)


def _digest_artifact(layout: OutputLayout, relative_path: str) -> tuple[str, int]:
    """Digest an artifact through the backend's race-safe read boundary."""

    digest = hashlib.sha256()
    size = 0
    if is_s3_enabled():
        handle = S3.open(layout.artifact_rel_path(relative_path), "rb")
    else:
        handle = open_local_run_artifact(layout, relative_path)
    with handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _normalize_artifact_input_path(layout: OutputLayout, path: str) -> str:
    """Accept a run-relative path or this backend's expanded local run path."""

    if not is_s3_enabled() and isinstance(path, str) and path.strip():
        candidate = Path(path.strip())
        run_root = Path(S3.path(layout.run_root))
        try:
            backend_relative = candidate.relative_to(run_root)
        except ValueError:
            pass
        else:
            return normalize_run_relative_path(
                layout.run_id,
                backend_relative.as_posix(),
            )
    return normalize_run_relative_path(layout.run_id, path)


def _current_session_id() -> str:
    return PurePosixPath(S3.current_prefix().strip("/")).name or "session"


def _workflow_contract_snapshot(workflow_slug: str) -> dict[str, Any]:
    """Pin the selected workflow and all transitive catalog dependencies."""

    try:
        from .registry import get_workflow

        workflow = get_workflow(workflow_slug)
    except KeyError:
        return {}

    def snapshot(spec: Any, ancestry: tuple[str, ...]) -> dict[str, Any]:
        if spec.slug in ancestry:
            cycle = " -> ".join((*ancestry, spec.slug))
            raise ValueError(f"workflow dependency cycle detected while snapshotting: {cycle}")
        payload = spec.as_dict(include_content=True)
        payload["dependency_contracts"] = [
            snapshot(get_workflow(dependency), (*ancestry, spec.slug))
            for dependency in spec.depends_on
        ]
        payload["contract_schema_version"] = 1
        payload["contract_sha256"] = _workflow_contract_digest(payload)
        return payload

    return snapshot(workflow, ())


def _workflow_contract_digest(contract: Mapping[str, Any]) -> str:
    payload = dict(contract)
    payload.pop("contract_sha256", None)
    payload.pop("contract_schema_version", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_workflow_contract_snapshot(
    contract: Mapping[str, Any],
    *,
    expected_slug: str,
) -> None:
    if contract.get("slug") != expected_slug:
        raise ValueError("workflow contract slug does not match the run or dependency")
    if contract.get("contract_schema_version") != 1:
        raise ValueError("unsupported workflow contract snapshot schema")
    expected = str(contract.get("contract_sha256") or "")
    if expected != _workflow_contract_digest(contract):
        raise ValueError("workflow contract snapshot checksum mismatch")

    declared_dependencies = tuple(str(item) for item in contract.get("depends_on", ()))
    if len(set(declared_dependencies)) != len(declared_dependencies):
        raise ValueError("workflow contract contains duplicate dependencies")
    dependency_contracts = contract.get("dependency_contracts", ())
    if not isinstance(dependency_contracts, Sequence) or isinstance(
        dependency_contracts, (str, bytes)
    ):
        raise ValueError("workflow dependency contract snapshots must be a sequence")
    snapshots: dict[str, Mapping[str, Any]] = {}
    for dependency in dependency_contracts:
        if not isinstance(dependency, Mapping):
            raise ValueError("workflow dependency contract snapshot must be a mapping")
        dependency_slug = str(dependency.get("slug") or "")
        if dependency_slug in snapshots:
            raise ValueError(
                f"duplicate workflow dependency contract snapshot: {dependency_slug!r}"
            )
        snapshots[dependency_slug] = dependency
    if set(snapshots) != set(declared_dependencies):
        raise ValueError("workflow dependency snapshots do not match declared dependencies")
    for dependency_slug in declared_dependencies:
        _validate_workflow_contract_snapshot(
            snapshots[dependency_slug],
            expected_slug=dependency_slug,
        )


def _run_task_contract(
    run: WorkflowRun,
    task_id: str,
) -> Mapping[str, Any] | None:
    contract = run.workflow_contract
    return next(
        (
            task
            for task in contract.get("tasks", ())
            if isinstance(task, Mapping) and task.get("task_id") == task_id
        ),
        None,
    )


def _run_required_task_outputs(
    run: WorkflowRun,
    task_id: str,
) -> tuple[str, ...]:
    contract = run.workflow_contract
    task = _run_task_contract(run, task_id)
    if task is None:
        return ()
    globally_required = {
        str(artifact["name"])
        for artifact in contract.get("output_artifacts", ())
        if isinstance(artifact, Mapping) and artifact.get("name") and artifact.get("required", True)
    }
    return tuple(
        str(name) for name in task.get("output_artifacts", ()) if str(name) in globally_required
    )


def _run_workflow_input_contracts(
    run: WorkflowRun,
) -> dict[str, Mapping[str, Any]]:
    return {
        str(item["name"]): item
        for item in run.workflow_contract.get("input_artifacts", ())
        if isinstance(item, Mapping) and item.get("name")
    }


def _task_input_gaps(run: WorkflowRun, task_id: str) -> list[str]:
    """Return missing or ambiguous required inputs for one catalog task."""

    task_contract = _run_task_contract(run, task_id)
    if task_contract is None:
        return []
    task_record = run.tasks.get(task_id)
    selected_ids = tuple(task_record.input_artifact_ids) if task_record is not None else ()
    selected_artifacts = [
        run.artifacts[artifact_id] for artifact_id in selected_ids if artifact_id in run.artifacts
    ]
    workflow_inputs = _run_workflow_input_contracts(run)
    workflow_outputs = {
        str(item["name"]): item
        for item in run.workflow_contract.get("output_artifacts", ())
        if isinstance(item, Mapping) and item.get("name")
    }
    gaps: list[str] = []
    unknown_selected = sorted(set(selected_ids) - set(run.artifacts))
    if unknown_selected:
        gaps.append("unknown selected artifacts " + ", ".join(unknown_selected))
    declared_types = tuple(str(item) for item in task_contract.get("input_artifacts", ()))
    undeclared_selected = sorted(
        artifact.artifact_id
        for artifact in selected_artifacts
        if artifact.artifact_type not in set(declared_types)
    )
    if undeclared_selected:
        gaps.append("selected artifacts outside task contract " + ", ".join(undeclared_selected))

    for artifact_type in declared_types:
        selected_matches = [
            artifact.artifact_id
            for artifact in selected_artifacts
            if artifact.artifact_type == artifact_type
        ]
        input_contract = workflow_inputs.get(artifact_type)
        if input_contract is not None:
            artifact_id = run.workflow_inputs.get(artifact_type)
            artifact = run.artifacts.get(artifact_id or "")
            if len(selected_matches) > 1:
                gaps.append(
                    f"ambiguous selected workflow input {artifact_type} "
                    f"({', '.join(sorted(selected_matches))})"
                )
            elif selected_matches and selected_matches != [artifact_id]:
                gaps.append(
                    f"selected workflow input {artifact_type} does not match "
                    f"pinned artifact {artifact_id}"
                )
            if input_contract.get("required", True) and (
                artifact is None or artifact.artifact_type != artifact_type
            ):
                gaps.append(f"missing workflow input {artifact_type}")
            continue

        output_contract = workflow_outputs.get(artifact_type)
        if output_contract is not None and not output_contract.get("required", True):
            if len(selected_matches) > 1:
                gaps.append(
                    f"ambiguous selected upstream artifact {artifact_type} "
                    f"({', '.join(sorted(selected_matches))})"
                )
            continue
        if len(selected_matches) == 1:
            continue
        if len(selected_matches) > 1:
            gaps.append(
                f"ambiguous selected upstream artifact {artifact_type} "
                f"({', '.join(sorted(selected_matches))})"
            )
            continue
        matches = [
            artifact.artifact_id
            for artifact in run.artifacts.values()
            if artifact.artifact_type == artifact_type
        ]
        if not matches:
            gaps.append(f"missing upstream artifact {artifact_type}")
        elif len(matches) > 1:
            gaps.append(
                f"ambiguous upstream artifact {artifact_type} " f"({', '.join(sorted(matches))})"
            )
    return gaps


def _validate_and_normalize_handoff(
    run: WorkflowRun,
    handoff: HandoffEnvelope,
    *,
    input_artifact_contracts: Sequence[str] | None = None,
) -> HandoffEnvelope:
    """Validate a handoff against its durable task and pinned catalog contract."""

    if handoff.run_id != run.run_id:
        raise ValueError("handoff run_id does not match this run")
    if handoff.workflow_slug != run.workflow_slug:
        raise ValueError("handoff workflow_slug does not match this run")
    try:
        task = run.tasks[handoff.task_id]
    except KeyError as exc:
        raise ValueError(f"handoff references unknown task {handoff.task_id!r}") from exc
    if handoff.receiver_role != task.role:
        raise ValueError(
            f"handoff receiver {handoff.receiver_role!r} does not match " f"task role {task.role!r}"
        )
    if run.workflow_contract.get("tasks") and task.status is TaskStatus.RUNNING:
        raise InvalidTransitionError(
            f"catalog task {task.task_id!r} handoff must be recorded before "
            "the task starts or after it leaves RUNNING"
        )
    if task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.CANCELLED,
        TaskStatus.SKIPPED,
    }:
        raise InvalidTransitionError(
            f"cannot hand off terminal task {handoff.task_id!r} ({task.status.value})"
        )
    if handoff.task_attempt is not None and handoff.task_attempt != task.attempts:
        raise ValueError(
            f"handoff task_attempt {handoff.task_attempt} does not match "
            f"task {task.task_id!r} attempt {task.attempts}"
        )
    normalized = replace(handoff, task_attempt=task.attempts)
    _validate_catalog_handoff_budget(run, normalized)

    explicit_ids = tuple(normalized.input_artifact_ids)
    if len(set(explicit_ids)) != len(explicit_ids):
        raise ValueError("handoff input_artifact_ids cannot contain duplicates")
    unknown_artifacts = sorted(set(explicit_ids) - set(run.artifacts))
    if unknown_artifacts:
        raise ValueError(f"handoff references unknown artifacts: {', '.join(unknown_artifacts)}")

    task_contract = _run_task_contract(run, task.task_id)
    if task_contract is None:
        return normalized

    declared_inputs = tuple(str(item) for item in task_contract.get("input_artifacts", ()))
    if input_artifact_contracts is not None:
        _require_exact_handoff_values(
            task.task_id,
            field="input artifact contracts",
            requested=tuple(str(item) for item in input_artifact_contracts),
            declared=declared_inputs,
        )
    _require_exact_handoff_values(
        task.task_id,
        field="required capabilities",
        requested=normalized.required_capabilities,
        declared=tuple(str(item) for item in task_contract.get("required_tools", ())),
    )
    _require_exact_handoff_values(
        task.task_id,
        field="expected output artifacts",
        requested=normalized.expected_output_artifacts,
        declared=tuple(str(item) for item in task_contract.get("output_artifacts", ())),
    )
    _require_exact_handoff_values(
        task.task_id,
        field="acceptance criteria",
        requested=normalized.acceptance_criteria,
        declared=tuple(str(item) for item in task_contract.get("acceptance_criteria", ())),
    )
    resolved_ids = _resolve_catalog_handoff_inputs(
        run,
        task_id=task.task_id,
        declared_inputs=declared_inputs,
        explicit_ids=explicit_ids,
        allow_implicit_resolution=input_artifact_contracts is not None,
    )
    return replace(normalized, input_artifact_ids=resolved_ids)


def _require_exact_handoff_values(
    task_id: str,
    *,
    field: str,
    requested: Sequence[str],
    declared: Sequence[str],
) -> None:
    """Require one handoff field to be an exact, duplicate-free contract set."""

    requested_values = tuple(str(item) for item in requested)
    declared_values = tuple(str(item) for item in declared)
    duplicates = sorted(
        value for value in set(requested_values) if requested_values.count(value) > 1
    )
    if duplicates:
        raise ValueError(
            f"handoff for task {task_id!r} {field} contains duplicates: " + ", ".join(duplicates)
        )
    missing = sorted(set(declared_values) - set(requested_values))
    undeclared = sorted(set(requested_values) - set(declared_values))
    if not missing and not undeclared:
        return
    details: list[str] = []
    if missing:
        details.append("missing: " + "; ".join(missing))
    if undeclared:
        details.append("undeclared: " + "; ".join(undeclared))
    raise ValueError(
        f"handoff for task {task_id!r} {field} must match the pinned "
        f"task contract ({'; '.join(details)})"
    )


def _resolve_catalog_handoff_inputs(
    run: WorkflowRun,
    *,
    task_id: str,
    declared_inputs: Sequence[str],
    explicit_ids: Sequence[str],
    allow_implicit_resolution: bool,
) -> tuple[str, ...]:
    """Resolve one artifact per declared task input, honoring explicit choices."""

    workflow_inputs = _run_workflow_input_contracts(run)
    workflow_outputs = {
        str(item["name"]): item
        for item in run.workflow_contract.get("output_artifacts", ())
        if isinstance(item, Mapping) and item.get("name")
    }
    declared_set = set(declared_inputs)
    wrong_explicit_types = sorted(
        f"{artifact_id} ({run.artifacts[artifact_id].artifact_type})"
        for artifact_id in explicit_ids
        if run.artifacts[artifact_id].artifact_type not in declared_set
    )
    if wrong_explicit_types:
        raise ValueError(
            f"handoff for task {task_id!r} has explicit input artifacts whose "
            "types are not declared by the pinned task contract: " + ", ".join(wrong_explicit_types)
        )

    resolved: list[str] = []
    missing_required: list[str] = []
    for artifact_type in declared_inputs:
        explicit_matches = [
            artifact_id
            for artifact_id in explicit_ids
            if run.artifacts[artifact_id].artifact_type == artifact_type
        ]
        if len(explicit_matches) > 1:
            raise ValueError(
                f"handoff input artifact contract {artifact_type!r} has multiple "
                f"explicit selections: {', '.join(sorted(explicit_matches))}"
            )

        input_contract = workflow_inputs.get(artifact_type)
        if input_contract is not None:
            bound_id = run.workflow_inputs.get(artifact_type)
            if explicit_matches and explicit_matches[0] != bound_id:
                raise ValueError(
                    f"handoff for task {task_id!r} references workflow input "
                    f"{artifact_type!r} that is not the run's pinned artifact "
                    f"{bound_id!r}"
                )
            if explicit_matches:
                resolved.append(explicit_matches[0])
            elif allow_implicit_resolution and bound_id is not None and bound_id in run.artifacts:
                resolved.append(bound_id)
            elif input_contract.get("required", True):
                missing_required.append(artifact_type)
            continue

        if explicit_matches:
            resolved.append(explicit_matches[0])
            continue
        if not allow_implicit_resolution:
            output_contract = workflow_outputs.get(artifact_type)
            if output_contract is None or output_contract.get("required", True):
                missing_required.append(artifact_type)
            continue
        matches = [
            artifact.artifact_id
            for artifact in run.artifacts.values()
            if artifact.artifact_type == artifact_type
        ]
        if len(matches) > 1:
            raise ValueError(
                f"handoff input artifact contract {artifact_type!r} is ambiguous; "
                f"registered artifact IDs: {', '.join(sorted(matches))}"
            )
        if matches:
            resolved.append(matches[0])
            continue
        output_contract = workflow_outputs.get(artifact_type)
        if output_contract is None or output_contract.get("required", True):
            missing_required.append(artifact_type)

    if missing_required:
        raise ValueError(
            f"handoff for task {task_id!r} is missing required registered input "
            f"artifact contracts: {', '.join(sorted(missing_required))}"
        )
    return tuple(resolved)


def _task_handoff_gap(run: WorkflowRun, task: TaskRecord) -> str | None:
    """Require a current handoff before a catalog task starts or retries."""

    if not run.workflow_contract.get("tasks"):
        return None
    task_handoffs = [
        item
        for item in run.handoffs
        if item.task_id == task.task_id and item.task_attempt == task.attempts
    ]
    if task.status is TaskStatus.PENDING and not task_handoffs:
        return (
            f"task {task.task_id!r} cannot run before a validated structured " "handoff is recorded"
        )
    if task.status is TaskStatus.FAILED and not task_handoffs:
        return (
            f"task {task.task_id!r} cannot retry before a fresh validated "
            "structured handoff is recorded after the failed attempt"
        )
    if task.status is TaskStatus.INPUT_REQUIRED and not task_handoffs:
        return (
            f"task {task.task_id!r} cannot resume after requesting input before "
            "a fresh validated structured handoff is recorded for the next attempt"
        )
    return None


def _validate_catalog_handoff_budget(
    run: WorkflowRun,
    handoff: HandoffEnvelope,
) -> None:
    """Require a complete bounded delegation budget for catalog tasks."""

    if not run.workflow_contract.get("tasks"):
        return
    required = ("max_tokens", "max_tool_calls", "timeout_seconds")
    missing = [name for name in required if name not in handoff.budget]
    if missing:
        raise ValueError(
            f"catalog task handoff {handoff.handoff_id!r} is missing required "
            f"budget fields: {', '.join(missing)}"
        )
    for name in ("max_tokens", "max_tool_calls"):
        value = handoff.budget[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"catalog task handoff budget {name} must be a positive integer")
    timeout = handoff.budget["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError(
            "catalog task handoff budget timeout_seconds must be a finite positive number"
        )


def _completion_gaps(
    run: WorkflowRun,
    *,
    required_artifact_types: Sequence[str] | None = None,
    required_task_ids: Sequence[str] | None = None,
) -> list[str]:
    """Return unmet completion requirements without allowing contract weakening."""

    artifact_requirements = {
        str(item["name"])
        for item in run.workflow_contract.get("output_artifacts", ())
        if isinstance(item, Mapping) and item.get("name") and item.get("required", True)
    }
    required_workflow_inputs = {
        name
        for name, contract in _run_workflow_input_contracts(run).items()
        if contract.get("required", True)
    }
    task_requirements = {
        str(task["task_id"])
        for task in run.workflow_contract.get("tasks", ())
        if isinstance(task, Mapping) and task.get("task_id")
    }
    # Ad-hoc runs have no catalog DAG, but tasks that were explicitly created
    # are still part of the run and must reach completion.
    if not run.workflow_contract.get("tasks"):
        task_requirements.update(run.tasks)
    if required_artifact_types is not None:
        artifact_requirements.update(
            str(item).strip() for item in required_artifact_types if str(item).strip()
        )
    if required_task_ids is not None:
        task_requirements.update(
            validate_identifier(item, field="task_id") for item in required_task_ids
        )

    present = {artifact.artifact_type for artifact in run.artifacts.values()}
    missing_artifacts = sorted(artifact_requirements - present)
    missing_workflow_inputs = sorted(
        name
        for name in required_workflow_inputs
        if name not in run.workflow_inputs
        or run.workflow_inputs[name] not in run.artifacts
        or run.artifacts[run.workflow_inputs[name]].artifact_type != name
    )
    incomplete_tasks = sorted(
        task_id
        for task_id in task_requirements
        if task_id not in run.tasks or run.tasks[task_id].status is not TaskStatus.COMPLETED
    )
    reasons: list[str] = []
    if incomplete_tasks:
        reasons.append(f"incomplete required tasks: {', '.join(incomplete_tasks)}")
    if missing_workflow_inputs:
        reasons.append(
            "missing required workflow input artifacts: " + ", ".join(missing_workflow_inputs)
        )
    if missing_artifacts:
        reasons.append(f"missing required artifact types: {', '.join(missing_artifacts)}")
    return reasons


def _require_mutable_run(run: WorkflowRun, *, operation: str) -> None:
    if run.status in _TERMINAL_RUN_STATUSES:
        raise InvalidTransitionError(
            f"cannot {operation} after run entered terminal status {run.status.value!r}"
        )


def _activate_session(session_id: str) -> None:
    """Bind relative storage operations to an explicitly selected session."""

    selected = validate_identifier(session_id, field="session_id")
    if _current_session_id() != selected:
        S3.set_session_prefix(f"sessions/{selected}")


def _run_lock(layout: OutputLayout) -> threading.RLock:
    """Return the in-process lock shared by every context for one run."""

    key = (layout.session_id, layout.run_id)
    with _RUN_LOCKS_GUARD:
        return _RUN_LOCKS.setdefault(key, threading.RLock())


def _pending_tool_invocations(
    events: Sequence[WorkflowEvent],
    *,
    task_id: str | None = None,
    task_attempt: int | None = None,
    authoritative_run: WorkflowRun | None = None,
    domain_only: bool = False,
) -> list[str]:
    """Return MCP spans that started but have no durable terminal progress event."""

    pending: dict[str, str] = {}
    for event in events:
        if event.event_type != "tool_progress":
            continue
        payload = event.payload
        event_task_id = str(payload.get("task_id") or "")
        if task_id is not None and event_task_id != task_id:
            continue
        expected_attempt = task_attempt
        if (
            expected_attempt is None
            and authoritative_run is not None
            and event_task_id in authoritative_run.tasks
        ):
            expected_attempt = int(authoritative_run.tasks[event_task_id].attempts)
        event_attempt = payload.get("task_attempt")
        if (
            expected_attempt is not None
            and event_attempt is not None
            and int(event_attempt) != expected_attempt
        ):
            continue
        span_id = str(payload.get("span_id") or "")
        tool_name = str(payload.get("tool_name") or "unknown_tool")
        if domain_only and tool_name.startswith("workflow_"):
            continue
        key = span_id or f"{event_task_id}:{tool_name}"
        stage = str(payload.get("stage") or "")
        if stage == "started":
            pending[key] = tool_name
        elif stage in {
            "abandoned",
            "cache_hit",
            "cancelled",
            "completed",
            "failed",
            "result_accepted",
        }:
            pending.pop(key, None)
    return sorted(f"{tool_name} ({span_id})" for span_id, tool_name in pending.items())


def _is_pending_tool_span(
    events: Sequence[WorkflowEvent],
    span_id: str,
    *,
    authoritative_run: WorkflowRun | None,
) -> bool:
    """Return whether one exact domain span is pending for its current attempt."""

    pending = False
    for event in events:
        if event.event_type != "tool_progress":
            continue
        payload = event.payload
        if str(payload.get("span_id") or "") != span_id:
            continue
        tool_name = str(payload.get("tool_name") or "")
        if tool_name.startswith("workflow_"):
            continue
        task_id = str(payload.get("task_id") or "")
        if authoritative_run is not None and task_id in authoritative_run.tasks:
            event_attempt = payload.get("task_attempt")
            if event_attempt is not None and int(event_attempt) != int(
                authoritative_run.tasks[task_id].attempts
            ):
                continue
        stage = str(payload.get("stage") or "")
        if stage == "started":
            pending = True
        elif stage in {
            "abandoned",
            "cache_hit",
            "cancelled",
            "completed",
            "failed",
            "result_accepted",
        }:
            pending = False
    return pending


def _authorize_artifact_registration(
    run: WorkflowRun | None,
    events: Sequence[WorkflowEvent],
    *,
    record: ArtifactRecord,
    active_task_id: str | None,
) -> None:
    """Keep artifact publication attributable while domain calls are in flight."""

    pending = _pending_tool_invocations(
        events,
        authoritative_run=run,
        domain_only=True,
    )
    if not pending:
        return

    provenance = record.provenance
    if provenance.get("registration") != "automatic":
        raise InvalidTransitionError(
            "artifacts cannot be registered while MCP domain tool calls are in "
            f"flight: {', '.join(pending)}"
        )
    span_id = str(provenance.get("invocation_span_id") or "")
    if not span_id:
        raise InvalidTransitionError(
            "automatic artifact registration requires its originating MCP span"
        )

    started: Mapping[str, Any] | None = None
    terminal = False
    for event in events:
        if event.event_type != "tool_progress":
            continue
        payload = event.payload
        if str(payload.get("span_id") or "") != span_id:
            continue
        stage = str(payload.get("stage") or "")
        if stage == "started":
            started = payload
            terminal = False
        elif stage in {
            "abandoned",
            "cache_hit",
            "cancelled",
            "completed",
            "failed",
            "result_accepted",
        }:
            terminal = True

    if started is None or terminal:
        raise InvalidTransitionError(
            "automatic artifact registration does not match an in-flight MCP invocation"
        )
    if str(started.get("tool_name") or "") != str(record.producer_tool or ""):
        raise InvalidTransitionError(
            "automatic artifact producer tool does not match its originating MCP invocation"
        )
    started_task_id = str(started.get("task_id") or "")
    if record.producer_task_id is not None and started_task_id != record.producer_task_id:
        raise InvalidTransitionError(
            "automatic artifact producer task does not match its originating MCP invocation"
        )
    if record.producer_task_id is not None and active_task_id != record.producer_task_id:
        raise InvalidTransitionError(
            "automatic artifact registration is outside its active producer task"
        )
    if run is not None and record.producer_task_id is not None:
        task = run.tasks.get(record.producer_task_id)
        started_attempt = started.get("task_attempt")
        if (
            task is None
            or task.status is not TaskStatus.RUNNING
            or started_attempt is None
            or int(started_attempt) != int(task.attempts)
        ):
            raise InvalidTransitionError(
                "automatic artifact registration belongs to a stale task attempt"
            )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("non-finite floats cannot be persisted")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "as_dict"):
        return _json_safe(value.as_dict())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    return tuple(str(item).strip() for item in value if str(item).strip())
