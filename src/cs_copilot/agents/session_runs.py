"""One durable v2 workflow run per in-process chat.

Chainlit and the CLI keep conversation history in Agno's SQLite store; the
scientific state of a chat — tool events, registered artifacts, handoffs — is
recorded in one ad-hoc ``agno-session`` workflow run per chat. The run is
created lazily before the first message is processed, reused when a chat
resumes, and never completed automatically because a chat can always resume.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
from pathlib import PurePosixPath
from typing import Any, Mapping

from cs_copilot.execution.context import AGNO_RUNTIME
from cs_copilot.storage import OUTPUT_CONTEXT_KEY, S3
from cs_copilot.workflows import (
    ArtifactIntegrityError,
    EventReplayError,
    RunContext,
    RunStatus,
    TaskStatus,
    ToolError,
)

from .execution_binding import (
    PROCESS_OWNER,
    ExecutionMode,
    attach_execution,
    execution_mode_from_env,
    get_binding,
)

logger = logging.getLogger(__name__)

AGNO_SESSION_SLUG = AGNO_RUNTIME.ad_hoc_workflow_slug
_TERMINAL_RUN_STATUSES = frozenset(
    {RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED, RunStatus.CANCELLED}
)
_TERMINAL_STAGES = frozenset(
    {"abandoned", "cache_hit", "cancelled", "completed", "failed", "result_accepted"}
)
# Team-owned keys a restored session state must not overwrite.
_TEAM_OWNED_KEYS = ("resource_profile", "agentic_contracts")


def ensure_agno_session_run(
    team: Any,
    *,
    session_id: str,
    mode: ExecutionMode | str | None = None,
) -> RunContext | None:
    """Return the chat's durable run, creating and binding it on first use.

    Returns ``None`` when the execution mode is ``off``. Idempotent: once a
    run is attached to ``team`` it is returned as is.
    """

    selected = execution_mode_from_env(mode if mode is not None else _team_mode(team))
    if selected is ExecutionMode.OFF:
        return None
    binding = get_binding(team)
    if binding is not None and binding.run_context is not None:
        return binding.run_context

    state = _team_state(team)
    enforce = selected is ExecutionMode.ENFORCE
    run_context = _load_chat_run(state, session_id, verify_artifacts=enforce)
    if run_context is None:
        previous = state.pop(OUTPUT_CONTEXT_KEY, None)
        if isinstance(previous, Mapping):
            legacy = state.setdefault("legacy_output_contexts", [])
            if isinstance(legacy, list):
                legacy.append(dict(previous))
        run_context = RunContext.create(
            AGNO_SESSION_SLUG,
            session_state=state,
            session_id=session_id,
        )
        run_context.transition_run(RunStatus.RUNNING, reason="chat session started")
    else:
        run_context.bind_session_state(state)
        reconcile_interrupted_work(run_context)
    if enforce:
        adopt_session_uploads(run_context, state)

    _publish_run_identity(team, state, run_context)
    attach_execution(team, run_context=run_context, mode=selected)
    return run_context


def restore_team_session_state(team: Any, saved_state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore a persisted chat state into the team's shared dict in place.

    Members hold a reference to the team's ``session_state`` dict; replacing
    the dict would leave them writing to a stale copy. Saved values win,
    except for the team-owned runtime keys of the freshly built team.
    """

    state = _team_state(team)
    preserved = {key: state[key] for key in _TEAM_OWNED_KEYS if key in state}
    state.clear()
    state.update(dict(saved_state))
    state.update(preserved)
    return state


def finalize_agno_turn(team: Any, *, failed: bool = False) -> None:
    """Close the ad-hoc tasks delegated during the turn that just ended."""

    binding = get_binding(team)
    if binding is None or binding.run_context is None:
        return
    run_context = binding.run_context
    for task_id in binding.take_turn_tasks():
        try:
            task = run_context.refresh().tasks.get(task_id)
            if task is None or task.status is not TaskStatus.RUNNING:
                continue
            if run_context.pending_tool_invocations(task_id=task_id, domain_only=True):
                continue
            if failed:
                run_context.transition_task(
                    task_id,
                    TaskStatus.FAILED,
                    error=ToolError(
                        code="internal",
                        message="the chat turn failed before the task finished",
                        retryable=True,
                    ),
                )
            else:
                run_context.transition_task(task_id, TaskStatus.COMPLETED)
        except Exception as exc:  # noqa: BLE001 - bookkeeping must not fail a turn
            logger.warning("Could not finalize ad-hoc task %s: %s", task_id, exc)
            binding.note_problem(f"finalize {task_id}: {exc}")


def reconcile_interrupted_work(run_context: RunContext) -> list[str]:
    """Close tool spans and tasks left open by a previous process of this chat.

    A span recorded by this runtime whose owning process is not this one can
    no longer finish: the chat server restarted or crashed. It is marked
    abandoned so it no longer blocks run, task, and artifact transitions, and
    tasks still marked running are failed as interrupted.
    """

    abandoned: list[str] = []
    for span_id, owner in _open_agno_spans(run_context):
        if owner == PROCESS_OWNER:
            continue
        try:
            run_context.abandon_tool_invocation(
                span_id,
                reason="the chat process that ran this tool call ended before it finished",
            )
            abandoned.append(span_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not abandon orphaned tool span %s: %s", span_id, exc)
    run = run_context.refresh()
    for task in list(run.tasks.values()):
        if task.status is not TaskStatus.RUNNING:
            continue
        if run_context.pending_tool_invocations(task_id=task.task_id, domain_only=True):
            continue
        try:
            run_context.transition_task(
                task.task_id,
                TaskStatus.FAILED,
                error=ToolError(
                    code="internal",
                    message="the chat session was interrupted while this task was running",
                    retryable=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not fail interrupted task %s: %s", task.task_id, exc)
    return abandoned


def store_chat_upload(
    run_context: RunContext,
    *,
    filename: str,
    content: bytes,
) -> tuple[str, str]:
    """Write a user upload inside the chat's run and register it as untrusted.

    Returns ``(storage_path, artifact_id)``. Identical re-uploads reuse the
    same content-addressed file and artifact.
    """

    name = PurePosixPath(str(filename).replace("\\", "/")).name or "upload"
    stem, dot, extension = name.partition(".")
    digest = hashlib.sha256(content).hexdigest()
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80] or "upload"
    relative = f"inputs/uploads/{safe_stem}-{digest[:8]}{dot}{extension}"
    storage_key = run_context.layout.artifact_rel_path(relative)
    existing = next(
        (
            record
            for record in run_context.run.artifacts.values()
            if record.relative_path == relative
        ),
        None,
    )
    if existing is None:
        with S3.open(storage_key, "wb") as handle:
            handle.write(content)
    record = run_context.register_artifact(
        relative,
        artifact_type="user_upload",
        mime_type=mimetypes.guess_type(name)[0] or "application/octet-stream",
        producer_tool="chat_upload",
        provenance={"registration": "manual", "source": "chat_upload", "original_name": name},
        trust="untrusted",
    )
    return S3.path(storage_key), record.artifact_id


def adopt_session_uploads(run_context: RunContext, state: dict[str, Any]) -> list[str]:
    """Move uploads stored outside the run (older chats) into it as artifacts.

    In enforce mode tools may only read registered artifacts, so uploads that
    predate the chat's run are copied into it and registered; the original
    files are left in place.
    """

    uploaded = state.get("uploaded_files")
    if not isinstance(uploaded, dict):
        return []
    artifacts = state.setdefault("uploaded_artifacts", {})
    run_prefix = S3.path(run_context.layout.run_root).rstrip("/") + "/"
    adopted: list[str] = []
    for name, path in list(uploaded.items()):
        if not isinstance(path, str) or path.startswith(run_prefix) or name in artifacts:
            continue
        try:
            with S3.open(path, "rb") as handle:
                content = handle.read()
            stored, artifact_id = store_chat_upload(run_context, filename=name, content=content)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not adopt upload %s into the chat run: %s", name, exc)
            continue
        uploaded[name] = stored
        artifacts[name] = artifact_id
        adopted.append(name)
    return adopted


def _load_chat_run(
    state: Mapping[str, Any],
    session_id: str,
    *,
    verify_artifacts: bool = False,
) -> RunContext | None:
    output_context = state.get(OUTPUT_CONTEXT_KEY)
    if not isinstance(output_context, Mapping):
        return None
    if output_context.get("workflow_slug") != AGNO_SESSION_SLUG:
        return None
    if str(output_context.get("session_id") or "") != session_id:
        return None
    run_id = str(output_context.get("run_id") or "")
    if not run_id:
        return None
    try:
        run_context = RunContext.load(
            run_id,
            session_id=session_id,
            verify_artifacts=verify_artifacts,
        )
    except (FileNotFoundError, EventReplayError, ArtifactIntegrityError, ValueError) as exc:
        logger.warning("Could not resume chat run %s; starting a new run: %s", run_id, exc)
        return None
    if run_context.run.status in _TERMINAL_RUN_STATUSES:
        return None
    if run_context.run.status is not RunStatus.RUNNING:
        run_context.transition_run(RunStatus.RUNNING, reason="chat session resumed")
    return run_context


def _open_agno_spans(run_context: RunContext) -> list[tuple[str, Any]]:
    run_context.refresh()
    started: dict[str, Any] = {}
    for event in run_context.events:
        if event.event_type != "tool_progress":
            continue
        payload = event.payload
        span_id = str(payload.get("span_id") or "")
        if not span_id:
            continue
        stage = str(payload.get("stage") or "")
        if stage == "started" and payload.get("runtime") == AGNO_RUNTIME.label:
            started[span_id] = payload.get("owner")
        elif stage in _TERMINAL_STAGES:
            started.pop(span_id, None)
    return list(started.items())


def _publish_run_identity(team: Any, state: dict[str, Any], run_context: RunContext) -> None:
    run = run_context.run
    contracts = state.get("agentic_contracts")
    if isinstance(contracts, dict):
        contracts["active_run"] = {
            "run_id": run.run_id,
            "workflow_slug": run.workflow_slug,
            "trace_id": run.trace_id,
        }
    state["current_run_id"] = run.run_id
    if hasattr(team, "run_context"):
        team.run_context = run_context


def _team_state(team: Any) -> dict[str, Any]:
    state = getattr(team, "session_state", None)
    if not isinstance(state, dict):
        state = {}
        team.session_state = state
    return state


def _team_mode(team: Any) -> ExecutionMode | str | None:
    return getattr(team, "execution_mode", None)
