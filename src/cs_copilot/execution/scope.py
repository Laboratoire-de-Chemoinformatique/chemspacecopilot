"""Invocation attribution, catalog-task authorization, and handoff budgets."""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .context import ExecutionContext, _optional_str, is_ad_hoc_run
from .errors import ToolErrorCode, ToolExecutionError
from .spec import ToolSpec, _is_control_plane_spec


@dataclass(frozen=True)
class _InvocationScope:
    """Immutable role/task attribution captured before a tool can await."""

    run_id: str | None
    task_id: str | None
    role: str | None
    profile: str | None
    task_attempt: int | None = None
    catalog_task: bool = False
    handoff_id: str | None = None
    handoff_created_at: str | None = None
    max_tool_calls: int | None = None
    timeout_seconds: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "role": self.role,
            "profile": self.profile,
            "task_attempt": self.task_attempt,
            "handoff_id": self.handoff_id,
        }


def _capture_invocation_scope(
    spec: ToolSpec,
    ctx: ExecutionContext,
) -> _InvocationScope:
    """Capture authoritative task attribution without granting authorization."""

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    task_id = _optional_str(state.get("active_task_id"))
    role = _optional_str(state.get("active_role"))
    profile = _optional_str(state.get("active_profile")) or _optional_str(state.get("mcp_profile"))
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    task = run.tasks.get(task_id) if run is not None and task_id is not None else None
    if task is not None:
        role = str(task.role)
        profile = str(task.profile)
    if _is_control_plane_spec(spec):
        role = "supervisor"
        profile = _optional_str(state.get("mcp_profile")) or profile
    return _InvocationScope(
        run_id=(str(run.run_id) if run is not None else None),
        task_id=task_id,
        role=role,
        profile=profile,
        task_attempt=(int(task.attempts) if task is not None else None),
    )


def _authorize_invocation(
    spec: ToolSpec,
    ctx: ExecutionContext,
) -> _InvocationScope:
    """Enforce pinned catalog task, role, and tool boundaries for domain calls."""

    scope = _capture_invocation_scope(spec, ctx)
    if _is_control_plane_spec(spec):
        return scope
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if run is None or is_ad_hoc_run(run):
        return scope
    if getattr(run.status, "value", str(run.status)) != "running":
        raise ToolExecutionError(
            "a catalog workflow domain tool requires an active RUNNING workflow run",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    workflow_contract = run.workflow_contract if isinstance(run.workflow_contract, Mapping) else {}
    task_contracts = workflow_contract.get("tasks")
    if not isinstance(task_contracts, list) or not task_contracts:
        # Legacy/taskless catalog workflows retain the startup profile boundary.
        return scope

    state = ctx.session_state if isinstance(ctx.session_state, dict) else {}
    task_id = _optional_str(state.get("active_task_id"))
    task = run.tasks.get(task_id) if task_id is not None else None
    if task is None or getattr(task.status, "value", str(task.status)) != "running":
        raise ToolExecutionError(
            "a catalog workflow domain tool requires an active RUNNING task",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    if spec.roles and task.role not in spec.roles:
        raise ToolExecutionError(
            f"active role {task.role!r} is not allowed to call {spec.mcp_name}",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    if spec.profiles and task.profile not in spec.profiles:
        raise ToolExecutionError(
            f"active task profile {task.profile!r} does not expose {spec.mcp_name}",
            code=ToolErrorCode.PERMISSION_DENIED,
        )

    contract = next(
        (
            item
            for item in task_contracts
            if isinstance(item, Mapping) and str(item.get("task_id") or "") == task.task_id
        ),
        None,
    )
    if contract is None:
        raise ToolExecutionError(
            f"active task {task.task_id!r} is not declared by the pinned workflow contract",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    required_tools = contract.get("required_tools")
    optional_tools = workflow_contract.get("optional_tools")
    allowed_tools = {str(item) for item in required_tools or ()} | {
        str(item) for item in optional_tools or ()
    }
    if spec.mcp_name not in allowed_tools:
        raise ToolExecutionError(
            f"tool {spec.mcp_name!r} is outside active task {task.task_id!r}'s allowlist",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    handoff = next(
        (item for item in reversed(run.handoffs) if item.task_id == task.task_id),
        None,
    )
    if handoff is None:
        raise ToolExecutionError(
            f"active task {task.task_id!r} has no structured handoff",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    expected_handoff_attempt = max(0, int(task.attempts) - 1)
    if handoff.task_attempt != expected_handoff_attempt:
        raise ToolExecutionError(
            f"active task {task.task_id!r} has no handoff for its current attempt",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    budget = handoff.budget if isinstance(handoff.budget, Mapping) else {}
    return _InvocationScope(
        run_id=str(run.run_id),
        task_id=task.task_id,
        role=task.role,
        profile=task.profile,
        task_attempt=int(task.attempts),
        catalog_task=True,
        handoff_id=handoff.handoff_id,
        handoff_created_at=handoff.created_at,
        max_tool_calls=_required_positive_int_budget(budget, "max_tool_calls"),
        timeout_seconds=_required_positive_float_budget(budget, "timeout_seconds"),
    )


def _assert_invocation_epoch_current(
    spec: ToolSpec,
    ctx: ExecutionContext,
    captured: _InvocationScope,
) -> None:
    """Reject stale catalog results after a task attempt or handoff changes."""

    if not captured.catalog_task:
        return
    run_context = getattr(ctx, "run_context", None)
    if run_context is not None and hasattr(run_context, "refresh"):
        run_context.refresh()
    current = _authorize_invocation(spec, ctx)
    captured_epoch = (
        captured.run_id,
        captured.task_id,
        captured.task_attempt,
        captured.handoff_id,
    )
    current_epoch = (
        current.run_id,
        current.task_id,
        current.task_attempt,
        current.handoff_id,
    )
    if current_epoch != captured_epoch:
        raise ToolExecutionError(
            "the workflow task attempt or structured handoff changed while the "
            f"{spec.mcp_name} call was running; its stale result was discarded",
            code=ToolErrorCode.PERMISSION_DENIED,
        )


def _enforce_execution_budget(
    ctx: ExecutionContext,
    invocation_scope: _InvocationScope,
    *,
    events: typing.Sequence[Any] | None = None,
) -> None:
    run_context = getattr(ctx, "run_context", None)
    selected_events = events if events is not None else getattr(run_context, "events", ())
    handoff_sequence = 0
    for event in selected_events:
        if (
            event.event_type == "handoff_recorded"
            and str(event.payload.get("handoff", {}).get("handoff_id") or "")
            == invocation_scope.handoff_id
        ):
            handoff_sequence = event.sequence
    if handoff_sequence <= 0:
        raise ToolExecutionError(
            "the active structured handoff is missing from the authoritative event stream",
            code=ToolErrorCode.PERMISSION_DENIED,
        )

    started_calls = sum(
        1
        for event in selected_events
        if event.sequence > handoff_sequence
        and event.event_type == "tool_progress"
        and str(event.payload.get("task_id") or "") == invocation_scope.task_id
        and str(event.payload.get("stage") or "") == "started"
    )
    if (
        invocation_scope.max_tool_calls is not None
        and started_calls >= invocation_scope.max_tool_calls
    ):
        raise ToolExecutionError(
            f"task {invocation_scope.task_id!r} exhausted its handoff budget of "
            f"{invocation_scope.max_tool_calls} tool calls",
            code=ToolErrorCode.RESOURCE_LIMIT,
        )

    _remaining_handoff_seconds(invocation_scope)


def _remaining_handoff_seconds(invocation_scope: _InvocationScope) -> float | None:
    """Return the live handoff allowance, failing once its deadline is reached."""

    if invocation_scope.timeout_seconds is None or invocation_scope.handoff_created_at is None:
        return None
    created = _parse_utc_timestamp(invocation_scope.handoff_created_at)
    elapsed = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    remaining = invocation_scope.timeout_seconds - elapsed
    if remaining <= 0:
        raise _handoff_timeout_error(invocation_scope)
    return remaining


def _handoff_timeout_error(invocation_scope: _InvocationScope) -> ToolExecutionError:
    return ToolExecutionError(
        f"task {invocation_scope.task_id!r} exceeded its handoff timeout of "
        f"{invocation_scope.timeout_seconds:g}s",
        code=ToolErrorCode.TIMEOUT,
    )


def _required_positive_int_budget(budget: Mapping[str, Any], name: str) -> int:
    value = budget.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ToolExecutionError(
            f"catalog handoff budget requires positive integer {name}",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    return value


def _required_positive_float_budget(budget: Mapping[str, Any], name: str) -> float:
    value = budget.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ToolExecutionError(
            f"catalog handoff budget requires positive finite {name}",
            code=ToolErrorCode.PERMISSION_DENIED,
        )
    return float(value)


def _parse_utc_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ToolExecutionError(
            "catalog handoff has an invalid created_at timestamp",
            code=ToolErrorCode.PERMISSION_DENIED,
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
