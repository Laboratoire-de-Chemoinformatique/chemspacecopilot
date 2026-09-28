"""Golden-trajectory evaluation for artifact-backed workflow events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class GoldenTrajectory:
    """Semantic invariants expected from a workflow trace."""

    required_event_types: tuple[str, ...] = ()
    required_tool_order: tuple[str, ...] = ()
    required_artifact_types: tuple[str, ...] = ()
    preflight_requirements: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    role_tool_allowlists: Mapping[str, frozenset[str]] = field(default_factory=dict)
    allowed_terminal_statuses: tuple[str, ...] = ("completed",)
    max_duplicate_calls: int = 0


@dataclass(frozen=True)
class TrajectoryReport:
    """Result and metrics from evaluating one normalized event stream."""

    passed: bool
    violations: tuple[str, ...]
    terminal_status: str | None
    event_count: int
    tool_call_count: int
    handoff_count: int
    retry_count: int
    duplicate_call_count: int
    artifact_types: tuple[str, ...]
    total_tool_duration_ms: float

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        payload["artifact_types"] = list(self.artifact_types)
        return payload

    def metrics(self) -> dict[str, float]:
        return {
            "trajectory_passed": float(self.passed),
            "trajectory_events": float(self.event_count),
            "trajectory_tool_calls": float(self.tool_call_count),
            "trajectory_handoffs": float(self.handoff_count),
            "trajectory_retries": float(self.retry_count),
            "trajectory_duplicate_calls": float(self.duplicate_call_count),
            "trajectory_tool_duration_ms": float(self.total_tool_duration_ms),
        }


def golden_for_workflow(
    workflow_slug: str,
    *,
    allowed_terminal_statuses: Sequence[str] = ("completed",),
    max_duplicate_calls: int = 0,
) -> GoldenTrajectory:
    """Derive semantic trajectory invariants from one catalog workflow."""

    from cs_copilot.workflows import get_workflow

    workflow = get_workflow(workflow_slug)
    tool_order: list[str] = []
    role_tools: dict[str, set[str]] = {}
    task_tools: dict[str, tuple[str, ...]] = {}
    task_dependencies: dict[str, tuple[str, ...]] = {}
    for task in workflow.tasks:
        task_tools[task.task_id] = task.required_tools
        task_dependencies[task.task_id] = task.depends_on
        role_tools.setdefault(task.role, set()).update(task.required_tools)
        for tool_name in task.required_tools:
            if tool_name not in tool_order:
                tool_order.append(tool_name)

    # Optional tools are valid workflow capabilities too. Attribute each one
    # only to task roles whose declared profile and tool metadata both allow it.
    if workflow.tasks and workflow.optional_tools:
        from cs_copilot.mcp.tools_registry import all_specs

        optional_specs = {
            spec.mcp_name: spec for spec in all_specs() if spec.mcp_name in workflow.optional_tools
        }
        for task in workflow.tasks:
            for tool_name, spec in optional_specs.items():
                if task.profile in spec.profiles and task.role in spec.roles:
                    role_tools.setdefault(task.role, set()).add(tool_name)

    preflight_tools = set(workflow.preflight_tools)
    preflight_requirements: dict[str, tuple[str, ...]] = {}
    for task in workflow.tasks:
        ancestors = _task_ancestors(task.task_id, task_dependencies)
        prior_preflights = [
            tool_name
            for prior in workflow.tasks
            if prior.task_id in ancestors
            for tool_name in task_tools[prior.task_id]
            if tool_name in preflight_tools
        ]
        if prior_preflights:
            required = tuple(dict.fromkeys(prior_preflights))
            for tool_name in task.required_tools:
                if tool_name not in preflight_tools:
                    preflight_requirements[tool_name] = required

    required_artifacts = tuple(
        contract.name for contract in workflow.output_artifacts if contract.required
    )
    required_events: list[str] = []
    if len({task.role for task in workflow.tasks}) > 1:
        required_events.append("handoff_recorded")
    if required_artifacts:
        required_events.append("artifact_registered")
    return GoldenTrajectory(
        required_event_types=tuple(required_events),
        required_tool_order=tuple(tool_order),
        required_artifact_types=required_artifacts,
        preflight_requirements=preflight_requirements,
        role_tool_allowlists={role: frozenset(tools) for role, tools in sorted(role_tools.items())},
        allowed_terminal_statuses=tuple(allowed_terminal_statuses),
        max_duplicate_calls=max_duplicate_calls,
    )


def evaluate_trajectory(
    events: Iterable[Mapping[str, Any]],
    golden: GoldenTrajectory,
) -> TrajectoryReport:
    """Evaluate event semantics without comparing model-generated wording."""

    normalized = [dict(event) for event in events]
    event_types = [_event_type(event) for event in normalized]
    calls = [_tool_call(event) for event in normalized]
    calls = [call for call in calls if call is not None]
    artifacts = [_artifact(event) for event in normalized]
    artifacts = [artifact for artifact in artifacts if artifact is not None]
    terminal_status = _terminal_status(normalized)
    violations: list[str] = []

    for required in golden.required_event_types:
        if required not in event_types:
            violations.append(f"missing event type: {required}")

    successful_calls = [call for call in calls if _successful_call(call)]
    successful_tool_names = [str(call.get("tool_name") or "") for call in successful_calls]
    cursor = 0
    for expected in golden.required_tool_order:
        try:
            cursor = successful_tool_names.index(expected, cursor) + 1
        except ValueError:
            violations.append(f"missing or out-of-order tool: {expected}")
            break

    artifact_types = tuple(
        sorted({str(item.get("artifact_type")) for item in artifacts if item.get("artifact_type")})
    )
    for required in golden.required_artifact_types:
        if required not in artifact_types:
            violations.append(f"missing artifact type: {required}")

    allowed_domain_tools = frozenset().union(*golden.role_tool_allowlists.values())
    successful_prior: set[str] = set()
    for call in calls:
        tool_name = str(call.get("tool_name") or "")
        role = str(call.get("role") or "")
        allowed = golden.role_tool_allowlists.get(role)
        if tool_name in allowed_domain_tools and not role:
            violations.append(f"tool {tool_name!r} is missing its executing role")
        elif tool_name in allowed_domain_tools and allowed is None:
            violations.append(f"unknown role {role!r} called workflow tool {tool_name!r}")
        elif allowed is not None and tool_name not in allowed:
            violations.append(f"role {role!r} called disallowed tool {tool_name!r}")
        if _successful_call(call):
            required_preflight = golden.preflight_requirements.get(tool_name, ())
            missing = [name for name in required_preflight if name not in successful_prior]
            if missing:
                violations.append(f"tool {tool_name!r} ran before preflight: {', '.join(missing)}")
            successful_prior.add(tool_name)

    duplicate_count = _duplicate_calls(calls)
    if duplicate_count > golden.max_duplicate_calls:
        violations.append(
            f"duplicate tool calls {duplicate_count} exceeded {golden.max_duplicate_calls}"
        )

    if terminal_status not in golden.allowed_terminal_statuses:
        violations.append(
            f"terminal status {terminal_status!r} not in "
            f"{', '.join(golden.allowed_terminal_statuses)}"
        )

    retry_count = sum(
        max(
            0,
            _integer_metric(call.get("attempts", call.get("attempt", 1)), default=1) - 1,
        )
        for call in calls
    )
    duration = sum(max(0.0, _float_metric(call.get("duration_ms"), default=0.0)) for call in calls)
    return TrajectoryReport(
        passed=not violations,
        violations=tuple(violations),
        terminal_status=terminal_status,
        event_count=len(normalized),
        tool_call_count=len(calls),
        handoff_count=sum(kind == "handoff_recorded" for kind in event_types),
        retry_count=retry_count,
        duplicate_call_count=duplicate_count,
        artifact_types=artifact_types,
        total_tool_duration_ms=duration,
    )


def replay_run(
    run_id: str,
    golden: GoldenTrajectory,
    *,
    session_id: str | None = None,
) -> TrajectoryReport:
    """Load the authoritative event stream for ``run_id`` and evaluate it."""

    from cs_copilot.workflows.runtime import RunStore

    return evaluate_trajectory(
        RunStore(session_id=session_id).events(run_id),
        golden,
    )


def log_trajectory_report(report: TrajectoryReport, tracker: Any = None) -> None:
    """Send trajectory metrics and violations to the existing MLflow tracker."""

    if tracker is None:
        from .core import get_tracker

        tracker = get_tracker()
    tracker.log_metrics(report.metrics())
    tracker.log_dict(report.as_dict(), "trajectory_report.json")


def _event_type(event: Mapping[str, Any]) -> str:
    return str(event.get("event_type") or event.get("type") or "")


def _payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, Mapping) else event


def _tool_call(event: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = _event_type(event)
    if kind not in {"tool_call", "tool_call_recorded", "mcp_tool_call"}:
        return None
    payload = _payload(event)
    nested = payload.get("tool_call")
    return dict(nested if isinstance(nested, Mapping) else payload)


def _artifact(event: Mapping[str, Any]) -> dict[str, Any] | None:
    if _event_type(event) != "artifact_registered":
        return None
    payload = _payload(event)
    nested = payload.get("artifact")
    return dict(nested if isinstance(nested, Mapping) else payload)


def _terminal_status(events: Sequence[Mapping[str, Any]]) -> str | None:
    for event in reversed(events):
        if _event_type(event) not in {"run_status", "run_status_changed", "run_completed"}:
            continue
        payload = _payload(event)
        return str(payload.get("status") or "completed")
    return None


def _duplicate_calls(calls: Sequence[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for call in calls:
        if not _successful_call(call) or _cache_hit(call):
            continue
        key = str(call.get("idempotency_key") or "")
        if not key:
            signature = {
                "tool_name": call.get("tool_name"),
                "public_args": call.get("public_args", call.get("arguments", {})),
            }
            raw = json.dumps(signature, sort_keys=True, default=str).encode("utf-8")
            key = hashlib.sha256(raw).hexdigest()
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def _successful_call(call: Mapping[str, Any]) -> bool:
    return str(call.get("status") or "success").lower() in {
        "success",
        "completed",
        "ok",
    }


def _cache_hit(call: Mapping[str, Any]) -> bool:
    if call.get("cached") is True:
        return True
    metrics = call.get("metrics")
    return isinstance(metrics, Mapping) and metrics.get("cached") is True


def _integer_metric(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _float_metric(value: Any, *, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return (
        resolved
        if resolved == resolved and resolved not in {float("inf"), float("-inf")}
        else default
    )


def _task_ancestors(
    task_id: str,
    dependencies: Mapping[str, Sequence[str]],
) -> set[str]:
    ancestors: set[str] = set()
    pending = list(dependencies.get(task_id, ()))
    while pending:
        dependency = pending.pop()
        if dependency in ancestors:
            continue
        ancestors.add(dependency)
        pending.extend(dependencies.get(dependency, ()))
    return ancestors
