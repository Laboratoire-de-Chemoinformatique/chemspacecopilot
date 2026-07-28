"""Agno coordinator guard for structured, bounded specialist delegation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Mapping, MutableMapping, Sequence

from agno.exceptions import RetryAgentRun
from agno.team import Team
from agno.utils.team import get_member_id

from .contracts import (
    ROLE_POLICIES,
    ExecutionBudget,
    HandoffEnvelope,
    record_handoff,
)

DELEGATE_TOOL_NAME = "delegate_task_to_member"
DELEGATE_ALL_TOOL_NAME = "delegate_task_to_members"
COORDINATOR_ROLE = "coordinator"
_RESERVED_DELEGATE_TOOL_NAMES = frozenset({DELEGATE_TOOL_NAME, DELEGATE_ALL_TOOL_NAME})

_REQUIRED_HANDOFF_FIELDS = frozenset(
    {
        "run_id",
        "workflow_slug",
        "task_id",
        "sender_role",
        "receiver_role",
        "objective",
        "constraints",
        "required_capabilities",
        "input_artifact_ids",
        "expected_output_artifacts",
        "expected_output_schema",
        "acceptance_criteria",
        "context_summary",
        "budget",
        "trace_id",
        "span_id",
    }
)
_REQUIRED_BUDGET_FIELDS = frozenset({"max_tokens", "max_tool_calls", "timeout_seconds"})
_OPTIONAL_HANDOFF_FIELDS = frozenset(
    {
        "created_at",
        "handoff_id",
        "parent_span_id",
        "schema_version",
    }
)
_ALLOWED_HANDOFF_FIELDS = _REQUIRED_HANDOFF_FIELDS | _OPTIONAL_HANDOFF_FIELDS
_FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "agent_scratch",
        "chain_of_thought",
        "chat_history",
        "conversation",
        "conversation_history",
        "full_history",
        "history",
        "messages",
        "private_reasoning",
        "reasoning",
        "scratchpad",
    }
)
_ALLOWED_DELEGATE_ARGUMENTS = frozenset({"member_id", "task_description", "expected_output"})


@dataclass(frozen=True)
class DelegationLimits:
    """Hard limits applied to one coordinator run."""

    max_delegations_per_run: int = 12
    max_delegations_per_member: int = 5
    max_delegations_per_task: int = 2
    max_identical_handoffs: int = 1
    max_handoff_bytes: int = 32_000
    max_tokens: int = 12_000
    max_tool_calls: int = 24
    max_timeout_seconds: float = 1_800
    max_tracked_runs: int = 64

    def __post_init__(self) -> None:
        integer_limits = (
            "max_delegations_per_run",
            "max_delegations_per_member",
            "max_delegations_per_task",
            "max_identical_handoffs",
            "max_handoff_bytes",
            "max_tokens",
            "max_tool_calls",
            "max_tracked_runs",
        )
        for name in integer_limits:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"delegation limit {name} must be an integer")
        if isinstance(self.max_timeout_seconds, bool) or not isinstance(
            self.max_timeout_seconds, (int, float)
        ):
            raise TypeError("delegation limit max_timeout_seconds must be numeric")
        if not math.isfinite(float(self.max_timeout_seconds)):
            raise ValueError("delegation limit max_timeout_seconds must be finite")
        for name, value in self.as_dict().items():
            if value <= 0:
                raise ValueError(f"delegation limit {name} must be positive")

    def as_dict(self) -> dict[str, int | float]:
        return {
            "max_delegations_per_run": self.max_delegations_per_run,
            "max_delegations_per_member": self.max_delegations_per_member,
            "max_delegations_per_task": self.max_delegations_per_task,
            "max_identical_handoffs": self.max_identical_handoffs,
            "max_handoff_bytes": self.max_handoff_bytes,
            "max_tokens": self.max_tokens,
            "max_tool_calls": self.max_tool_calls,
            "max_timeout_seconds": self.max_timeout_seconds,
            "max_tracked_runs": self.max_tracked_runs,
        }


DEFAULT_DELEGATION_LIMITS = DelegationLimits()


@dataclass
class _RunDelegationState:
    total: int = 0
    per_member: Counter[str] = field(default_factory=Counter)
    per_task: Counter[tuple[str, str]] = field(default_factory=Counter)
    fingerprints: Counter[str] = field(default_factory=Counter)


class StructuredDelegationGuard:
    """Validate and canonicalize Agno's generated delegation tool calls.

    The guard is installed as the generated delegation ``Function.pre_hook``.
    Agno invokes synchronous pre-hooks in both its sync and async execution
    paths, so validation is identical without wrapping unrelated team tools.
    """

    def __init__(self, limits: DelegationLimits = DEFAULT_DELEGATION_LIMITS) -> None:
        self.limits = limits
        self._states: OrderedDict[str, _RunDelegationState] = OrderedDict()
        self._lock = Lock()

    def pre_hook(
        self,
        *,
        team: Any = None,
        session_state: Mapping[str, Any] | None = None,
        fc: Any = None,
    ) -> None:
        """Agno ``Function.pre_hook`` entry point."""
        self._pre_hook(team=team, session_state=session_state, fc=fc)

    def hook_for_run(self, coordinator_run_id: str | None) -> Any:
        """Bind counters to Agno's immutable run ID instead of mutable session state."""

        def validate_structured_delegation(
            *,
            team: Any = None,
            session_state: Mapping[str, Any] | None = None,
            fc: Any = None,
        ) -> None:
            self._pre_hook(
                team=team,
                session_state=session_state,
                fc=fc,
                coordinator_run_id=coordinator_run_id,
            )

        return validate_structured_delegation

    def _pre_hook(
        self,
        *,
        team: Any,
        session_state: Mapping[str, Any] | None,
        fc: Any,
        coordinator_run_id: str | None = None,
    ) -> None:
        function = getattr(fc, "function", None)
        function_name = getattr(function, "name", "")
        if function_name not in {DELEGATE_TOOL_NAME, DELEGATE_ALL_TOOL_NAME}:
            return
        arguments = getattr(fc, "arguments", None)
        dependencies = getattr(function, "_dependencies", None)
        self._guard_call(
            function_name=function_name,
            arguments=arguments,
            team=team,
            session_state=session_state,
            dependencies=dependencies,
            coordinator_run_id=coordinator_run_id,
        )

    def _guard_call(
        self,
        *,
        function_name: str,
        arguments: MutableMapping[str, Any],
        team: Any,
        session_state: Mapping[str, Any] | None,
        dependencies: Mapping[str, Any] | None,
        coordinator_run_id: str | None,
    ) -> None:
        if function_name == DELEGATE_ALL_TOOL_NAME:
            raise RetryAgentRun(
                "Broadcast delegation is disabled: create one role-scoped structured "
                "handoff per specialist."
            )

        try:
            envelope = self._validate_call(arguments, team)
            fingerprint = _handoff_fingerprint(envelope)
            run_key = coordinator_run_id or _coordinator_run_key(session_state, envelope)
            self._reserve(run_key, arguments["member_id"], envelope.task_id, fingerprint)
            try:
                runtime = _resolve_run_context(team, dependencies)
                record_handoff(runtime, envelope)
            except Exception:
                self._release(
                    run_key,
                    arguments["member_id"],
                    envelope.task_id,
                    fingerprint,
                )
                raise
            _canonicalize_delegate_arguments(arguments, envelope)
        except RetryAgentRun:
            raise
        except Exception as exc:
            raise RetryAgentRun(
                "Delegation rejected by the structured handoff guard: "
                f"{exc}. Pass task_description as one JSON handoff object."
            ) from exc

    def _validate_call(
        self,
        arguments: MutableMapping[str, Any],
        team: Any,
    ) -> HandoffEnvelope:
        if not isinstance(arguments, MutableMapping):
            raise TypeError("delegation arguments must be a mapping")
        unexpected = sorted(set(arguments) - _ALLOWED_DELEGATE_ARGUMENTS)
        if unexpected:
            raise ValueError(f"unexpected delegation arguments: {', '.join(unexpected)}")

        member_id = str(arguments.get("member_id") or "").strip()
        if not member_id:
            raise ValueError("member_id is required")
        raw_description = arguments.get("task_description")
        if not isinstance(raw_description, str) or not raw_description.strip():
            raise TypeError("task_description must be a non-empty JSON string")
        if len(raw_description.encode("utf-8")) > self.limits.max_handoff_bytes:
            raise ValueError(f"handoff exceeds {self.limits.max_handoff_bytes} encoded bytes")

        try:
            payload = json.loads(
                raw_description,
                object_pairs_hook=_mapping_without_duplicate_keys,
                parse_constant=_reject_non_finite_json,
            )
        except json.JSONDecodeError as exc:
            raise ValueError("task_description is not valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise TypeError("task_description JSON must contain an object")
        _reject_private_context(payload)

        missing = sorted(_REQUIRED_HANDOFF_FIELDS - set(payload))
        if missing:
            raise ValueError(f"handoff is missing required fields: {', '.join(missing)}")
        unexpected = sorted(set(payload) - _ALLOWED_HANDOFF_FIELDS)
        if unexpected:
            raise ValueError(f"handoff has unknown fields: {', '.join(unexpected)}")
        _validate_structured_fields(payload)
        self._validate_budget(payload["budget"])

        envelope = HandoffEnvelope.from_mapping(payload)
        selected_member = _selected_member(team, member_id)
        selected_role = _member_role(selected_member)
        if selected_role is None:
            raise ValueError(f"member '{member_id}' has no declared agentic role")
        if envelope.receiver_role != selected_role:
            raise ValueError(
                f"receiver_role '{envelope.receiver_role}' does not match member "
                f"'{member_id}' role '{selected_role}'"
            )
        allowed_senders = set(ROLE_POLICIES) | {COORDINATOR_ROLE}
        if envelope.sender_role not in allowed_senders:
            raise ValueError(f"unknown sender_role '{envelope.sender_role}'")
        if envelope.receiver_role not in ROLE_POLICIES:
            raise ValueError(f"unknown receiver_role '{envelope.receiver_role}'")
        _suppress_member_context(selected_member)
        return envelope

    def _validate_budget(self, value: Any) -> None:
        if not isinstance(value, Mapping):
            raise TypeError("handoff budget must be a mapping")
        missing = sorted(_REQUIRED_BUDGET_FIELDS - set(value))
        if missing:
            raise ValueError(f"handoff budget is missing: {', '.join(missing)}")
        unexpected = sorted(set(value) - _REQUIRED_BUDGET_FIELDS)
        if unexpected:
            raise ValueError(f"handoff budget has unknown fields: {', '.join(unexpected)}")
        budget = ExecutionBudget(
            max_tokens=_strict_int(value["max_tokens"], "max_tokens"),
            max_tool_calls=_strict_int(value["max_tool_calls"], "max_tool_calls"),
            timeout_seconds=_strict_number(value["timeout_seconds"], "timeout_seconds"),
        )
        if budget.max_tokens is not None and budget.max_tokens > self.limits.max_tokens:
            raise ValueError(f"max_tokens exceeds guard limit {self.limits.max_tokens}")
        if budget.max_tool_calls is not None and budget.max_tool_calls > self.limits.max_tool_calls:
            raise ValueError(f"max_tool_calls exceeds guard limit {self.limits.max_tool_calls}")
        if (
            budget.timeout_seconds is not None
            and budget.timeout_seconds > self.limits.max_timeout_seconds
        ):
            raise ValueError(
                "timeout_seconds exceeds guard limit " f"{self.limits.max_timeout_seconds:g}"
            )

    def _reserve(
        self,
        run_key: str,
        member_id: str,
        task_id: str,
        fingerprint: str,
    ) -> None:
        with self._lock:
            state = self._state_for(run_key)
            if state.total >= self.limits.max_delegations_per_run:
                raise ValueError(
                    "coordinator delegation limit reached for this run "
                    f"({self.limits.max_delegations_per_run})"
                )
            if state.per_member[member_id] >= self.limits.max_delegations_per_member:
                raise ValueError(
                    f"member '{member_id}' delegation limit reached "
                    f"({self.limits.max_delegations_per_member})"
                )
            task_key = (member_id, task_id)
            if state.per_task[task_key] >= self.limits.max_delegations_per_task:
                raise ValueError(
                    f"task '{task_id}' repeat limit reached "
                    f"({self.limits.max_delegations_per_task})"
                )
            if state.fingerprints[fingerprint] >= self.limits.max_identical_handoffs:
                raise ValueError(
                    "identical handoff already delegated; revise the task or context "
                    "instead of repeating the same call"
                )
            state.total += 1
            state.per_member[member_id] += 1
            state.per_task[task_key] += 1
            state.fingerprints[fingerprint] += 1

    def _release(
        self,
        run_key: str,
        member_id: str,
        task_id: str,
        fingerprint: str,
    ) -> None:
        with self._lock:
            state = self._states.get(run_key)
            if state is None:
                return
            state.total = max(0, state.total - 1)
            _decrement(state.per_member, member_id)
            _decrement(state.per_task, (member_id, task_id))
            _decrement(state.fingerprints, fingerprint)

    def _state_for(self, run_key: str) -> _RunDelegationState:
        state = self._states.get(run_key)
        if state is not None:
            self._states.move_to_end(run_key)
            return state
        state = _RunDelegationState()
        self._states[run_key] = state
        while len(self._states) > self.limits.max_tracked_runs:
            self._states.popitem(last=False)
        return state


class StructuredHandoffTeam(Team):
    """Team variant that keeps implicit coordinator context out of member prompts.

    Agno 2.1.9 has one ``add_history_to_context`` flag for both the coordinator
    and delegated members. The only narrow seam that can preserve coordinator
    history while disabling member history is the generated delegate-function
    builder. This override changes only those member-call flags.
    """

    def __init__(
        self,
        *args: Any,
        run_context: Any = None,
        delegation_guard: StructuredDelegationGuard | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        collisions = sorted(
            _RESERVED_DELEGATE_TOOL_NAMES.intersection(_configured_tool_names(self.tools or ()))
        )
        if collisions:
            raise ValueError(
                "configured team tools cannot shadow Agno delegation tools: "
                + ", ".join(collisions)
            )
        for member in self.members:
            _suppress_member_context(member)
        self.run_context = run_context
        self.delegation_guard = delegation_guard or StructuredDelegationGuard()

    def _get_delegate_task_function(self, *args: Any, **kwargs: Any) -> Any:
        run_response = kwargs.get("run_response")
        if run_response is None and args:
            run_response = args[0]
        coordinator_run_id = getattr(run_response, "run_id", None)
        kwargs["add_history_to_context"] = False
        kwargs["add_session_state_to_context"] = False
        kwargs["add_dependencies_to_context"] = False
        delegate_function = super()._get_delegate_task_function(*args, **kwargs)
        delegate_function.pre_hook = self.delegation_guard.hook_for_run(
            str(coordinator_run_id) if coordinator_run_id else None
        )
        delegate_function.description = (
            "Delegate one bounded task to one specialist. task_description must be a "
            "JSON-encoded v2 HandoffEnvelope; unstructured, private-history, oversized, "
            "or repeated handoffs are rejected."
        )
        properties = (delegate_function.parameters or {}).get("properties", {})
        if "task_description" in properties:
            properties["task_description"][
                "description"
            ] = "A JSON string containing the complete structured handoff contract."
        if "expected_output" in properties:
            properties["expected_output"][
                "description"
            ] = "Optional placeholder; the guard derives this value from the handoff."
        return delegate_function


def _validate_structured_fields(payload: Mapping[str, Any]) -> None:
    for field_name in (
        "run_id",
        "workflow_slug",
        "task_id",
        "sender_role",
        "receiver_role",
        "objective",
        "context_summary",
        "trace_id",
        "span_id",
    ):
        value = payload[field_name]
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"handoff {field_name} must be a non-empty string")
    parent_span_id = payload.get("parent_span_id")
    if parent_span_id is not None and (
        not isinstance(parent_span_id, str) or not parent_span_id.strip()
    ):
        raise TypeError("handoff parent_span_id must be null or a non-empty string")
    for field_name in (
        "constraints",
        "required_capabilities",
        "input_artifact_ids",
        "expected_output_artifacts",
        "acceptance_criteria",
    ):
        value = payload[field_name]
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise TypeError(f"handoff {field_name} must be a list of strings")
        if any(not isinstance(item, str) or not item.strip() for item in value):
            raise TypeError(f"handoff {field_name} must contain non-empty strings")
        if len(value) > 64:
            raise ValueError(f"handoff {field_name} cannot contain more than 64 entries")
    for required_non_empty in (
        "required_capabilities",
        "expected_output_artifacts",
        "acceptance_criteria",
    ):
        if not payload[required_non_empty]:
            raise ValueError(f"handoff {required_non_empty} cannot be empty")
    if not isinstance(payload["expected_output_schema"], Mapping):
        raise TypeError("handoff expected_output_schema must be a mapping")
    if not payload["expected_output_schema"]:
        raise ValueError("handoff expected_output_schema cannot be empty")


def _reject_private_context(
    value: Any,
    *,
    path: str = "handoff",
    depth: int = 0,
) -> None:
    if depth > 32:
        raise ValueError("handoff JSON nesting exceeds 32 levels")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_CONTEXT_KEYS:
                raise ValueError(f"{path} contains forbidden private/history field '{key}'")
            _reject_private_context(nested, path=f"{path}.{key}", depth=depth + 1)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_private_context(
                nested,
                path=f"{path}[{index}]",
                depth=depth + 1,
            )


def _strict_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"handoff budget {field_name} must be an integer")
    return value


def _strict_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"handoff budget {field_name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"handoff budget {field_name} must be finite")
    return normalized


def _mapping_without_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"handoff JSON contains duplicate key '{key}'")
        value[key] = item
    return value


def _reject_non_finite_json(value: str) -> Any:
    raise ValueError(f"handoff JSON contains non-finite number '{value}'")


def _selected_member(team: Any, member_id: str) -> Any:
    for member in getattr(team, "members", ()) or ():
        if get_member_id(member) == member_id:
            return member
    return None


def _member_role(member: Any) -> str | None:
    explicit_role = getattr(member, "agentic_role", None)
    if explicit_role:
        return str(explicit_role)
    policy = getattr(member, "role_policy", None)
    if policy is not None and getattr(policy, "role", None):
        return str(policy.role)
    return None


def _suppress_member_context(member: Any) -> None:
    if member is None:
        return
    member.add_history_to_context = False
    member.add_session_state_to_context = False
    member.add_dependencies_to_context = False


def _configured_tool_names(tools: Sequence[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        functions = getattr(tool, "functions", None)
        if isinstance(functions, Mapping):
            names.update(str(name) for name in functions)
            continue
        if isinstance(tool, Mapping):
            function = tool.get("function")
            if isinstance(function, Mapping) and function.get("name"):
                names.add(str(function["name"]))
            continue
        name = getattr(tool, "name", None)
        if name and getattr(tool, "entrypoint", None) is not None:
            names.add(str(name))
            continue
        callable_name = getattr(tool, "__name__", None)
        if callable_name:
            names.add(str(callable_name))
    return names


def _resolve_run_context(
    team: Any,
    dependencies: Mapping[str, Any] | None,
) -> Any:
    attached = getattr(team, "run_context", None)
    if attached is not None:
        return attached
    if dependencies:
        return dependencies.get("run_context")
    return None


def _coordinator_run_key(
    session_state: Mapping[str, Any] | None,
    envelope: HandoffEnvelope,
) -> str:
    if session_state:
        current_run_id = session_state.get("current_run_id")
        if current_run_id:
            return str(current_run_id)
    return f"unscoped:{envelope.run_id}"


def _handoff_fingerprint(envelope: HandoffEnvelope) -> str:
    payload = envelope.to_dict()
    for volatile in (
        "handoff_id",
        "created_at",
        "trace_id",
        "span_id",
        "parent_span_id",
    ):
        payload.pop(volatile, None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonicalize_delegate_arguments(
    arguments: MutableMapping[str, Any],
    envelope: HandoffEnvelope,
) -> None:
    member_id = str(arguments["member_id"])
    task_description = json.dumps(
        envelope.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_output = (
        "Return the declared artifacts "
        f"({', '.join(envelope.expected_output_artifacts)}) and satisfy every "
        "acceptance criterion and expected_output_schema in the handoff."
    )
    arguments.clear()
    arguments.update(
        member_id=member_id,
        task_description=task_description,
        expected_output=expected_output,
    )


def _decrement(counter: Counter[Any], key: Any) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]
