"""Route the in-process Agno team's toolkit calls through the execution kernel.

:func:`attach_execution` wraps every toolkit function of a team (coordinator
toolkits and every member's toolkits and bare-function tools) with a
signature-preserving entrypoint. Each call is then recorded in the chat's
durable workflow run like an MCP tool call: same event protocol, same tool
identity (via :mod:`cs_copilot.agents.execution_contracts`), same artifact
registration.

The mode comes from ``CS_COPILOT_AGNO_EXECUTION`` (default ``enforce``):

* ``off``: no wrapping; behaviour and storage are exactly as before.
* ``observe``: calls run unchanged; events, artifact registration, and an
  audit of what enforcement would rewrite or deny are recorded
  (:func:`cs_copilot.execution.runner.observe_sync`).
* ``enforce`` (default): the full kernel (:func:`cs_copilot.execution.runner.execute_sync`):
  output paths are rewritten into the chat's run, writes are confined and
  create-only, declared file inputs must be registered artifacts, every file
  a call writes is registered, and a call's files are rolled back if it fails.
  Denials reach the model as tool errors.

Agno hooks are deliberately not used: ``Team.tool_hooks`` would also wrap the
async-generator delegation tool, hooks cannot rewrite arguments, and they do
not know which toolkit a call belongs to.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import inspect
import logging
import os
import socket
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

from cs_copilot import capabilities
from cs_copilot.execution.context import AGNO_RUNTIME, BasicExecutionContext
from cs_copilot.execution.errors import ToolExecutionError
from cs_copilot.execution.runner import execute_sync, observe_sync
from cs_copilot.execution.scope import _InvocationScope
from cs_copilot.execution.spec import ToolSpec
from cs_copilot.storage import OUTPUT_CONTEXT_KEY, S3

from .contracts import ROLE_POLICIES
from .execution_contracts import resolve_contract

logger = logging.getLogger(__name__)

EXECUTION_MODE_ENV = "CS_COPILOT_AGNO_EXECUTION"
BINDING_ATTRIBUTE = "cs_execution_binding"
# Parameters Agno injects by name (agno.tools.function.FunctionCall).
_AGNO_INJECTED = frozenset(
    {"agent", "team", "session_state", "dependencies", "fc", "images", "videos", "audios", "files"}
)
_WRAPPER_INJECTED = ("agent", "team", "session_state")
_RESERVED_STATE_KEYS = frozenset({OUTPUT_CONTEXT_KEY, "agentic_contracts", "mcp_profile"})
_RESERVED_STATE_PREFIXES = ("current_", "active_")
_IN_BOUND_CALL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "cs_copilot_agno_bound_call", default=False
)
# Identifies this process in "started" events so a later process can recognise
# spans orphaned by a crash or restart.
PROCESS_OWNER = {"host": socket.gethostname(), "pid": os.getpid(), "boot": uuid.uuid4().hex}


class ExecutionMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    ENFORCE = "enforce"


DEFAULT_EXECUTION_MODE = ExecutionMode.ENFORCE


def execution_mode_from_env(value: str | ExecutionMode | None = None) -> ExecutionMode:
    """Resolve an execution mode from ``value`` or ``CS_COPILOT_AGNO_EXECUTION``."""

    if isinstance(value, ExecutionMode):
        return value
    raw = os.getenv(EXECUTION_MODE_ENV) if value is None else value
    if raw is None or not str(raw).strip():
        return DEFAULT_EXECUTION_MODE
    try:
        return ExecutionMode(str(raw).strip().lower())
    except ValueError as exc:
        raise ValueError(
            f"{EXECUTION_MODE_ENV} must be one of off, observe, enforce; got {raw!r}"
        ) from exc


@dataclass
class ExecutionBinding:
    """Per-team routing state: the chat's run, the mode, and task attribution."""

    mode: ExecutionMode
    team: Any
    run_context: Any = None
    active_tasks: dict[str, str] = field(default_factory=dict)
    turn_tasks: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _in_flight: int = 0
    _idle: threading.Condition = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._idle = threading.Condition(self._lock)

    @property
    def enabled(self) -> bool:
        return self.mode is not ExecutionMode.OFF and self.run_context is not None

    def note_task_started(self, role: str, task_id: str) -> None:
        with self._lock:
            self.active_tasks[role] = task_id
            if task_id not in self.turn_tasks:
                self.turn_tasks.append(task_id)

    def take_turn_tasks(self) -> list[str]:
        with self._lock:
            tasks, self.turn_tasks = self.turn_tasks, []
            self.active_tasks.clear()
            return tasks

    def note_problem(self, message: str) -> None:
        with self._lock:
            self.problems = [*self.problems[-49:], message]

    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until no bound call is running (used before cancelling a run)."""

        with self._idle:
            return self._idle.wait_for(lambda: self._in_flight == 0, timeout=timeout)

    def invoke(
        self,
        function: Callable[..., Any],
        owner: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> Any:
        """Run one Agno tool call, recording it through the kernel when enabled.

        ``kwargs`` are what ``function`` receives. ``context`` holds Agno's
        ``agent`` / ``team`` / ``session_state`` for functions that do not
        declare them; they inform the recording but are never passed on.
        """

        if not self.enabled or _IN_BOUND_CALL.get():
            return function(*args, **kwargs)
        token = _IN_BOUND_CALL.set(True)
        with self._lock:
            self._in_flight += 1
        try:
            if self.mode is ExecutionMode.ENFORCE:
                return self._enforce(function, owner, args, kwargs, dict(context or {}))
            return self._observe(function, owner, args, kwargs, dict(context or {}))
        finally:
            _IN_BOUND_CALL.reset(token)
            with self._idle:
                self._in_flight -= 1
                self._idle.notify_all()

    def _observe(
        self,
        function: Callable[..., Any],
        owner: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        run_context = self.run_context
        try:
            contract, _source = resolve_contract(function, owner=owner)
            known = {**context, **{key: kwargs[key] for key in _WRAPPER_INJECTED if key in kwargs}}
            state = known.get("session_state")
            if not isinstance(state, dict):
                state = self.team_state()
            self._ensure_bound(state)
            ctx = BasicExecutionContext(
                execution_runtime=AGNO_RUNTIME,
                session_state=state,
                run_context=run_context,
            )
            scope = self._scope(known.get("agent"))
            public = {key: value for key, value in kwargs.items() if key not in _AGNO_INJECTED}
            injected = {key: value for key, value in kwargs.items() if key in _AGNO_INJECTED}
        except Exception as exc:  # noqa: BLE001 - never block the call on bookkeeping
            logger.warning("Could not prepare kernel bookkeeping for %r: %s", function, exc)
            self.note_problem(f"prepare {getattr(function, '__name__', function)}: {exc}")
            return function(*args, **kwargs)

        def call(arguments: dict[str, Any]) -> Any:
            return function(*args, **arguments, **injected)

        with S3.scoped_session_prefix(f"sessions/{run_context.run.session_id}"):
            return observe_sync(
                contract,
                ctx,
                public,
                invoke=call,
                scope=scope,
                extra={"owner": PROCESS_OWNER, "execution_mode": self.mode.value},
            )

    def _enforce(
        self,
        function: Callable[..., Any],
        owner: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        context: dict[str, Any],
    ) -> Any:
        run_context = self.run_context
        contract, _source = resolve_contract(function, owner=owner)
        contract = _in_process_contract(contract, function)
        known = {**context, **{key: kwargs[key] for key in _WRAPPER_INJECTED if key in kwargs}}
        state = known.get("session_state")
        if not isinstance(state, dict):
            state = self.team_state()
        self._ensure_bound(state)
        ctx = BasicExecutionContext(
            execution_runtime=AGNO_RUNTIME,
            session_state=state,
            run_context=run_context,
        )
        public = {key: value for key, value in kwargs.items() if key not in _AGNO_INJECTED}
        injected = {key: value for key, value in kwargs.items() if key in _AGNO_INJECTED}

        def inject(call_kwargs: dict[str, Any], session_view: dict[str, Any] | None) -> None:
            call_kwargs.update(injected)
            if session_view is not None and "session_state" in injected:
                call_kwargs["session_state"] = session_view

        def call(call_kwargs: dict[str, Any]) -> Any:
            return function(*args, **call_kwargs)

        with S3.scoped_session_prefix(f"sessions/{run_context.run.session_id}"):
            outcome = execute_sync(
                contract,
                ctx,
                public,
                invoke=call,
                inject=inject,
                publication_policy="all_published",
                verify_artifacts_on_lock=False,
                scope=self._scope(known.get("agent")),
                extra={"owner": PROCESS_OWNER, "execution_mode": self.mode.value},
            )
        if outcome.ok:
            return outcome.value
        if outcome.error is not None:
            raise outcome.error
        raise ToolExecutionError(str((outcome.envelope.get("error") or {}).get("message")))

    def team_state(self) -> dict[str, Any]:
        state = getattr(self.team, "session_state", None)
        if not isinstance(state, dict):
            state = {}
            self.team.session_state = state
        return state

    def _ensure_bound(self, state: dict[str, Any]) -> None:
        run = self.run_context.run
        output_context = state.get(OUTPUT_CONTEXT_KEY)
        if (
            isinstance(output_context, Mapping)
            and output_context.get("run_id") == run.run_id
            and output_context.get("session_id") == run.session_id
        ):
            return
        self.run_context.bind_session_state(state)
        team_state = self.team_state()
        if team_state is not state:
            self.run_context.bind_session_state(team_state)
        self.note_problem("re-bound a session_state whose output_context did not name the run")

    def _scope(self, agent: Any) -> _InvocationScope:
        run = self.run_context.run
        role = getattr(agent, "agentic_role", None) if agent is not None else None
        if role is None:
            role = capabilities.COORDINATOR_ROLE
        policy = ROLE_POLICIES.get(role)
        with self._lock:
            task_id = self.active_tasks.get(role)
        task = run.tasks.get(task_id) if task_id else None
        return _InvocationScope(
            run_id=str(run.run_id),
            task_id=task_id if task is not None else None,
            role=role,
            profile=policy.profile if policy is not None else None,
            task_attempt=int(task.attempts) if task is not None else None,
        )


def attach_execution(
    team: Any,
    *,
    run_context: Any,
    mode: ExecutionMode | str | None = None,
) -> ExecutionBinding | None:
    """Route ``team``'s tool calls through the kernel for ``run_context``.

    Idempotent: re-attaching updates the run and mode of the existing binding.
    Must run before the team's first run, because Agno processes tool
    entrypoints when it first builds the model's tool list.
    """

    selected = execution_mode_from_env(mode)
    run = getattr(run_context, "run", None)
    if selected is ExecutionMode.ENFORCE and run is not None and not _is_ad_hoc(run):
        raise ValueError(
            "enforce mode supports ad-hoc chat runs only; catalog workflow runs "
            "need MCP task scope"
        )
    existing: ExecutionBinding | None = getattr(team, BINDING_ATTRIBUTE, None)
    if selected is ExecutionMode.OFF:
        if existing is not None:
            existing.mode = selected
        return existing
    if existing is not None:
        existing.mode = selected
        existing.run_context = run_context
        return existing
    binding = ExecutionBinding(mode=selected, team=team, run_context=run_context)
    setattr(team, BINDING_ATTRIBUTE, binding)
    for entity in _team_entities(team):
        _wrap_tools(entity, binding)
        _guard_agentic_state(entity)
    return binding


def get_binding(team: Any) -> ExecutionBinding | None:
    return getattr(team, BINDING_ATTRIBUTE, None)


def _team_entities(team: Any) -> Iterable[Any]:
    yield team
    for member in getattr(team, "members", None) or ():
        if getattr(member, "members", None):
            yield from _team_entities(member)
        else:
            yield member


def _wrap_tools(entity: Any, binding: ExecutionBinding) -> None:
    tools = getattr(entity, "tools", None)
    if not tools:
        return
    wrapped_tools = []
    for tool in tools:
        functions = getattr(tool, "functions", None)
        if isinstance(functions, dict):
            for function in functions.values():
                entrypoint = getattr(function, "entrypoint", None)
                if callable(entrypoint):
                    function.entrypoint = _wrap_callable(entrypoint, binding, owner=tool)
            wrapped_tools.append(tool)
        elif inspect.isfunction(tool):
            wrapped_tools.append(_wrap_callable(tool, binding, owner=None))
        else:
            wrapped_tools.append(tool)
    entity.tools = wrapped_tools


def _wrap_callable(
    function: Callable[..., Any],
    binding: ExecutionBinding,
    *,
    owner: Any,
) -> Callable[..., Any]:
    if getattr(function, "__cs_execution_binding__", None) is not None:
        return function
    signature = inspect.signature(function)
    extra = [name for name in _WRAPPER_INJECTED if name not in signature.parameters]
    parameters = list(signature.parameters.values())
    insert_at = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD
        ),
        len(parameters),
    )
    parameters[insert_at:insert_at] = [
        inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=None) for name in extra
    ]
    real_state = binding.mode is ExecutionMode.ENFORCE
    if real_state:
        # Agno validates arguments with pydantic, which hands an annotated
        # ``session_state`` parameter a copy; enforce mode passes the live
        # dict so top-level state writes persist, as on the MCP path.
        parameters = [
            (
                parameter.replace(annotation=inspect.Parameter.empty)
                if parameter.name == "session_state"
                else parameter
            )
            for parameter in parameters
        ]

    @functools.wraps(function)
    def entrypoint(*args: Any, **kwargs: Any) -> Any:
        # Agno injects these because the wrapper declares them; the wrapped
        # function does not take them, so they only inform the recording.
        context = {name: kwargs.pop(name, None) for name in extra}
        return binding.invoke(function, owner, args, kwargs, context=context)

    entrypoint.__signature__ = signature.replace(parameters=parameters)  # type: ignore[attr-defined]
    if real_state:
        entrypoint.__annotations__ = {
            key: value
            for key, value in getattr(function, "__annotations__", {}).items()
            if key != "session_state"
        }
    entrypoint.__cs_execution_binding__ = binding  # type: ignore[attr-defined]
    return entrypoint


def _in_process_contract(contract: ToolSpec, function: Callable[..., Any]) -> ToolSpec:
    """Adapt an MCP contract to an in-process synchronous call.

    Agno tools always run in this process: worker dispatch, cancellable
    timeouts, and kernel retries (Agno's model loop retries instead) are
    dropped, and forced arguments apply only to parameters the function takes.
    """

    parameters = inspect.signature(function).parameters
    return dataclasses.replace(
        contract,
        run_in_worker_process=False,
        worker_timeout_s=None,
        timeout_s=None,
        max_retries=0,
        forces={key: value for key, value in contract.forces.items() if key in parameters},
    )


def _is_ad_hoc(run: Any) -> bool:
    from cs_copilot.execution.context import is_ad_hoc_run

    return is_ad_hoc_run(run)


def _guard_agentic_state(entity: Any) -> None:
    """Shadow Agno's ``update_session_state`` tool with one that protects run identity."""

    if not hasattr(type(entity), "update_session_state"):
        return

    def update_session_state(session_state, session_state_updates: dict) -> str:
        """
        Update the shared session state.  Provide any updates as a dictionary of key-value pairs.
        Example:
            "session_state_updates": {"shopping_list": ["milk", "eggs", "bread"]}

        Args:
            session_state_updates (dict): The updates to apply to the shared session state. Should be a dictionary of key-value pairs.
        """
        refused = sorted(key for key in session_state_updates if _is_reserved_state_key(key))
        for key, value in session_state_updates.items():
            if key not in refused:
                session_state[key] = value
        message = f"Updated session state: {session_state}"
        if refused:
            message += f" (runtime keys are read-only and were not changed: {', '.join(refused)})"
        return message

    entity.update_session_state = update_session_state


def _is_reserved_state_key(key: Any) -> bool:
    text = str(key)
    return text in _RESERVED_STATE_KEYS or text.startswith(_RESERVED_STATE_PREFIXES)
