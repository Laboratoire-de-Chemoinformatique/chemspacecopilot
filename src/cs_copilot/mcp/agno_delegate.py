"""Opt-in MCP delegation into the in-process Agno team (``agno_team_run``).

The tool is an ordinary kernel-governed MCP tool (see
``cs_copilot.mcp.tool_specs.agno``) whose spec declares
``delegates_execution=True``: the outer invocation only records events, and
every tool the team calls runs through the execution kernel itself, in
``enforce`` mode, inside the MCP session's ad-hoc run. Inner calls are linked
to the outer span through ``parent_span_id``.

The team uses the MCP session's configured model (``--llm-policy
agno-model``); calls are refused under the ``external`` and ``disabled``
policies and in catalog workflow runs. It sees a JSON snapshot of the MCP
session state, plus the session's in-memory objects (data frames, fitted
models) by reference, as an in-process MCP tool would. Its JSON-serializable
changes are merged back with the same optimistic concurrency as
worker-process tools; in-memory objects it creates or replaces stay with the
team for that call, and their files are registered artifacts of the run.

Imports of the Agno team are dynamic so the default MCP path never loads the
team, its factories, or the model stack.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import threading
from typing import Any, Mapping

from cs_copilot.execution.context import is_ad_hoc_run
from cs_copilot.execution.llm import llm_policy_of, resolve_model
from cs_copilot.execution.runner import _run_sync_to_completion
from cs_copilot.execution.tracing import current_tool_span_id

from .errors import MCPErrorCode, MCPToolError
from .jobs import _EXECUTION_SCOPE_KEYS, _merge_worker_session_state

logger = logging.getLogger(__name__)

TOOL_NAME = "agno_team_run"
# Team-private keys: Agno's own bookkeeping and the state each fresh team
# owns. They are never merged into the MCP session.
TEAM_PRIVATE_STATE_KEYS = frozenset(
    {
        "current_run_id",
        "current_session_id",
        "current_user_id",
        "resource_profile",
        "agentic_contracts",
    }
)
_DRAIN_TIMEOUT_S = 300.0


class AgnoTeamFacade:
    """Run one prompt through the cs_copilot Agno team for an MCP session."""

    def __init__(self) -> None:
        self._busy = threading.Lock()
        self._team: Any = None
        self._team_run_id: str | None = None
        self._team_defaults: dict[str, Any] = {}

    async def run(self, prompt: str, agent: Any | None = None) -> dict[str, Any]:
        """Delegate ``prompt`` to the in-process cs_copilot Agno team."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise MCPToolError("prompt cannot be empty", code=MCPErrorCode.INVALID_INPUT)
        run_context = _require_team_session(agent)
        if not self._busy.acquire(blocking=False):
            raise MCPToolError(
                f"{TOOL_NAME} is already running for this MCP session; wait for it to finish.",
                code=MCPErrorCode.RESOURCE_LIMIT,
            )
        try:
            return await self._run(prompt, agent, run_context)
        finally:
            self._busy.release()

    async def _run(self, prompt: str, ctx: Any, run_context: Any) -> dict[str, Any]:
        team = await asyncio.to_thread(self._team_for, ctx, run_context)
        session_runs = __import__(
            "cs_copilot.agents.session_runs",
            fromlist=["finalize_agno_turn", "restore_team_session_state"],
        )
        binding = __import__(
            "cs_copilot.agents.execution_binding", fromlist=["get_binding"]
        ).get_binding(team)

        base, by_reference = _split_session_state(ctx.session_state)
        seed = {**copy.deepcopy(self._team_defaults), **copy.deepcopy(base), **by_reference}
        for key in _EXECUTION_SCOPE_KEYS - {"output_context"}:
            seed.pop(key, None)
        output_context = seed.get("output_context")
        if isinstance(output_context, dict):
            output_context.pop("span_id", None)
            output_context["parent_span_id"] = current_tool_span_id()
        state = session_runs.restore_team_session_state(team, seed)
        artifacts_before = set(run_context.refresh().artifacts)
        if binding is not None:
            binding.take_problems()

        failed = True
        try:
            result = await team.arun(prompt, stream=False, session_id=run_context.run.session_id)
            failed = False
        except asyncio.CancelledError:
            agno_run_id = state.get("current_run_id")
            if isinstance(agno_run_id, str):
                team.cancel_run(agno_run_id)
            raise
        finally:
            # Tool threads cannot be interrupted: wait for them (even if this
            # call is cancelled again) before the session can be used again.
            await _run_sync_to_completion(_settle_turn, session_runs, team, binding, failed)

        shared, unshared = _shareable_changes(base, state, by_reference)
        _merge_worker_session_state(
            target=ctx.session_state,
            base=base,
            worker=shared,
            tool_name=TOOL_NAME,
            retryable=False,
        )
        run = run_context.refresh()
        warnings = binding.take_problems() if binding is not None else []
        if unshared:
            warnings.append(
                "Session keys holding in-memory objects were not shared with the MCP "
                f"session: {', '.join(unshared)}; use the run's artifacts instead."
            )
        return {
            "status": "ok",
            "runtime": "agno_team",
            "content": _content_text(getattr(result, "content", None)),
            "run_id": run.run_id,
            "artifact_ids": sorted(set(run.artifacts) - artifacts_before),
            "warnings": warnings,
        }

    def _team_for(self, ctx: Any, run_context: Any) -> Any:
        run_id = run_context.run.run_id
        if self._team is not None and self._team_run_id == run_id:
            return self._team
        teams = __import__("cs_copilot.agents.teams", fromlist=["get_cs_copilot_agent_team"])
        team = teams.get_cs_copilot_agent_team(
            resolve_model(ctx),
            show_members_responses=False,
            enable_memory=False,
            enable_mlflow_tracking=False,
            run_context=run_context,
            execution_mode="enforce",
        )
        state = getattr(team, "session_state", None)
        self._team_defaults = copy.deepcopy(state) if isinstance(state, dict) else {}
        self._team, self._team_run_id = team, run_id
        return team


def _require_team_session(ctx: Any) -> Any:
    policy = llm_policy_of(ctx)
    if policy != "agno-model" or resolve_model(ctx) is None:
        raise MCPToolError(
            f"{TOOL_NAME} needs the MCP server's own model: start it with "
            f"--llm-policy agno-model (current policy: {policy or 'external'!r}).",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    run_context = getattr(ctx, "run_context", None)
    run = getattr(run_context, "run", None)
    if run is None:
        raise MCPToolError(
            f"{TOOL_NAME} needs an active MCP session run.",
            code=MCPErrorCode.INVALID_INPUT,
        )
    if not is_ad_hoc_run(run):
        raise MCPToolError(
            f"{TOOL_NAME} runs only in the ad-hoc MCP session run; catalog workflow "
            f"run {run.run_id!r} ({run.workflow_slug}) must be driven task by task.",
            code=MCPErrorCode.PERMISSION_DENIED,
        )
    return run_context


def _split_session_state(state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split MCP state into a JSON snapshot and in-memory values (kept by reference)."""

    snapshot: dict[str, Any] = {}
    by_reference: dict[str, Any] = {}
    for key, value in list(state.items()):
        try:
            snapshot[key] = json.loads(json.dumps(value))
        except (TypeError, ValueError, OverflowError, RecursionError):
            by_reference[key] = value
    return snapshot, by_reference


def _shareable_changes(
    base: Mapping[str, Any],
    state: Mapping[str, Any],
    by_reference: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return the team's final state as the MCP merge expects it, plus unshared keys.

    Execution-scope and team-private keys keep their MCP values. In-memory
    values are never merged: the session's own objects were shared by
    reference, and the keys of objects the team created or replaced are
    reported instead.
    """

    kept = _EXECUTION_SCOPE_KEYS | TEAM_PRIVATE_STATE_KEYS
    shared: dict[str, Any] = {}
    unshared: list[str] = []
    for key, value in state.items():
        if key in kept:
            continue
        if key in by_reference:
            if value is not by_reference[key]:
                unshared.append(str(key))
            continue
        try:
            shared[key] = json.loads(json.dumps(value))
        except (TypeError, ValueError, OverflowError, RecursionError):
            unshared.append(str(key))
    for key in kept | set(unshared):
        if key in base:
            shared[key] = base[key]
    return shared, sorted(unshared)


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    model_dump = getattr(content, "model_dump", None)
    if callable(model_dump):
        content = model_dump()
    try:
        return json.dumps(content, default=str)
    except (TypeError, ValueError):
        return str(content)


def _settle_turn(session_runs: Any, team: Any, binding: Any, failed: bool) -> None:
    """Wait for the team's in-flight tool calls, then close the turn's tasks."""

    if binding is not None and not binding.wait_idle(_DRAIN_TIMEOUT_S):
        logger.warning(
            "%s: team tool calls were still running after %ss", TOOL_NAME, _DRAIN_TIMEOUT_S
        )
    try:
        session_runs.finalize_agno_turn(team, failed=failed)
    except Exception as exc:  # noqa: BLE001 - task bookkeeping must not mask the result
        logger.warning("%s: could not finalize the team's tasks: %s", TOOL_NAME, exc)
