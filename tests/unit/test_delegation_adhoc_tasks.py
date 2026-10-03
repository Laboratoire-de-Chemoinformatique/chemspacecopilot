"""Delegation into ad-hoc chat runs: invented task ids become durable tasks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from agno.exceptions import RetryAgentRun
from agno.tools.function import Function, FunctionCall

from cs_copilot.agents.delegation import DELEGATE_TOOL_NAME, StructuredDelegationGuard
from cs_copilot.agents.execution_binding import attach_execution, get_binding
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext, TaskRecord, TaskStatus


def _payload(**updates):
    payload = {
        "schema_version": 2,
        "run_id": "guessed-by-the-model",
        "workflow_slug": "ad-hoc",
        "task_id": "map-egfr",
        "sender_role": "coordinator",
        "receiver_role": "gtm_agent",
        "objective": "Build a GTM map of the retrieved EGFR inhibitors.",
        "constraints": [],
        "required_capabilities": ["gtm_optimization"],
        "input_artifact_ids": [],
        "expected_output_artifacts": ["gtm_model_path"],
        "expected_output_schema": {"type": "object"},
        "acceptance_criteria": ["The map is saved."],
        "context_summary": "EGFR inhibitors were retrieved.",
        "budget": {"max_tokens": 2_000, "max_tool_calls": 4, "timeout_seconds": 120},
        "trace_id": "trace-guess",
        "span_id": "span-1",
    }
    payload.update(updates)
    return payload


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/adhoc-chat")
    state: dict = {}
    run = RunContext.create("agno-session", session_state=state, session_id="adhoc-chat")
    run.transition_run("running")
    member = SimpleNamespace(
        name="gtm_agent_agent",
        agentic_role="gtm_agent",
        add_history_to_context=True,
        add_session_state_to_context=True,
        add_dependencies_to_context=True,
        tools=[],
    )
    team = SimpleNamespace(members=[member], tools=[], session_state=state, run_context=run)
    return team, run


def _delegate(team, payload, *, guard=None):
    guard = guard or StructuredDelegationGuard()

    def delegate_task_to_member(member_id: str, task_description: str, expected_output=None):
        return {"member_id": member_id, "task_description": task_description}

    function = Function.from_callable(delegate_task_to_member, name=DELEGATE_TOOL_NAME)
    function._team = team
    function._session_state = {"current_run_id": "agno-run-1"}
    function.pre_hook = guard.pre_hook
    call = FunctionCall(
        function=function,
        arguments={"member_id": "gtm-agent-agent", "task_description": json.dumps(payload)},
    )
    return call.execute()


def test_invented_tasks_are_created_started_and_attributed(chat):
    team, run = chat
    attach_execution(team, run_context=run, mode="observe")

    execution = _delegate(team, _payload())

    assert execution.status == "success"
    canonical = json.loads(execution.result["task_description"])
    assert canonical["run_id"] == run.run.run_id
    assert canonical["workflow_slug"] == "agno-session"
    assert canonical["trace_id"] == run.run.trace_id
    run.refresh()
    task = run.run.tasks["map-egfr"]
    assert (task.role, task.profile, task.status) == (
        "gtm_agent",
        "gtm-analysis",
        TaskStatus.RUNNING,
    )
    assert [handoff.task_id for handoff in run.run.handoffs] == ["map-egfr"]
    binding = get_binding(team)
    assert binding.active_tasks == {"gtm_agent": "map-egfr"}
    assert binding.turn_tasks == ["map-egfr"]


def test_redelegating_a_running_task_is_recorded(chat):
    team, run = chat
    attach_execution(team, run_context=run, mode="observe")
    _delegate(team, _payload())

    _delegate(team, _payload(objective="Refine the map with a finer grid."))

    run.refresh()
    assert len(run.run.handoffs) == 2
    assert run.run.tasks["map-egfr"].status is TaskStatus.RUNNING


def _claim_task_for_another_role(run) -> None:
    run.add_task(
        TaskRecord(
            task_id="map-egfr", role="chemoinformatician", profile="chemoinformatics", step="x"
        )
    )


def test_unknown_input_artifacts_are_dropped_not_fatal(chat):
    team, run = chat
    attach_execution(team, run_context=run, mode="observe")

    execution = _delegate(team, _payload(input_artifact_ids=["ds_001", "map_006"]))

    assert execution.status == "success"
    [handoff] = run.refresh().handoffs
    assert handoff.input_artifact_ids == ()
    assert run.run.tasks["map-egfr"].status is TaskStatus.RUNNING
    assert any("ds_001" in problem for problem in get_binding(team).problems)


def test_observe_mode_never_blocks_delegation_on_the_ledger(chat):
    team, run = chat
    attach_execution(team, run_context=run, mode="observe")
    _claim_task_for_another_role(run)

    execution = _delegate(team, _payload())

    assert execution.status == "success"
    assert run.refresh().handoffs == []
    assert any("does not match task role" in problem for problem in get_binding(team).problems)


def test_enforce_mode_keeps_delegation_fail_closed(chat):
    team, run = chat
    attach_execution(team, run_context=run, mode="enforce")
    _claim_task_for_another_role(run)

    with pytest.raises(RetryAgentRun, match="does not match task role"):
        _delegate(team, _payload())


def test_without_a_binding_ledger_failures_stay_fail_closed(chat):
    team, run = chat
    _claim_task_for_another_role(run)

    with pytest.raises(RetryAgentRun, match="does not match task role"):
        _delegate(team, _payload())


def test_a_failed_handoff_skips_the_task_it_created(chat, monkeypatch):
    team, run = chat
    attach_execution(team, run_context=run, mode="observe")

    def failing_record(envelope, **kwargs):
        raise ValueError("ledger unavailable")

    monkeypatch.setattr(run, "record_handoff", failing_record)
    execution = _delegate(team, _payload())

    assert execution.status == "success"
    assert run.refresh().tasks["map-egfr"].status is TaskStatus.SKIPPED
