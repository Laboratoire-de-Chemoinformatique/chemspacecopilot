"""Tests for guarded Agno specialist delegation."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from agno.agent import Agent
from agno.exceptions import RetryAgentRun
from agno.run.team import TeamRunOutput
from agno.session.team import TeamSession
from agno.team import Team
from agno.tools.function import Function, FunctionCall

from cs_copilot.agents.delegation import (
    DELEGATE_TOOL_NAME,
    DelegationLimits,
    StructuredDelegationGuard,
    StructuredHandoffTeam,
)
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext, TaskRecord


def _handoff_payload(**updates):
    payload = {
        "schema_version": 2,
        "run_id": "run-001",
        "workflow_slug": "chembl-to-gtm-report",
        "task_id": "report",
        "sender_role": "gtm_agent",
        "receiver_role": "report_generator",
        "objective": "Write the evidence-backed report.",
        "constraints": ["Do not infer missing activity values."],
        "required_capabilities": ["report_save_rich"],
        "input_artifact_ids": [],
        "expected_output_artifacts": ["html_report_path"],
        "expected_output_schema": {
            "type": "object",
            "required": ["html_report_path"],
        },
        "acceptance_criteria": ["Every scientific claim cites its source artifact."],
        "context_summary": "The validated GTM outputs are ready.",
        "budget": {
            "max_tokens": 2_000,
            "max_tool_calls": 4,
            "timeout_seconds": 120,
        },
        "trace_id": "trace-001",
        "span_id": "span-report",
    }
    payload.update(updates)
    return payload


def _team(runtime=None, *, role="report_generator"):
    member = SimpleNamespace(
        name=f"{role}_agent",
        agentic_role=role,
        add_history_to_context=True,
        add_session_state_to_context=True,
        add_dependencies_to_context=True,
    )
    return SimpleNamespace(members=[member], run_context=runtime)


def _delegate_function():
    def delegate_task_to_member(
        member_id: str,
        task_description: str,
        expected_output: str | None = None,
    ):
        return {
            "member_id": member_id,
            "task_description": task_description,
            "expected_output": expected_output,
        }

    return Function.from_callable(delegate_task_to_member, name=DELEGATE_TOOL_NAME)


def _execute_delegate(
    guard,
    team,
    payload,
    *,
    expected_output="untrusted extra context",
    pre_hook=None,
    coordinator_run_id="agno-run-001",
    member_id="report-generator-agent",
):
    function = _delegate_function()
    function._team = team
    function._session_state = {"current_run_id": coordinator_run_id}
    function.pre_hook = pre_hook or guard.pre_hook
    call = FunctionCall(
        function=function,
        arguments={
            "member_id": member_id,
            "task_description": json.dumps(payload),
            "expected_output": expected_output,
        },
    )
    return call.execute()


def test_guard_canonicalizes_and_records_only_structured_delegation():
    recorded = []
    runtime = SimpleNamespace(record_handoff=recorded.append)
    guard = StructuredDelegationGuard()
    team = _team(runtime)

    execution = _execute_delegate(guard, team, _handoff_payload())

    assert execution.status == "success"
    result = execution.result
    canonical = json.loads(result["task_description"])
    assert canonical["receiver_role"] == "report_generator"
    assert canonical["handoff_id"].startswith("handoff-")
    assert "untrusted extra context" not in result["expected_output"]
    assert len(recorded) == 1
    assert recorded[0].to_dict() == canonical
    member = team.members[0]
    assert member.add_history_to_context is False
    assert member.add_session_state_to_context is False
    assert member.add_dependencies_to_context is False


def test_guard_rejects_private_context_role_mismatch_and_unbounded_budget():
    guard = StructuredDelegationGuard()
    team = _team()

    private_payload = _handoff_payload(
        expected_output_schema={
            "type": "object",
            "properties": {"history": {"type": "array"}},
        }
    )
    with pytest.raises(RetryAgentRun, match="forbidden private/history"):
        _execute_delegate(guard, team, private_payload)

    with pytest.raises(RetryAgentRun, match="unknown fields: notes"):
        _execute_delegate(
            StructuredDelegationGuard(),
            team,
            _handoff_payload(notes="undeclared context channel"),
        )

    with pytest.raises(RetryAgentRun, match="does not match member"):
        _execute_delegate(
            StructuredDelegationGuard(),
            team,
            _handoff_payload(sender_role="coordinator", receiver_role="gtm_agent"),
        )

    with pytest.raises(RetryAgentRun, match="max_tool_calls exceeds"):
        _execute_delegate(
            StructuredDelegationGuard(),
            team,
            _handoff_payload(
                budget={
                    "max_tokens": 2_000,
                    "max_tool_calls": 25,
                    "timeout_seconds": 120,
                }
            ),
        )


def test_guard_blocks_identical_repeats_without_touching_other_tools():
    guard = StructuredDelegationGuard()
    team = _team()
    payload = _handoff_payload()

    _execute_delegate(guard, team, payload)
    with pytest.raises(RetryAgentRun, match="identical handoff"):
        _execute_delegate(guard, team, payload)

    function = Function.from_callable(
        lambda slug: {"slug": slug},
        name="fetch_skill",
    )
    call = FunctionCall(
        function=function,
        arguments={"slug": "report-generation"},
    )
    execution = call.execute()
    assert execution.status == "success"
    assert execution.result == {"slug": "report-generation"}


def test_bound_agno_run_id_prevents_session_state_counter_bypass():
    guard = StructuredDelegationGuard()
    team = _team()
    hook = guard.hook_for_run("immutable-agno-run")
    payload = _handoff_payload()

    _execute_delegate(
        guard,
        team,
        payload,
        pre_hook=hook,
        coordinator_run_id="mutable-value-one",
    )
    with pytest.raises(RetryAgentRun, match="identical handoff"):
        _execute_delegate(
            guard,
            team,
            payload,
            pre_hook=hook,
            coordinator_run_id="mutable-value-two",
        )


def test_guard_enforces_per_run_delegation_limit():
    limits = DelegationLimits(
        max_delegations_per_run=1,
        max_delegations_per_member=2,
        max_delegations_per_task=2,
        max_identical_handoffs=1,
    )
    guard = StructuredDelegationGuard(limits)
    team = _team()

    _execute_delegate(guard, team, _handoff_payload())
    with pytest.raises(RetryAgentRun, match="delegation limit reached"):
        _execute_delegate(
            guard,
            team,
            _handoff_payload(
                task_id="report-revision",
                context_summary="A distinct report revision is requested.",
                span_id="span-report-revision",
            ),
        )


def test_runtime_recording_failure_is_fail_closed_and_releases_reservation():
    guard = StructuredDelegationGuard()
    payload = _handoff_payload()

    def fail_recording(_envelope):
        raise OSError("event store unavailable")

    with pytest.raises(RetryAgentRun, match="event store unavailable"):
        _execute_delegate(
            guard,
            _team(SimpleNamespace(record_handoff=fail_recording)),
            payload,
        )

    recorded = []
    execution = _execute_delegate(
        guard,
        _team(SimpleNamespace(record_handoff=recorded.append)),
        payload,
    )
    assert execution.status == "success"
    assert len(recorded) == 1


def test_sync_guard_composes_with_agno_async_tool_execution():
    recorded = []
    guard = StructuredDelegationGuard()

    async def delegate_task_to_member(
        member_id: str,
        task_description: str,
        expected_output: str | None = None,
    ):
        return json.loads(task_description)["task_id"]

    function = Function.from_callable(delegate_task_to_member, name=DELEGATE_TOOL_NAME)
    function._team = _team(SimpleNamespace(record_handoff=recorded.append))
    function._session_state = {"current_run_id": "agno-async-run"}
    function.pre_hook = guard.pre_hook
    call = FunctionCall(
        function=function,
        arguments={
            "member_id": "report-generator-agent",
            "task_description": json.dumps(_handoff_payload()),
        },
    )

    execution = asyncio.run(call.aexecute())

    assert execution.status == "success"
    assert execution.result == "report"
    assert len(recorded) == 1


def test_structured_team_suppresses_implicit_member_context_channels(monkeypatch):
    captured = {}

    def fake_delegate_builder(self, *args, **kwargs):
        captured.update(kwargs)
        return Function.from_callable(lambda: "delegated", name=DELEGATE_TOOL_NAME)

    monkeypatch.setattr(Team, "_get_delegate_task_function", fake_delegate_builder)
    team = object.__new__(StructuredHandoffTeam)
    team.delegation_guard = StructuredDelegationGuard()

    result = team._get_delegate_task_function(
        add_history_to_context=True,
        add_session_state_to_context=True,
        add_dependencies_to_context=True,
    )

    assert callable(result.pre_hook)
    assert captured["add_history_to_context"] is False
    assert captured["add_session_state_to_context"] is False
    assert captured["add_dependencies_to_context"] is False


def test_structured_team_attaches_guard_to_agno_generated_delegate_function():
    member = Agent(name="report_generator_agent", telemetry=False)
    member.agentic_role = "report_generator"
    team = StructuredHandoffTeam(members=[member], telemetry=False)

    function = team._get_delegate_task_function(
        run_response=TeamRunOutput(run_id="agno-run-001"),
        session=TeamSession(session_id="session-001"),
        session_state={},
        team_run_context={},
    )

    assert function.name == DELEGATE_TOOL_NAME
    assert callable(function.pre_hook)
    assert function.pre_hook.__name__ == "validate_structured_delegation"
    schema = function.to_dict()
    assert "JSON-encoded v2 HandoffEnvelope" in schema["description"]
    assert (
        "complete structured handoff"
        in schema["parameters"]["properties"]["task_description"]["description"]
    )
    assert member.add_history_to_context is False
    assert member.add_session_state_to_context is False
    assert member.add_dependencies_to_context is False


def test_structured_team_rejects_a_configured_delegate_tool_collision():
    shadow = Function.from_callable(lambda: "shadowed", name=DELEGATE_TOOL_NAME)

    with pytest.raises(ValueError, match="cannot shadow Agno delegation tools"):
        StructuredHandoffTeam(members=[], tools=[shadow], telemetry=False)


def test_guard_records_through_an_attached_run_context(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_S3", "false")
    monkeypatch.chdir(tmp_path)
    old_prefix = S3.current_prefix()
    S3.set_session_prefix(f"sessions/delegation-{tmp_path.name}")
    try:
        runtime = RunContext.create(
            "chembl-to-gtm-report",
            run_id="run-001",
            trace_id="trace-001",
        )
        with S3.open("workflows/run-001/inputs/retrieval-request.json", "w") as handle:
            json.dump({"target": "EGFR"}, handle)
        request = runtime.register_artifact(
            "inputs/retrieval-request.json",
            artifact_type="retrieval_request",
            mime_type="application/json",
            artifact_id="retrieval-request",
            trust="external",
        )
        runtime.add_task(
            TaskRecord(
                task_id="chembl-preflight",
                role="chembl_downloader",
                profile="chembl-retrieval",
                step="Validate retrieval request",
            )
        )
        payload = _handoff_payload(
            task_id="chembl-preflight",
            sender_role="coordinator",
            receiver_role="chembl_downloader",
            objective="Validate the persisted retrieval request.",
            required_capabilities=["chembl_prepare_retrieval"],
            input_artifact_ids=[request.artifact_id],
            expected_output_artifacts=["retrieval_plan"],
            acceptance_criteria=[
                "Preflight either permits retrieval or records all required "
                "clarification questions."
            ],
        )

        execution = _execute_delegate(
            StructuredDelegationGuard(),
            _team(runtime, role="chembl_downloader"),
            payload,
            member_id="chembl-downloader-agent",
        )

        assert execution.status == "success"
        assert runtime.run is not None
        assert len(runtime.run.handoffs) == 1
        assert runtime.run.handoffs[0].task_id == "chembl-preflight"
        assert runtime.run.tasks["chembl-preflight"].input_artifact_ids == [request.artifact_id]
    finally:
        S3.set_session_prefix(old_prefix)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "required_capabilities",
            ["gtm_optimization"],
            "required capabilities must match the pinned task contract",
        ),
        (
            "expected_output_artifacts",
            ["html_report_path"],
            "expected output artifacts must match the pinned task contract",
        ),
        (
            "acceptance_criteria",
            ["Accept results without scientific validation."],
            "acceptance criteria must match the pinned task contract",
        ),
        (
            "input_artifact_ids",
            [],
            "missing required registered input artifact contracts",
        ),
    ],
)
def test_agno_guard_enforces_the_pinned_runtime_task_contract(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
):
    monkeypatch.setenv("USE_S3", "false")
    monkeypatch.chdir(tmp_path)
    old_prefix = S3.current_prefix()
    S3.set_session_prefix(f"sessions/delegation-contract-{tmp_path.name}")
    try:
        runtime = RunContext.create(
            "chembl-to-gtm-report",
            run_id="run-001",
            trace_id="trace-001",
        )
        with S3.open("workflows/run-001/inputs/retrieval-request.json", "w") as handle:
            json.dump({"target": "EGFR"}, handle)
        request = runtime.register_artifact(
            "inputs/retrieval-request.json",
            artifact_type="retrieval_request",
            mime_type="application/json",
            artifact_id="retrieval-request",
            trust="external",
        )
        runtime.add_task(
            TaskRecord(
                task_id="chembl-preflight",
                role="chembl_downloader",
                profile="chembl-retrieval",
                step="Validate retrieval request",
            )
        )
        payload = _handoff_payload(
            task_id="chembl-preflight",
            sender_role="coordinator",
            receiver_role="chembl_downloader",
            objective="Validate the persisted retrieval request.",
            required_capabilities=["chembl_prepare_retrieval"],
            input_artifact_ids=[request.artifact_id],
            expected_output_artifacts=["retrieval_plan"],
            acceptance_criteria=[
                "Preflight either permits retrieval or records all required "
                "clarification questions."
            ],
        )
        payload[field] = value
        event_count = len(runtime.events)

        with pytest.raises(RetryAgentRun, match=message):
            _execute_delegate(
                StructuredDelegationGuard(),
                _team(runtime, role="chembl_downloader"),
                payload,
                member_id="chembl-downloader-agent",
            )

        assert len(runtime.events) == event_count
        assert runtime.run.handoffs == []
    finally:
        S3.set_session_prefix(old_prefix)
