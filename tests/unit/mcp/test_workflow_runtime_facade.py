"""MCP lifecycle coverage for the shared v2 workflow runtime."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

import cs_copilot.workflows as workflows_module
from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.facades.workflows import WorkflowRuntimeFacade
from cs_copilot.mcp.tool_adapter import build_tool
from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext, get_workflow


@pytest.fixture
def runtime_ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_S3", "false")
    monkeypatch.chdir(tmp_path)
    old_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/mcp-runtime-facade")
    ctx = MCPAgentContext()
    ctx.run_context = RunContext.create(
        "pilot",
        session_state=ctx.session_state,
        run_id="mcp-runtime-run",
    )
    try:
        yield ctx
    finally:
        S3.set_session_prefix(old_prefix)


def _tool(name: str, ctx: MCPAgentContext):
    spec = next(item for item in all_specs() if item.mcp_name == name)
    return build_tool(spec, spec.toolkit_factory(), ctx)


def _call(name: str, ctx: MCPAgentContext, **kwargs):
    return asyncio.run(_tool(name, ctx)(**kwargs))


def _catalog_handoff_contract(task_id: str) -> dict[str, list[str]]:
    task = next(
        item for item in get_workflow("chembl-to-gtm-report").tasks if item.task_id == task_id
    )
    return {
        "required_capabilities": list(task.required_tools),
        "input_artifact_contracts": list(task.input_artifacts),
        "expected_output_artifacts": list(task.output_artifacts),
        "acceptance_criteria": list(task.acceptance_criteria),
        "budget": {
            "max_tokens": 8_000,
            "max_tool_calls": 24,
            "timeout_seconds": 900,
        },
    }


def test_mcp_runtime_tools_manage_tasks_handoffs_artifacts_and_completion(runtime_ctx):
    assert _call("workflow_transition_run", runtime_ctx, status="running")["status"] == "success"
    created = _call(
        "workflow_add_task",
        runtime_ctx,
        task_id="analyze",
        role="gtm_agent",
        profile="gtm-analysis",
        step="Analyze the registered dataset",
    )
    assert created["data"]["status"] == "pending"
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="analyze",
        status="running",
    )
    assert runtime_ctx.session_state["active_task_id"] == "analyze"
    assert runtime_ctx.session_state["active_role"] == "gtm_agent"
    assert runtime_ctx.session_state["active_profile"] == "gtm-analysis"

    with S3.open("workflows/mcp-runtime-run/clean.csv", "w") as handle:
        handle.write("smiles\nCCO\n")
    registered = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="clean.csv",
        artifact_type="clean_dataset_path",
        mime_type="text/csv",
        artifact_id="clean-dataset",
    )
    assert registered["data"]["sha256"]
    assert registered["artifact_ids"] == ["clean-dataset"]

    handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="analyze",
        sender_role="supervisor",
        receiver_role="gtm_agent",
        objective="Analyze the clean dataset.",
        input_artifact_ids=["clean-dataset"],
        expected_output_artifacts=["analysis_summary"],
        acceptance_criteria=["Use only the registered input."],
        budget={"max_tool_calls": 4},
    )
    assert handoff["data"]["receiver_role"] == "gtm_agent"

    verified = _call(
        "workflow_verify_artifact",
        runtime_ctx,
        artifact_id="clean-dataset",
    )
    assert verified["data"]["artifact_id"] == "clean-dataset"
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="analyze",
        status="completed",
    )
    assert "active_task_id" not in runtime_ctx.session_state
    assert "active_role" not in runtime_ctx.session_state
    assert "active_profile" not in runtime_ctx.session_state
    completed = _call(
        "workflow_complete_run",
        runtime_ctx,
        required_artifact_types=["clean_dataset_path"],
        required_task_ids=["analyze"],
    )
    assert completed["data"]["status"] == "completed"


def test_mcp_runtime_invalid_transition_is_a_normalized_error(runtime_ctx):
    result = _call("workflow_transition_run", runtime_ctx, status="completed")

    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_input"
    assert "illegal run transition" in result["error"]["message"]


def test_mcp_abandon_tool_invocation_requires_confirmation_and_inactive_span(
    runtime_ctx,
    monkeypatch,
):
    _call("workflow_transition_run", runtime_ctx, status="running")
    _call(
        "workflow_add_task",
        runtime_ctx,
        task_id="analyze",
        role="gtm_agent",
        profile="gtm-analysis",
        step="Analyze a registered dataset.",
    )
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="analyze",
        status="running",
    )
    runtime = runtime_ctx.run_context
    task = runtime.run.tasks["analyze"]
    span_id = "orphaned-facade-span"
    runtime.append_event(
        "tool_progress",
        {
            "runtime": "mcp",
            "session_id": runtime.run.session_id,
            "run_id": runtime.run.run_id,
            "workflow_slug": runtime.run.workflow_slug,
            "trace_id": runtime.run.trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "tool_name": "gtm_create",
            "task_id": task.task_id,
            "role": task.role,
            "profile": task.profile,
            "stage": "started",
            "attempt": 0,
            "max_attempts": 1,
            "cached": False,
            "task_attempt": task.attempts,
            "handoff_id": None,
        },
    )
    reason = "confirmed the prior server process and its worker have stopped"

    missing_confirmation = _call(
        "workflow_abandon_tool_invocation",
        runtime_ctx,
        span_id=span_id,
        reason=reason,
    )

    assert missing_confirmation["status"] == "error"
    assert missing_confirmation["error"]["code"] == "invalid_input"
    assert "confirm_not_running=true is required" in missing_confirmation["error"]["message"]
    assert runtime.pending_tool_invocations(domain_only=True) == (f"gtm_create ({span_id})",)

    monkeypatch.setattr(
        "cs_copilot.mcp.manifests.is_tool_span_active",
        lambda candidate: candidate == span_id,
    )
    active = _call(
        "workflow_abandon_tool_invocation",
        runtime_ctx,
        span_id=span_id,
        reason=reason,
        confirm_not_running=True,
    )

    assert active["status"] == "error"
    assert active["error"]["code"] == "invalid_input"
    assert "still active in this server process" in active["error"]["message"]
    assert runtime.pending_tool_invocations(domain_only=True) == (f"gtm_create ({span_id})",)

    monkeypatch.setattr(
        "cs_copilot.mcp.manifests.is_tool_span_active",
        lambda _candidate: False,
    )
    recovered = _call(
        "workflow_abandon_tool_invocation",
        runtime_ctx,
        span_id=span_id,
        reason=reason,
        confirm_not_running=True,
    )

    assert recovered["status"] == "success"
    assert recovered["data"]["payload"]["stage"] == "abandoned"
    assert recovered["data"]["payload"]["recovery"] == {
        "confirmed_not_running": True,
        "reason": reason,
    }
    assert runtime.pending_tool_invocations(domain_only=True) == ()


def _use_placeholder_run(ctx: MCPAgentContext, *, profile: str = "standard") -> None:
    ctx.session_state["mcp_profile"] = profile
    ctx.run_context = RunContext.create(
        "mcp-session",
        session_state=ctx.session_state,
        run_id="mcp-placeholder-run",
    )


def test_start_run_materializes_catalog_tasks_and_reuses_matching_contract(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    constraints = {"target": "EGFR"}
    budget = {"max_tool_calls": 12}
    workflow_inputs = {"retrieval_request": {"target": "EGFR"}}

    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        constraints=constraints,
        budget=budget,
        workflow_inputs=workflow_inputs,
    )

    assert started["status"] == "success"
    data = started["data"]
    assert data["workflow_slug"] == "chembl-to-gtm-report"
    assert data["profile"] == "standard"
    assert data["reused"] is False
    assert data["constraints"] == constraints
    assert data["budget"] == budget
    assert data["workflow_inputs"] == {"retrieval_request": "workflow-input-retrieval_request"}
    root_input = runtime_ctx.run_context.run.artifacts[data["workflow_inputs"]["retrieval_request"]]
    assert root_input.artifact_type == "retrieval_request"
    assert root_input.producer_tool == "workflow_start_run"
    assert root_input.provenance["input_contract"]["kind"] == "request"
    assert root_input.trust.value == "external"
    assert data["output_context"]["run_id"] == data["run_id"]
    assert started["trace"]["run_id"] == data["run_id"]
    assert started["trace"]["trace_id"] == runtime_ctx.run_context.run.trace_id
    assert data["materialized_task_ids"][0] == "chembl-preflight"
    assert {task["task_id"] for task in data["tasks"]} == {
        "chembl-preflight",
        "chembl-retrieval",
        "gtm-preflight",
        "gtm-model",
        "gtm-landscapes",
        "report",
    }
    run_id = data["run_id"]
    runtime = runtime_ctx.run_context

    reused = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
    )

    assert reused["status"] == "success"
    assert reused["data"]["reused"] is True
    assert reused["data"]["run_id"] == run_id
    assert reused["data"]["materialized_task_ids"] == []
    assert runtime_ctx.run_context is runtime

    same_inputs = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs=workflow_inputs,
    )
    assert same_inputs["status"] == "success"
    assert same_inputs["data"]["reused"] is True

    changed_inputs = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "HER2"}},
    )
    assert changed_inputs["status"] == "error"
    assert "differs from the pinned value" in changed_inputs["error"]["message"]

    explicit_empty_constraints = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        constraints={},
    )
    assert explicit_empty_constraints["status"] == "error"
    assert "different constraints or budget" in explicit_empty_constraints["error"]["message"]

    explicit_empty_budget = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        budget={},
    )
    assert explicit_empty_budget["status"] == "error"
    assert "different constraints or budget" in explicit_empty_budget["error"]["message"]

    explicit_empty_inputs = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={},
    )
    assert explicit_empty_inputs["status"] == "error"
    assert "pinned workflow inputs omitted" in explicit_empty_inputs["error"]["message"]

    changed = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        constraints={"target": "HER2"},
        budget=budget,
    )
    assert changed["status"] == "error"
    assert changed["error"]["code"] == "invalid_input"
    assert "different constraints or budget" in changed["error"]["message"]


def test_catalog_handoff_accepts_contract_names_and_derives_run_identity(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    _call("workflow_transition_run", runtime_ctx, status="planning")
    _call("workflow_transition_run", runtime_ctx, status="running")

    handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Clarify and validate the ChEMBL retrieval request.",
        context_summary="Root catalog task; use only the request and registered artifacts.",
        **_catalog_handoff_contract("chembl-preflight"),
    )

    assert handoff["status"] == "success"
    assert handoff["data"]["run_id"] == started["data"]["run_id"]
    assert handoff["data"]["workflow_slug"] == "chembl-to-gtm-report"
    assert handoff["data"]["trace_id"] == runtime_ctx.run_context.run.trace_id
    assert handoff["data"]["span_id"]
    assert handoff["data"]["input_artifact_ids"] == ["workflow-input-retrieval_request"]
    assert handoff["data"]["expected_output_artifacts"] == ["retrieval_plan"]

    missing = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-retrieval",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Retrieve compounds from the clarified plan.",
        **_catalog_handoff_contract("chembl-retrieval"),
    )
    assert missing["status"] == "error"
    assert (
        "missing required registered input artifact contracts: retrieval_plan"
        in missing["error"]["message"]
    )

    invalid_contract = _catalog_handoff_contract("chembl-preflight")
    invalid_contract["input_artifact_contracts"] = ["clean_dataset_path"]
    invalid = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Use an invalid artifact contract.",
        **invalid_contract,
    )
    assert invalid["status"] == "error"
    assert (
        "input artifact contracts must match the pinned task contract"
        in invalid["error"]["message"]
    )

    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    run_id = runtime_ctx.run_context.run.run_id
    for index in (1, 2):
        filename = f"retrieval-plan-{index}.json"
        with S3.open(f"workflows/{run_id}/{filename}", "w") as handle:
            handle.write('{"can_proceed": true}')
        registered = runtime_ctx.run_context.register_artifact(
            filename,
            artifact_type="retrieval_plan",
            mime_type="application/json",
            artifact_id=f"retrieval-plan-{index}",
            producer_task_id="chembl-preflight",
            active_task_id="chembl-preflight",
            producer_tool="chembl_prepare_retrieval",
        )
        assert registered.artifact_id == f"retrieval-plan-{index}"

    ambiguous = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-retrieval",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Retrieve compounds from the clarified plan.",
        **_catalog_handoff_contract("chembl-retrieval"),
    )
    assert ambiguous["status"] == "error"
    assert "input artifact contract 'retrieval_plan' is ambiguous" in ambiguous["error"]["message"]

    selected = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-retrieval",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Retrieve compounds using the explicitly selected plan.",
        input_artifact_ids=["retrieval-plan-2"],
        **_catalog_handoff_contract("chembl-retrieval"),
    )
    assert selected["status"] == "success"
    assert selected["data"]["input_artifact_ids"] == [
        "workflow-input-retrieval_request",
        "retrieval-plan-2",
    ]
    assert runtime_ctx.run_context.run.tasks["chembl-retrieval"].input_artifact_ids == [
        "workflow-input-retrieval_request",
        "retrieval-plan-2",
    ]


def test_required_root_input_blocks_handoff_task_activation_and_completion(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
    )
    assert started["status"] == "success"
    _call("workflow_transition_run", runtime_ctx, status="running")

    handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate a request that was never persisted.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    assert handoff["status"] == "error"
    assert (
        "missing required registered input artifact contracts: retrieval_request"
        in handoff["error"]["message"]
    )

    activation = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    assert activation["status"] == "error"
    assert "missing workflow input retrieval_request" in activation["error"]["message"]

    completion = _call("workflow_complete_run", runtime_ctx)
    assert completion["status"] == "success"
    assert completion["data"]["status"] == "partial"
    completion_reason = next(
        event.payload["reason"]
        for event in reversed(runtime_ctx.run_context.events)
        if event.event_type == "run_status_changed" and "reason" in event.payload
    )
    assert "missing required workflow input artifacts: retrieval_request" in completion_reason


def test_catalog_task_requires_handoff_and_fresh_handoff_after_failure(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    _call("workflow_transition_run", runtime_ctx, status="running")

    missing_handoff = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    assert missing_handoff["status"] == "error"
    assert "validated structured handoff" in missing_handoff["error"]["message"]

    first_handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate the persisted retrieval request.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    assert first_handoff["status"] == "success"
    assert first_handoff["data"]["task_attempt"] == 0
    assert (
        _call(
            "workflow_transition_task",
            runtime_ctx,
            task_id="chembl-preflight",
            status="running",
        )["status"]
        == "success"
    )

    failed = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="failed",
        error_code="transient_external",
        error_message="temporary service failure",
        error_retryable=True,
    )
    assert failed["status"] == "success"
    stale_retry = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    assert stale_retry["status"] == "error"
    assert "fresh validated structured handoff" in stale_retry["error"]["message"]

    retry_handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Retry validation after the temporary service failure.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    assert retry_handoff["status"] == "success"
    assert retry_handoff["data"]["task_attempt"] == 1
    retried = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    assert retried["status"] == "success"
    assert retried["data"]["attempts"] == 2


@pytest.mark.parametrize(
    ("budget", "message"),
    [
        ({"max_tool_calls": 2, "timeout_seconds": 30}, "missing required budget fields"),
        (
            {"max_tokens": 100.5, "max_tool_calls": 2, "timeout_seconds": 30},
            "max_tokens must be a positive integer",
        ),
        (
            {"max_tokens": 100, "max_tool_calls": 2.5, "timeout_seconds": 30},
            "max_tool_calls must be a positive integer",
        ),
        (
            {"max_tokens": 100, "max_tool_calls": 2, "timeout_seconds": float("inf")},
            "timeout_seconds must be a finite positive number",
        ),
    ],
)
def test_catalog_handoff_requires_complete_bounded_budget(runtime_ctx, budget, message):
    _use_placeholder_run(runtime_ctx)
    _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    contract = _catalog_handoff_contract("chembl-preflight")
    contract["budget"] = budget

    handoff = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate the persisted retrieval request.",
        **contract,
    )

    assert handoff["status"] == "error"
    assert message in handoff["error"]["message"]


def test_catalog_artifact_producer_must_be_the_active_running_task(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    _call("workflow_transition_run", runtime_ctx, status="running")
    run_id = started["data"]["run_id"]
    with S3.open(f"workflows/{run_id}/plans/retrieval.json", "w") as handle:
        handle.write('{"can_proceed": true}')

    pending_producer = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="plans/retrieval.json",
        artifact_type="retrieval_plan",
        mime_type="application/json",
        producer_task_id="chembl-preflight",
    )
    assert pending_producer["status"] == "error"
    assert "cannot assert producer identity" in pending_producer["error"]["message"]

    _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate the persisted retrieval request.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    forged_producer = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="plans/retrieval.json",
        artifact_type="raw_dataset_path",
        mime_type="application/json",
        producer_task_id="chembl-retrieval",
    )
    assert forged_producer["status"] == "error"
    assert "cannot assert producer identity" in forged_producer["error"]["message"]


def test_public_artifact_registration_cannot_spoof_internal_model_provenance(
    runtime_ctx,
):
    run_id = runtime_ctx.run_context.run.run_id
    with S3.open(f"workflows/{run_id}/malicious.pkl.gz", "wb") as handle:
        handle.write(b"not-a-trusted-model")

    forged_trust = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="malicious.pkl.gz",
        artifact_type="gtm_model_path",
        mime_type="application/gzip",
        trust="internal",
    )
    assert forged_trust["status"] == "error"
    assert "accepts only external or untrusted" in forged_trust["error"]["message"]

    forged_producer = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="malicious.pkl.gz",
        artifact_type="gtm_model_path",
        mime_type="application/gzip",
        producer_tool="gtm_save_model_and_data",
    )
    assert forged_producer["status"] == "error"
    assert "cannot assert producer identity" in forged_producer["error"]["message"]


def test_file_backed_workflow_input_requires_and_pins_registered_artifact(runtime_ctx):
    _use_placeholder_run(runtime_ctx, profile="chemoinformatics")

    fabricated = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="dataset-normalization",
        workflow_inputs={"source_dataset": {"rows": [{"smiles": "CCO"}]}},
    )
    assert fabricated["status"] == "error"
    assert "file-backed kind 'dataset'" in fabricated["error"]["message"]
    assert runtime_ctx.run_context.run.workflow_slug == "mcp-session"

    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="dataset-normalization",
    )
    assert started["status"] == "success"
    run_id = started["data"]["run_id"]
    with S3.open(f"workflows/{run_id}/inputs/source.csv", "w") as handle:
        handle.write("smiles\nCCO\n")
    registered = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="inputs/source.csv",
        artifact_type="source_dataset",
        mime_type="text/csv",
        artifact_id="source-dataset",
        provenance={"source": "user_upload"},
        trust="external",
    )
    assert registered["status"] == "success"
    assert runtime_ctx.run_context.run.workflow_inputs == {"source_dataset": "source-dataset"}

    reused = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="dataset-normalization",
        workflow_inputs={"source_dataset": {"artifact_id": "source-dataset"}},
    )
    assert reused["status"] == "success"
    assert reused["data"]["reused"] is True

    with S3.open(f"workflows/{run_id}/inputs/other.csv", "w") as handle:
        handle.write("smiles\nCCC\n")
    duplicate = _call(
        "workflow_register_artifact",
        runtime_ctx,
        path="inputs/other.csv",
        artifact_type="source_dataset",
        mime_type="text/csv",
        artifact_id="other-source",
    )
    assert duplicate["status"] == "error"
    assert "already bound to artifact 'source-dataset'" in duplicate["error"]["message"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "input_artifact_contracts",
            [],
            "input artifact contracts must match the pinned task contract",
        ),
        (
            "expected_output_artifacts",
            [],
            "expected output artifacts must match the pinned task contract",
        ),
        (
            "required_capabilities",
            [],
            "required capabilities must match the pinned task contract",
        ),
        (
            "required_capabilities",
            ["chembl_prepare_retrieval", "gtm_optimization"],
            "required capabilities must match the pinned task contract",
        ),
        (
            "acceptance_criteria",
            [],
            "acceptance criteria must match the pinned task contract",
        ),
    ],
)
def test_catalog_handoff_cannot_omit_pinned_task_contract_fields(
    runtime_ctx,
    field,
    value,
    message,
):
    _use_placeholder_run(runtime_ctx)
    _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    contract = _catalog_handoff_contract("chembl-preflight")
    contract[field] = value

    result = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Clarify and validate the ChEMBL retrieval request.",
        **contract,
    )

    assert result["status"] == "error"
    assert message in result["error"]["message"]


def test_catalog_handoff_rejects_explicit_artifact_with_wrong_contract_type(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    _call("workflow_transition_run", runtime_ctx, status="planning")
    _call("workflow_transition_run", runtime_ctx, status="running")
    _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate the persisted retrieval request.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    run_id = runtime_ctx.run_context.run.run_id
    with S3.open(f"workflows/{run_id}/retrieval-plan.json", "w") as handle:
        handle.write('{"can_proceed": true}')
    registered = runtime_ctx.run_context.register_artifact(
        "retrieval-plan.json",
        artifact_type="retrieval_plan",
        mime_type="application/json",
        artifact_id="wrong-type-for-gtm",
        producer_task_id="chembl-preflight",
        active_task_id="chembl-preflight",
        producer_tool="chembl_prepare_retrieval",
    )
    assert registered.artifact_id == "wrong-type-for-gtm"

    result = _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="gtm-preflight",
        sender_role="supervisor",
        receiver_role="gtm_agent",
        objective="Plan the GTM analysis.",
        input_artifact_ids=["wrong-type-for-gtm"],
        **_catalog_handoff_contract("gtm-preflight"),
    )

    assert result["status"] == "error"
    assert "types are not declared by the pinned task contract" in result["error"]["message"]


def test_start_run_reuses_pinned_contract_when_live_catalog_changes(
    runtime_ctx,
    monkeypatch,
):
    runtime_ctx.session_state["mcp_profile"] = "standard"
    runtime_ctx.run_context = RunContext.create(
        "chembl-to-gtm-report",
        session_state=runtime_ctx.session_state,
        run_id="startup-pinned-run",
    )
    pinned_hash = runtime_ctx.run_context.run.workflow_contract["contract_sha256"]
    live = get_workflow("chembl-to-gtm-report")
    mutated = replace(
        live,
        profiles=("reporting",),
        preflight_tools=("removed_preflight_tool",),
        required_tools=("removed_required_tool",),
        tasks=(),
    )
    monkeypatch.setattr(workflows_module, "get_workflow", lambda _slug: mutated)

    reused = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
    )

    assert reused["status"] == "success"
    assert reused["data"]["reused"] is True
    assert reused["data"]["materialized_task_ids"][0] == "chembl-preflight"
    assert runtime_ctx.run_context.run.workflow_contract["contract_sha256"] == pinned_hash
    assert "chembl-preflight" in runtime_ctx.run_context.run.tasks


def test_start_run_enforces_profile_and_active_run_replacement_rules(runtime_ctx):
    _use_placeholder_run(runtime_ctx, profile="reporting")
    incompatible = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-target-retrieval",
    )
    assert incompatible["status"] == "error"
    assert "does not permit MCP profile 'reporting'" in incompatible["error"]["message"]

    runtime_ctx.session_state["mcp_profile"] = "standard"
    active = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="dataset-normalization",
    )
    active_run_id = active["data"]["run_id"]
    rejected = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-target-retrieval",
    )
    assert rejected["status"] == "error"
    assert "nonterminal catalog run" in rejected["error"]["message"]
    assert runtime_ctx.run_context.run.run_id == active_run_id

    _call("workflow_transition_run", runtime_ctx, status="running")
    _call("workflow_transition_run", runtime_ctx, status="cancelled")
    replacement = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-target-retrieval",
    )
    assert replacement["status"] == "success"
    assert replacement["data"]["run_id"] != active_run_id
    assert replacement["data"]["workflow_slug"] == "chembl-target-retrieval"


def test_start_run_uses_explicit_empty_session_state(runtime_ctx):
    runtime_ctx.session_state["mcp_profile"] = "standard"

    with pytest.raises(ValueError, match="mcp_profile"):
        WorkflowRuntimeFacade().start_run(
            "dataset-normalization",
            agent=runtime_ctx,
            session_state={},
        )


def test_terminal_task_only_clears_matching_active_task(runtime_ctx):
    _call("workflow_transition_run", runtime_ctx, status="running")
    for task_id, role, profile in (
        ("first", "chemoinformatician", "chemoinformatics"),
        ("second", "gtm_agent", "gtm-analysis"),
    ):
        _call(
            "workflow_add_task",
            runtime_ctx,
            task_id=task_id,
            role=role,
            profile=profile,
            step=f"Run {task_id}",
        )
        _call(
            "workflow_transition_task",
            runtime_ctx,
            task_id=task_id,
            status="running",
        )

    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="first",
        status="completed",
    )
    assert runtime_ctx.session_state["active_task_id"] == "second"
    assert runtime_ctx.session_state["active_role"] == "gtm_agent"
    assert runtime_ctx.session_state["active_profile"] == "gtm-analysis"

    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="second",
        status="cancelled",
    )
    assert "active_task_id" not in runtime_ctx.session_state
    assert "active_role" not in runtime_ctx.session_state
    assert "active_profile" not in runtime_ctx.session_state


def test_input_required_task_clears_active_execution_scope(runtime_ctx):
    _call("workflow_transition_run", runtime_ctx, status="running")
    _call(
        "workflow_add_task",
        runtime_ctx,
        task_id="clarify",
        role="chemoinformatician",
        profile="chemoinformatics",
        step="Request missing scientific input.",
    )
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="clarify",
        status="running",
    )
    assert runtime_ctx.session_state["active_task_id"] == "clarify"

    result = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="clarify",
        status="input_required",
    )

    assert result["status"] == "success"
    assert result["data"]["status"] == "input_required"
    assert "active_task_id" not in runtime_ctx.session_state
    assert "active_role" not in runtime_ctx.session_state
    assert "active_profile" not in runtime_ctx.session_state
    assert "active_task_attempt" not in runtime_ctx.session_state
    assert "active_handoff_id" not in runtime_ctx.session_state


def test_input_required_run_clears_and_running_restores_unique_task_scope(runtime_ctx):
    _call("workflow_transition_run", runtime_ctx, status="running")
    _call(
        "workflow_add_task",
        runtime_ctx,
        task_id="clarify",
        role="chemoinformatician",
        profile="chemoinformatics",
        step="Pause the run for missing scientific input.",
    )
    _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="clarify",
        status="running",
    )
    assert runtime_ctx.session_state["active_task_attempt"] == 1

    paused = _call(
        "workflow_transition_run",
        runtime_ctx,
        status="input_required",
    )

    assert paused["status"] == "success"
    assert "active_task_id" not in runtime_ctx.session_state
    assert "active_task_attempt" not in runtime_ctx.session_state

    resumed = _call(
        "workflow_transition_run",
        runtime_ctx,
        status="running",
    )

    assert resumed["status"] == "success"
    assert runtime_ctx.session_state["active_task_id"] == "clarify"
    assert runtime_ctx.session_state["active_task_attempt"] == 1


def test_running_task_can_be_reactivated_without_duplicate_lifecycle_event(runtime_ctx):
    _call("workflow_transition_run", runtime_ctx, status="running")
    for task_id, role, profile in (
        ("first", "chemoinformatician", "chemoinformatics"),
        ("second", "gtm_agent", "gtm-analysis"),
    ):
        _call(
            "workflow_add_task",
            runtime_ctx,
            task_id=task_id,
            role=role,
            profile=profile,
            step=f"Run {task_id}",
        )
        _call(
            "workflow_transition_task",
            runtime_ctx,
            task_id=task_id,
            status="running",
        )
    before = sum(
        event.event_type == "task_status_changed" for event in runtime_ctx.run_context.events
    )

    activated = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="first",
        status="running",
    )

    assert activated["status"] == "success"
    assert activated["data"]["status"] == "running"
    assert runtime_ctx.session_state["active_task_id"] == "first"
    assert runtime_ctx.session_state["active_role"] == "chemoinformatician"
    assert runtime_ctx.session_state["active_profile"] == "chemoinformatics"
    assert (
        sum(event.event_type == "task_status_changed" for event in runtime_ctx.run_context.events)
        == before
    )


def test_preflight_artifact_is_attributed_to_materialized_active_task(runtime_ctx):
    _use_placeholder_run(runtime_ctx)
    started = _call(
        "workflow_start_run",
        runtime_ctx,
        workflow_slug="chembl-to-gtm-report",
        workflow_inputs={"retrieval_request": {"target": "EGFR"}},
    )
    assert started["status"] == "success"
    assert runtime_ctx.run_context.run.tasks["chembl-preflight"].status.value == "pending"

    _call("workflow_transition_run", runtime_ctx, status="running")
    _call(
        "workflow_record_handoff",
        runtime_ctx,
        task_id="chembl-preflight",
        sender_role="supervisor",
        receiver_role="chembl_downloader",
        objective="Validate the persisted retrieval request.",
        **_catalog_handoff_contract("chembl-preflight"),
    )
    task = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="running",
    )
    assert task["status"] == "success"
    assert runtime_ctx.session_state["active_task_id"] == "chembl-preflight"

    preflight = _call(
        "chembl_prepare_retrieval",
        runtime_ctx,
        target="EGFR",
        target_type="protein",
        organism="Homo sapiens",
        assay_types=["binding"],
        mechanism="any",
    )
    assert preflight["status"] == "success"
    plans = [
        runtime_ctx.run_context.run.artifacts[artifact_id]
        for artifact_id in preflight["artifact_ids"]
        if runtime_ctx.run_context.run.artifacts[artifact_id].artifact_type == "retrieval_plan"
    ]
    assert len(plans) == 1
    assert plans[0].producer_task_id == "chembl-preflight"
    assert plans[0].producer_tool == "chembl_prepare_retrieval"

    completed = _call(
        "workflow_transition_task",
        runtime_ctx,
        task_id="chembl-preflight",
        status="completed",
    )
    assert completed["status"] == "success"
    assert "active_task_id" not in runtime_ctx.session_state
