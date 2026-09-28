"""Tests for the v2 event-sourced workflow runtime."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import fsspec
import pytest

import cs_copilot.storage.layout as storage_layout
from cs_copilot.storage import OUTPUT_CONTEXT_KEY, S3
from cs_copilot.workflows import (
    SCHEMA_VERSION,
    ArtifactIntegrityError,
    EventReplayError,
    HandoffEnvelope,
    InvalidTransitionError,
    RunContext,
    RunStatus,
    RunStore,
    TaskRecord,
    TaskStatus,
    ToolError,
    ToolErrorCode,
    WorkflowRuntimeError,
)


@pytest.fixture
def runtime_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_S3", "false")
    monkeypatch.chdir(tmp_path)
    old_prefix = S3.current_prefix()
    S3.set_session_prefix(f"sessions/runtime-{tmp_path.name}")
    try:
        yield tmp_path
    finally:
        S3.set_session_prefix(old_prefix)


def test_create_distinguishes_session_run_and_workflow(runtime_storage):
    session_state = {}

    context = RunContext.create(
        "chembl-to-gtm-report",
        session_state=session_state,
        run_id="run-001",
        trace_id="trace-001",
    )

    assert context.run is not None
    assert context.run.schema_version == SCHEMA_VERSION == 2
    assert context.run.session_id.startswith("runtime-")
    assert context.run.run_id == "run-001"
    assert context.run.workflow_slug == "chembl-to-gtm-report"
    assert context.run.workflow_contract["version"] == "2.0.0"
    assert len(context.run.workflow_contract["contract_sha256"]) == 64
    assert context.run.workflow_contract["contract_schema_version"] == 1
    assert {task["task_id"] for task in context.run.workflow_contract["tasks"]} == {
        "chembl-preflight",
        "chembl-retrieval",
        "gtm-preflight",
        "gtm-model",
        "gtm-landscapes",
        "report",
    }
    dependencies = {
        contract["slug"]: contract
        for contract in context.run.workflow_contract["dependency_contracts"]
    }
    assert set(dependencies) == {
        "chembl-target-retrieval",
        "gtm-activity-landscape",
        "gtm-density-landscape",
    }
    assert all(contract["version"] == "2.0.0" for contract in dependencies.values())
    assert all(len(contract["contract_sha256"]) == 64 for contract in dependencies.values())
    assert all(contract["workflow_md"] for contract in dependencies.values())
    assert session_state[OUTPUT_CONTEXT_KEY] == {
        "layout_version": 4,
        "session_id": context.run.session_id,
        "run_id": "run-001",
        "workflow_slug": "chembl-to-gtm-report",
        "trace_id": "trace-001",
        "span_id": "trace-001",
        "parent_span_id": None,
    }

    event_files = list(_run_root(runtime_storage, context).glob("events/*.jsonl"))
    assert len(event_files) == 1
    lines = event_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["schema_version"] == 2
    assert event["event_type"] == "run_created"
    assert event["run_id"] == "run-001"


def test_explicit_session_identity_selects_the_matching_storage_root(runtime_storage):
    S3.set_session_prefix("sessions/unrelated-session")
    context = RunContext.create(
        "pilot",
        session_id="explicit-session",
        run_id="explicit-run",
    )

    expected_root = (
        runtime_storage / "data" / "sessions" / "explicit-session" / "workflows" / "explicit-run"
    )
    assert context.run.session_id == "explicit-session"
    assert expected_root.joinpath("manifest.json").is_file()

    S3.set_session_prefix("sessions/another-session")
    loaded = RunContext.load("explicit-run", session_id="explicit-session")
    assert loaded.run.run_id == "explicit-run"
    assert S3.current_prefix() == "sessions/explicit-session"


def test_run_and_task_transitions_are_validated(runtime_storage):
    context = RunContext.create("pilot", run_id="run-transitions")
    context.transition_run(RunStatus.PLANNING)
    context.transition_run(RunStatus.RUNNING)
    task = context.add_task(
        TaskRecord(
            task_id="fetch-chembl",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Fetch and normalize ChEMBL records",
        )
    )

    assert task.status is TaskStatus.PENDING
    assert context.transition_task(task.task_id, TaskStatus.RUNNING).attempts == 1
    failure = ToolError(
        ToolErrorCode.TRANSIENT_EXTERNAL,
        "ChEMBL timed out",
        retryable=True,
    )
    context.transition_task(task.task_id, TaskStatus.FAILED, error=failure)
    assert context.transition_task(task.task_id, TaskStatus.RUNNING).attempts == 2
    context.transition_task(task.task_id, TaskStatus.COMPLETED)

    event_count = len(context.events)
    with pytest.raises(InvalidTransitionError, match="completed -> running"):
        context.transition_task(task.task_id, TaskStatus.RUNNING)
    assert len(context.events) == event_count

    context.transition_run(RunStatus.COMPLETED)
    with pytest.raises(InvalidTransitionError, match="completed -> running"):
        context.transition_run(RunStatus.RUNNING)

    fresh = RunContext.create("pilot", run_id="failed-error-contract")
    with pytest.raises(ValueError, match="structured error"):
        fresh.transition_run(RunStatus.FAILED)


def test_artifact_registration_enforces_root_and_checksum(runtime_storage):
    context = RunContext.create("pilot", run_id="run-artifacts")
    context.transition_run(RunStatus.RUNNING)
    context.add_task(
        TaskRecord(
            task_id="fit-gtm",
            role="gtm_agent",
            profile="gtm-analysis",
            step="Fit GTM",
        )
    )
    context.transition_task("fit-gtm", TaskStatus.RUNNING)
    rel_path = "workflows/run-artifacts/01_chemical_space/datasets/clean.csv"
    with S3.open(rel_path, "w") as handle:
        handle.write("smiles,activity\nCCO,1\n")

    artifact = context.register_artifact(
        rel_path,
        artifact_id="clean-dataset",
        artifact_type="clean_dataset",
        mime_type="text/csv",
        producer_task_id="fit-gtm",
        producer_tool="clean_dataset",
        provenance={"source": "ChEMBL"},
    )

    assert artifact.relative_path == "01_chemical_space/datasets/clean.csv"
    assert artifact.size_bytes > 0
    assert len(artifact.sha256) == 64
    assert context.verify_artifact("clean-dataset") == artifact
    assert (
        context.register_artifact(
            rel_path,
            artifact_type="clean_dataset",
            mime_type="text/csv",
            producer_task_id="fit-gtm",
            producer_tool="clean_dataset",
        )
        == artifact
    )
    with pytest.raises(ValueError, match="different provenance"):
        context.register_artifact(
            rel_path,
            artifact_type="clean_dataset",
            mime_type="text/csv",
            producer_task_id="fit-gtm",
            producer_tool="clean_dataset",
            provenance={"source": "another database"},
        )
    assert context.run is not None
    assert context.run.tasks["fit-gtm"].output_artifact_ids == ["clean-dataset"]

    with pytest.raises(ValueError, match="outside workflow run"):
        context.register_artifact(
            "workflows/another-run/file.csv",
            artifact_type="invalid",
            mime_type="text/csv",
        )
    with pytest.raises(ValueError, match="traversal"):
        context.register_artifact(
            "../secret.txt",
            artifact_type="invalid",
            mime_type="text/plain",
        )
    outside = runtime_storage / "outside-run.txt"
    outside.write_text("not experiment-scoped", encoding="utf-8")
    symlink_path = _run_root(runtime_storage, context) / "results" / "escaped.txt"
    symlink_path.parent.mkdir(parents=True, exist_ok=True)
    symlink_path.symlink_to(outside)
    with pytest.raises(ValueError, match="resolves outside workflow run"):
        context.register_artifact(
            "results/escaped.txt",
            artifact_type="invalid",
            mime_type="text/plain",
        )
    hardlink_path = _run_root(runtime_storage, context) / "results" / "hard-linked.txt"
    hardlink_path.hardlink_to(outside)
    with pytest.raises(ValueError, match="non-linked"):
        context.register_artifact(
            "results/hard-linked.txt",
            artifact_type="invalid",
            mime_type="text/plain",
        )

    with S3.open(rel_path, "w") as handle:
        handle.write("tampered")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        context.verify_artifact("clean-dataset")


def test_local_artifact_digest_is_bound_to_open_directory_descriptors(
    runtime_storage,
    monkeypatch,
):
    context = RunContext.create("pilot", run_id="artifact-open-race")
    run_root = _run_root(runtime_storage, context)
    artifact_dir = run_root / "results"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "result.bin"
    trusted = b"trusted scientific result"
    artifact_path.write_bytes(trusted)

    outside_dir = runtime_storage / "attacker-controlled"
    outside_dir.mkdir()
    (outside_dir / artifact_path.name).write_bytes(b"attacker-controlled value")
    displaced_dir = run_root / "results-before-race"
    original_open = storage_layout.os.open
    swapped = False

    def racing_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == artifact_path.name and dir_fd is not None and not swapped:
            swapped = True
            artifact_dir.rename(displaced_dir)
            artifact_dir.symlink_to(outside_dir, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(storage_layout.os, "open", racing_open)
    artifact = context.register_artifact(
        "results/result.bin",
        artifact_id="race-safe-result",
        artifact_type="binary_result",
        mime_type="application/octet-stream",
    )

    assert swapped is True
    assert artifact.sha256 == hashlib.sha256(trusted).hexdigest()
    assert artifact.size_bytes == len(trusted)
    with pytest.raises(ValueError, match="symlink"):
        context.verify_artifact(artifact.artifact_id)


def test_handoff_round_trip_preserves_structured_context(runtime_storage):
    context = RunContext.create(
        "pilot",
        run_id="run-handoff",
        trace_id="trace-root",
    )
    context.add_task(
        TaskRecord(
            task_id="report",
            role="report_generator",
            profile="reporting",
            step="Generate evidence report",
        )
    )
    envelope = HandoffEnvelope.create(
        run_id="run-handoff",
        workflow_slug="pilot",
        task_id="report",
        sender_role="gtm_agent",
        receiver_role="report_generator",
        objective="Build a report from validated GTM artifacts",
        trace_id="trace-root",
        constraints=("Do not infer missing activities",),
        required_capabilities=("reporting",),
        expected_output_artifacts=("html_report",),
        expected_output_schema={"type": "object"},
        context_summary="GTM fitting completed successfully.",
        budget={
            "max_tokens": 1_000,
            "max_tool_calls": 3,
            "timeout_seconds": 60,
        },
        created_at="2999-01-01T00:00:00+00:00",
    )

    recorded = context.record_handoff(envelope)
    loaded = RunContext.load("run-handoff")

    assert loaded.run is not None
    assert loaded.run.handoffs[0].to_dict() == recorded.to_dict()
    assert loaded.run.handoffs[0].schema_version == 2
    assert recorded.created_at != "2999-01-01T00:00:00+00:00"
    external = HandoffEnvelope.from_mapping(envelope.to_dict())
    assert external.created_at != envelope.created_at
    with pytest.raises(ValueError, match="forbidden private/history"):
        HandoffEnvelope.from_mapping({**envelope.to_dict(), "private_reasoning": "hidden"})
    with pytest.raises(ValueError, match="forbidden private/history"):
        HandoffEnvelope.from_mapping(
            {
                **envelope.to_dict(),
                "expected_output_schema": {
                    "type": "object",
                    "properties": {"chain-of-thought": {"type": "array"}},
                },
            }
        )
    with pytest.raises(ValueError, match="forbidden private/history"):
        HandoffEnvelope.from_dict(
            {
                **envelope.to_dict(),
                "budget": {
                    "max_tool_calls": 3,
                    "messages": ["secret"],
                },
            }
        )
    with pytest.raises(ValueError, match="forbidden private/history"):
        replace(
            envelope,
            expected_output_schema={
                "type": "object",
                "properties": {"history": {"type": "array"}},
            },
        )


def test_catalog_task_roles_profiles_and_dependencies_are_enforced(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="catalog-task-guard",
    )
    context.transition_run(RunStatus.RUNNING)

    with pytest.raises(ValueError, match="requires role 'report_generator'"):
        context.add_task(
            TaskRecord(
                task_id="report",
                role="gtm_agent",
                profile="reporting",
                step="Write report",
            )
        )
    with pytest.raises(ValueError, match="requires profile 'reporting'"):
        context.add_task(
            TaskRecord(
                task_id="report",
                role="report_generator",
                profile="gtm-analysis",
                step="Write report",
            )
        )
    with pytest.raises(ValueError, match="is not declared"):
        context.add_task(
            TaskRecord(
                task_id="uncontracted",
                role="gtm_agent",
                profile="gtm-analysis",
                step="Bypass the declared task DAG",
            )
        )

    context.add_task(
        TaskRecord(
            task_id="report",
            role="report_generator",
            profile="reporting",
            step="Write report",
        )
    )
    with pytest.raises(InvalidTransitionError, match="dependencies complete"):
        context.transition_task("report", TaskStatus.RUNNING)


def test_task_creation_cannot_bypass_lifecycle(runtime_storage):
    context = RunContext.create("pilot", run_id="task-creation-guard")

    with pytest.raises(ValueError, match="pending"):
        context.add_task(
            TaskRecord(
                task_id="precompleted",
                role="gtm_agent",
                profile="gtm-analysis",
                step="Pretend to finish",
                status=TaskStatus.COMPLETED,
            )
        )
    with pytest.raises(ValueError, match="attempts, outputs"):
        context.add_task(
            TaskRecord(
                task_id="prepopulated",
                role="gtm_agent",
                profile="gtm-analysis",
                step="Pretend to produce",
                output_artifact_ids=["not-registered"],
            )
        )


def test_task_cannot_start_before_the_workflow_is_running(runtime_storage):
    context = RunContext.create("pilot", run_id="task-run-state-guard")
    context.add_task(
        TaskRecord(
            task_id="analysis",
            role="gtm_agent",
            profile="gtm-analysis",
            step="Analyze",
        )
    )

    with pytest.raises(InvalidTransitionError, match="workflow status is 'submitted'"):
        context.transition_task("analysis", TaskStatus.RUNNING)

    context.transition_run(RunStatus.RUNNING)
    assert context.transition_task("analysis", TaskStatus.RUNNING).attempts == 1


def test_catalog_task_completion_requires_its_registered_outputs(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="catalog-task-outputs",
    )
    context.transition_run(RunStatus.RUNNING)
    _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    _record_catalog_handoff(context, task_id="chembl-preflight")
    context.transition_task("chembl-preflight", TaskStatus.RUNNING)

    with pytest.raises(InvalidTransitionError, match="retrieval_plan"):
        context.transition_task("chembl-preflight", TaskStatus.COMPLETED)

    plan_path = "workflows/catalog-task-outputs/plans/retrieval.json"
    with S3.open(plan_path, "w") as handle:
        json.dump({"can_proceed": True}, handle)
    context.register_artifact(
        plan_path,
        artifact_type="retrieval_plan",
        mime_type="application/json",
        producer_task_id="chembl-preflight",
        active_task_id="chembl-preflight",
        producer_tool="chembl_prepare_retrieval",
    )

    assert (
        context.transition_task("chembl-preflight", TaskStatus.COMPLETED).status
        is TaskStatus.COMPLETED
    )


def test_catalog_task_requires_durable_root_input_binding(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="catalog-root-input",
    )
    context.transition_run(RunStatus.RUNNING)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )

    with pytest.raises(InvalidTransitionError, match="workflow input retrieval_request"):
        context.transition_task("chembl-preflight", TaskStatus.RUNNING)

    artifact = _register_retrieval_request(context)
    assert context.run.workflow_inputs == {"retrieval_request": artifact.artifact_id}
    _record_catalog_handoff(context, task_id="chembl-preflight")
    assert context.transition_task("chembl-preflight", TaskStatus.RUNNING).attempts == 1

    loaded = RunContext.load("catalog-root-input")
    assert loaded.run.workflow_inputs == {"retrieval_request": artifact.artifact_id}


def test_handoff_receiver_must_match_the_task_role(runtime_storage):
    context = RunContext.create("pilot", run_id="handoff-role-guard")
    context.add_task(
        TaskRecord(
            task_id="analysis",
            role="gtm_agent",
            profile="gtm-analysis",
            step="Analyze",
        )
    )
    envelope = HandoffEnvelope.create(
        run_id="handoff-role-guard",
        workflow_slug="pilot",
        task_id="analysis",
        sender_role="supervisor",
        receiver_role="report_generator",
        objective="Analyze the registered dataset",
    )

    with pytest.raises(ValueError, match="does not match task role"):
        context.record_handoff(envelope)


def test_handoff_rejects_invalid_execution_budget(runtime_storage):
    with pytest.raises(ValueError, match="max_tool_calls must be a positive"):
        HandoffEnvelope.create(
            run_id="budget-run",
            workflow_slug="pilot",
            task_id="analysis",
            sender_role="supervisor",
            receiver_role="gtm_agent",
            objective="Analyze",
            budget={"max_tool_calls": 0},
        )


def test_replay_rebuilds_deleted_snapshots_with_tasks_and_artifacts(runtime_storage):
    context = RunContext.create("pilot", run_id="run-replay", trace_id="trace-replay")
    context.transition_run(RunStatus.RUNNING)
    context.add_task(
        TaskRecord(
            task_id="build-map",
            role="gtm_agent",
            profile="gtm-analysis",
            step="Build GTM map",
        )
    )
    context.transition_task("build-map", TaskStatus.RUNNING)
    artifact_path = "workflows/run-replay/01_chemical_space/models/gtm.bin"
    with S3.open(artifact_path, "wb") as handle:
        handle.write(b"model-bytes")
    context.register_artifact(
        artifact_path,
        artifact_id="gtm-model",
        artifact_type="gtm_model",
        mime_type="application/octet-stream",
        producer_task_id="build-map",
        producer_tool="fit_gtm",
    )
    context.transition_task("build-map", TaskStatus.COMPLETED)
    expected_manifest = context.manifest_payload()
    expected_index = context.artifact_index_payload()

    root = _run_root(runtime_storage, context)
    (root / "manifest.json").unlink()
    (root / "artifacts" / "index.json").unlink()
    assert not (root / "manifest.json").exists()

    rebuilt = context.rebuild_snapshots()

    assert rebuilt.tasks["build-map"].status is TaskStatus.COMPLETED
    assert rebuilt.artifacts["gtm-model"].artifact_type == "gtm_model"
    assert json.loads((root / "manifest.json").read_text()) == expected_manifest
    assert json.loads((root / "artifacts" / "index.json").read_text()) == expected_index
    assert RunStore().events("run-replay")[-1]["event_type"] == "task_status_changed"


def test_catalog_contracts_prevent_false_completion(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="catalog-contract-run",
    )
    context.transition_run(RunStatus.RUNNING)

    completed = context.complete(
        required_artifact_types=[],
        required_task_ids=[],
    )

    assert completed.status is RunStatus.PARTIAL
    reason = str(context.events[-1].payload["reason"])
    assert "incomplete required tasks" in reason
    assert "missing required workflow input artifacts: retrieval_request" in reason
    assert "missing required artifact types" in reason
    assert "html_report_path" in reason


def test_direct_completed_transition_cannot_bypass_catalog_contracts(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="direct-completion-guard",
    )
    context.transition_run(RunStatus.RUNNING)

    with pytest.raises(InvalidTransitionError, match="contracts are satisfied"):
        context.transition_run(RunStatus.COMPLETED)


def test_terminal_runs_reject_scientific_state_mutation(runtime_storage):
    context = RunContext.create("pilot", run_id="terminal-mutation-guard")
    assert context.complete().status is RunStatus.COMPLETED

    with pytest.raises(InvalidTransitionError, match="terminal status"):
        context.add_task(
            TaskRecord(
                task_id="late-task",
                role="gtm_agent",
                profile="gtm-analysis",
                step="Mutate a completed run",
            )
        )

    artifact_path = "workflows/terminal-mutation-guard/results/late.txt"
    with S3.open(artifact_path, "w") as handle:
        handle.write("late")
    with pytest.raises(InvalidTransitionError, match="terminal status"):
        context.register_artifact(
            artifact_path,
            artifact_type="late_result",
            mime_type="text/plain",
        )


def test_completion_request_is_safe_and_idempotent_from_submitted_state(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="submitted-completion-run",
    )

    first = context.complete()
    second = context.complete()

    assert first.status is RunStatus.PARTIAL
    assert second.status is RunStatus.PARTIAL
    assert len([event for event in context.events if event.event_type == "run_status_changed"]) == 2


def test_completion_reverifies_registered_artifacts(runtime_storage):
    context = RunContext.create("pilot", run_id="completion-integrity-run")
    artifact_path = "workflows/completion-integrity-run/results/value.txt"
    with S3.open(artifact_path, "w") as handle:
        handle.write("original")
    context.register_artifact(
        artifact_path,
        artifact_type="result",
        mime_type="text/plain",
    )
    with S3.open(artifact_path, "w") as handle:
        handle.write("tampered")

    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        context.complete()


def test_concurrent_observation_events_keep_a_contiguous_sequence(runtime_storage):
    context = RunContext.create("pilot", run_id="concurrent-run")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(
            pool.map(
                lambda index: context.append_event("observation_recorded", {"index": index}),
                range(24),
            )
        )

    loaded = RunContext.load("concurrent-run")
    assert [event.sequence for event in loaded.events] == list(range(1, 26))


def test_stale_contexts_reserve_event_sequences_without_corruption(runtime_storage):
    first = RunContext.create("pilot", run_id="stale-writers")
    second = RunContext.load("stale-writers")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda item: item[0].append_event(
                    "observation_recorded",
                    {"writer": item[1]},
                ),
                ((first, "first"), (second, "second")),
            )
        )

    loaded = RunContext.load("stale-writers")
    assert [event.sequence for event in loaded.events] == [1, 2, 3]
    assert {event.payload.get("writer") for event in loaded.events[1:]} == {
        "first",
        "second",
    }


def test_snapshot_failure_does_not_report_a_committed_event_as_failed(
    runtime_storage,
    monkeypatch,
    caplog,
):
    context = RunContext.create("pilot", run_id="snapshot-failure")
    manifest_path = _run_root(runtime_storage, context) / "manifest.json"
    assert json.loads(manifest_path.read_text())["status"] == RunStatus.SUBMITTED.value

    def fail_snapshot_refresh():
        raise OSError("simulated replaceable snapshot failure")

    monkeypatch.setattr(context, "_write_snapshots", fail_snapshot_refresh)
    updated = context.transition_run(RunStatus.PLANNING)

    assert updated.status is RunStatus.PLANNING
    assert context.events[-1].sequence == 2
    assert "derived snapshot refresh failed" in caplog.text
    # The replaceable snapshot is stale, but replay sees the committed event.
    assert json.loads(manifest_path.read_text())["status"] == RunStatus.SUBMITTED.value
    loaded = RunContext.load("snapshot-failure")
    assert loaded.run.status is RunStatus.PLANNING
    assert len(loaded.events) == 2


def test_duplicate_run_identity_is_rejected_without_corrupting_events(runtime_storage):
    RunContext.create("pilot", run_id="duplicate-run")

    with pytest.raises(WorkflowRuntimeError, match="already exists"):
        RunContext.create("pilot", run_id="duplicate-run")

    loaded = RunContext.load("duplicate-run")
    assert len(loaded.events) == 1


def test_replay_rejects_tampered_dependency_contract_snapshot(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="tampered-dependency-contract",
    )
    event_path = next(_run_root(runtime_storage, context).glob("events/*.jsonl"))
    event = json.loads(event_path.read_text(encoding="utf-8"))
    dependencies = event["payload"]["run"]["workflow_contract"]["dependency_contracts"]
    dependencies[0]["workflow_md"] += "\nTampered after run creation."
    event_path.write_text(json.dumps(event) + "\n", encoding="utf-8")

    with pytest.raises(EventReplayError, match="could not apply event"):
        RunContext.load("tampered-dependency-contract")


def test_catalog_output_requires_authorized_running_producer_and_replay_enforces_it(
    runtime_storage,
):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="catalog-producer-authorization",
    )
    context.transition_run(RunStatus.RUNNING)
    _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    plan_path = "plans/retrieval.json"
    with S3.open(context.layout.artifact_rel_path(plan_path), "w") as handle:
        json.dump({"can_proceed": True}, handle)

    with pytest.raises(ValueError, match="does not match the active task None"):
        context.register_artifact(
            plan_path,
            artifact_type="retrieval_plan",
            mime_type="application/json",
            producer_task_id="chembl-preflight",
        )
    with pytest.raises(ValueError, match="must be an active RUNNING catalog task"):
        context.register_artifact(
            plan_path,
            artifact_type="retrieval_plan",
            mime_type="application/json",
            producer_task_id="chembl-preflight",
            active_task_id="chembl-preflight",
        )

    _record_catalog_handoff(context, task_id="chembl-preflight")
    context.transition_task("chembl-preflight", TaskStatus.RUNNING)
    with pytest.raises(ValueError, match="does not match the active task 'chembl-retrieval'"):
        context.register_artifact(
            plan_path,
            artifact_type="retrieval_plan",
            mime_type="application/json",
            producer_task_id="chembl-preflight",
            active_task_id="chembl-retrieval",
        )
    artifact = context.register_artifact(
        plan_path,
        artifact_type="retrieval_plan",
        mime_type="application/json",
        producer_task_id="chembl-preflight",
        active_task_id="chembl-preflight",
    )
    context.transition_task("chembl-preflight", TaskStatus.COMPLETED)
    with pytest.raises(ValueError, match="must be an active RUNNING catalog task"):
        context.register_artifact(
            plan_path,
            artifact_type="retrieval_plan",
            mime_type="application/json",
            artifact_id=artifact.artifact_id,
            producer_task_id="chembl-preflight",
            active_task_id="chembl-preflight",
        )

    running_event_path = next(
        path
        for path in _run_root(runtime_storage, context).glob("events/*.jsonl")
        if json.loads(path.read_text())["event_type"] == "task_status_changed"
        and json.loads(path.read_text())["payload"]["status"] == "running"
    )
    running_event = json.loads(running_event_path.read_text())
    running_event["payload"]["status"] = "cancelled"
    running_event_path.write_text(json.dumps(running_event) + "\n", encoding="utf-8")

    with pytest.raises(EventReplayError, match="registered output.*not 'running'"):
        RunContext.load("catalog-producer-authorization")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "required_capabilities",
            ("gtm_optimization",),
            "required capabilities must match the pinned task contract",
        ),
        (
            "expected_output_artifacts",
            ("html_report_path",),
            "expected output artifacts must match the pinned task contract",
        ),
        (
            "acceptance_criteria",
            ("Skip scientific validation.",),
            "acceptance criteria must match the pinned task contract",
        ),
    ],
)
def test_run_context_rejects_handoffs_that_drift_from_the_pinned_task_contract(
    runtime_storage,
    field,
    value,
    message,
):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id=f"handoff-drift-{field.replace('_', '-')}",
    )
    _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    envelope = _catalog_handoff_envelope(context, task_id="chembl-preflight")
    envelope = replace(envelope, **{field: value})
    event_count = len(context.events)

    with pytest.raises(ValueError, match=message):
        context.record_handoff(envelope)

    assert len(context.events) == event_count
    assert context.run.handoffs == []


def test_selected_task_inputs_are_persisted_and_disambiguate_same_type_artifacts(
    runtime_storage,
):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="selected-duplicate-input",
    )
    context.transition_run(RunStatus.RUNNING)
    request = _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    _record_catalog_handoff(context, task_id="chembl-preflight")
    context.transition_task("chembl-preflight", TaskStatus.RUNNING)
    plan_ids = []
    for index in (1, 2):
        plan_path = f"plans/retrieval-{index}.json"
        with S3.open(context.layout.artifact_rel_path(plan_path), "w") as handle:
            json.dump({"can_proceed": True, "index": index}, handle)
        plan_ids.append(
            context.register_artifact(
                plan_path,
                artifact_type="retrieval_plan",
                mime_type="application/json",
                artifact_id=f"retrieval-plan-{index}",
                producer_task_id="chembl-preflight",
                active_task_id="chembl-preflight",
            ).artifact_id
        )
    context.transition_task("chembl-preflight", TaskStatus.COMPLETED)
    context.add_task(
        TaskRecord(
            task_id="chembl-retrieval",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Retrieve compounds",
        )
    )
    selected = replace(
        _catalog_handoff_envelope(context, task_id="chembl-retrieval"),
        input_artifact_ids=(request.artifact_id, plan_ids[1]),
    )

    recorded = context.record_handoff(selected)

    assert recorded.input_artifact_ids == (request.artifact_id, plan_ids[1])
    assert context.run.tasks["chembl-retrieval"].input_artifact_ids == [
        request.artifact_id,
        plan_ids[1],
    ]
    assert context.transition_task("chembl-retrieval", TaskStatus.RUNNING).attempts == 1
    loaded = RunContext.load("selected-duplicate-input")
    assert loaded.run.tasks["chembl-retrieval"].input_artifact_ids == [
        request.artifact_id,
        plan_ids[1],
    ]


def test_task_input_checksums_are_verified_before_handoff_and_activation(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="task-input-integrity",
    )
    context.transition_run(RunStatus.RUNNING)
    request = _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    request_path = context.layout.artifact_rel_path(request.relative_path)
    with S3.open(request_path, "w") as handle:
        json.dump({"target": "tampered-before-handoff"}, handle)
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        _record_catalog_handoff(context, task_id="chembl-preflight")
    assert context.run.handoffs == []

    with S3.open(request_path, "w") as handle:
        json.dump({"target": "EGFR"}, handle)
    _record_catalog_handoff(context, task_id="chembl-preflight")
    with S3.open(request_path, "w") as handle:
        json.dump({"target": "tampered-before-activation"}, handle)
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        context.transition_task("chembl-preflight", TaskStatus.RUNNING)
    assert context.run.tasks["chembl-preflight"].status is TaskStatus.PENDING


def test_input_required_task_needs_a_fresh_handoff_before_resuming(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="input-required-fresh-handoff",
    )
    context.transition_run(RunStatus.RUNNING)
    _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    _record_catalog_handoff(context, task_id="chembl-preflight")
    context.transition_task("chembl-preflight", TaskStatus.RUNNING)
    context.transition_task("chembl-preflight", TaskStatus.INPUT_REQUIRED)

    with pytest.raises(InvalidTransitionError, match="fresh validated structured handoff"):
        context.transition_task("chembl-preflight", TaskStatus.RUNNING)

    fresh = _record_catalog_handoff(context, task_id="chembl-preflight")
    assert fresh.task_attempt == 1
    assert context.transition_task("chembl-preflight", TaskStatus.RUNNING).attempts == 2


def test_replay_rejects_a_tampered_pinned_handoff_contract(runtime_storage):
    context = RunContext.create(
        "chembl-to-gtm-report",
        run_id="tampered-handoff-contract",
    )
    _register_retrieval_request(context)
    context.add_task(
        TaskRecord(
            task_id="chembl-preflight",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Validate retrieval dimensions",
        )
    )
    _record_catalog_handoff(context, task_id="chembl-preflight")
    handoff_event_path = next(
        path
        for path in _run_root(runtime_storage, context).glob("events/*.jsonl")
        if json.loads(path.read_text())["event_type"] == "handoff_recorded"
    )
    handoff_event = json.loads(handoff_event_path.read_text())
    handoff_event["payload"]["handoff"]["required_capabilities"] = ["gtm_optimization"]
    handoff_event_path.write_text(json.dumps(handoff_event) + "\n", encoding="utf-8")

    with pytest.raises(EventReplayError, match="required capabilities must match"):
        RunContext.load("tampered-handoff-contract")


def test_replay_uses_object_store_safe_one_event_segments(runtime_storage, monkeypatch):
    memory = fsspec.filesystem("memory")
    store_root = "/chemspace/runtime-session"

    def store_open(cls, rel_path, mode="rb"):
        return memory.open(f"{store_root}/{str(rel_path).lstrip('/')}", mode)

    def store_path(cls, rel_path):
        return f"memory://{store_root.lstrip('/')}/{str(rel_path).lstrip('/')}"

    monkeypatch.setattr(S3, "open", classmethod(store_open))
    monkeypatch.setattr(S3, "open_atomic", classmethod(store_open))
    monkeypatch.setattr(S3, "path", classmethod(store_path))

    context = RunContext.create("pilot", run_id="object-run")
    context.transition_run(RunStatus.RUNNING)
    loaded = RunContext.load("object-run")

    event_objects = memory.glob(f"{store_root}/workflows/object-run/events/*.jsonl")
    assert len(event_objects) == 2
    assert all(len(memory.cat(path).splitlines()) == 1 for path in event_objects)
    assert loaded.run is not None
    assert loaded.run.status is RunStatus.RUNNING


def _run_root(tmp_path: Path, context: RunContext) -> Path:
    assert context.run is not None
    return (
        tmp_path / "data" / "sessions" / context.run.session_id / "workflows" / context.run.run_id
    )


def _register_retrieval_request(context: RunContext):
    path = f"workflows/{context.run.run_id}/inputs/retrieval-request.json"
    with S3.open(path, "w") as handle:
        json.dump({"target": "EGFR"}, handle)
    return context.register_artifact(
        path,
        artifact_type="retrieval_request",
        mime_type="application/json",
        artifact_id="retrieval-request",
        producer_tool="unit_test",
        trust="external",
    )


def _record_catalog_handoff(context: RunContext, *, task_id: str):
    return context.record_handoff(_catalog_handoff_envelope(context, task_id=task_id))


def _catalog_handoff_envelope(context: RunContext, *, task_id: str):
    task = context.run.tasks[task_id]
    contract = next(
        item for item in context.run.workflow_contract["tasks"] if item["task_id"] == task_id
    )
    selected_inputs = []
    for artifact_type in contract["input_artifacts"]:
        workflow_input_id = context.run.workflow_inputs.get(artifact_type)
        if workflow_input_id is not None:
            selected_inputs.append(workflow_input_id)
            continue
        matches = [
            artifact.artifact_id
            for artifact in context.run.artifacts.values()
            if artifact.artifact_type == artifact_type
        ]
        if len(matches) == 1:
            selected_inputs.append(matches[0])
    return HandoffEnvelope.create(
        run_id=context.run.run_id,
        workflow_slug=context.run.workflow_slug,
        task_id=task_id,
        sender_role="supervisor",
        receiver_role=task.role,
        objective=f"Execute {task_id}.",
        required_capabilities=contract["required_tools"],
        input_artifact_ids=selected_inputs,
        acceptance_criteria=contract["acceptance_criteria"],
        expected_output_artifacts=contract["output_artifacts"],
        budget={
            "max_tokens": 1_000,
            "max_tool_calls": 4,
            "timeout_seconds": 60,
        },
    )
