"""MCP bootstrap tests for v2 run identity and profile-safe resume."""

from __future__ import annotations

from dataclasses import replace

import pytest

import cs_copilot.workflows as workflows_module
from cs_copilot.mcp.profiles import MCPProfileError
from cs_copilot.mcp.session import BootstrapConfig, bootstrap
from cs_copilot.storage import S3
from cs_copilot.workflows import TaskRecord, get_workflow


@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("USE_S3", "false")
    old_prefix = S3.current_prefix()
    try:
        yield
    finally:
        S3.set_session_prefix(old_prefix)


def test_resume_preserves_run_identity(isolated_storage):
    first = bootstrap(
        BootstrapConfig(
            session_id="resume-session",
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )
    run_id = first.run_context.run.run_id

    resumed = bootstrap(
        BootstrapConfig(
            session_id="resume-session",
            run_id=run_id,
            profile="standard",
        )
    )

    assert resumed.run_context.run.run_id == run_id
    assert resumed.run_context.run.workflow_slug == "chembl-target-retrieval"
    assert resumed.session_state["output_context"]["run_id"] == run_id


def test_resume_restores_one_unambiguous_running_task(isolated_storage):
    first = bootstrap(
        BootstrapConfig(
            session_id="resume-active-task",
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )
    first.run_context.transition_run("running")
    first.run_context.add_task(
        TaskRecord(
            task_id="retrieve",
            role="chembl_downloader",
            profile="chembl-retrieval",
            step="Retrieve compounds",
        )
    )
    first.run_context.transition_task("retrieve", "running")

    resumed = bootstrap(
        BootstrapConfig(
            session_id="resume-active-task",
            run_id=first.run_context.run.run_id,
            profile="chembl-retrieval",
        )
    )

    assert resumed.session_state["active_task_id"] == "retrieve"
    assert resumed.session_state["active_role"] == "chembl_downloader"
    assert resumed.session_state["active_profile"] == "chembl-retrieval"


def test_resume_leaves_parallel_running_tasks_unselected(isolated_storage):
    first = bootstrap(
        BootstrapConfig(
            session_id="resume-parallel-tasks",
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )
    first.run_context.transition_run("running")
    for task_id in ("retrieve-a", "retrieve-b"):
        first.run_context.add_task(
            TaskRecord(
                task_id=task_id,
                role="chembl_downloader",
                profile="chembl-retrieval",
                step=task_id,
            )
        )
        first.run_context.transition_task(task_id, "running")

    resumed = bootstrap(
        BootstrapConfig(
            session_id="resume-parallel-tasks",
            run_id=first.run_context.run.run_id,
            profile="chembl-retrieval",
        )
    )

    assert "active_task_id" not in resumed.session_state
    assert "active_role" not in resumed.session_state
    assert "active_profile" not in resumed.session_state


def test_resume_rejects_profile_incompatible_with_stored_workflow(isolated_storage):
    first = bootstrap(
        BootstrapConfig(
            session_id="resume-profile-session",
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )

    with pytest.raises(MCPProfileError, match="does not permit"):
        bootstrap(
            BootstrapConfig(
                session_id="resume-profile-session",
                run_id=first.run_context.run.run_id,
                profile="reporting",
            )
        )


def test_resume_uses_pinned_contract_when_live_catalog_changes(
    isolated_storage,
    monkeypatch,
):
    first = bootstrap(
        BootstrapConfig(
            session_id="resume-pinned-session",
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )
    run_id = first.run_context.run.run_id
    pinned_hash = first.run_context.run.workflow_contract["contract_sha256"]
    live = get_workflow("chembl-target-retrieval")
    mutated = replace(
        live,
        profiles=("reporting",),
        preflight_tools=("removed_preflight_tool",),
        required_tools=("removed_required_tool",),
        tasks=(),
    )
    monkeypatch.setattr(workflows_module, "get_workflow", lambda _slug: mutated)

    resumed = bootstrap(
        BootstrapConfig(
            session_id="resume-pinned-session",
            run_id=run_id,
            workflow_slug="chembl-target-retrieval",
            profile="chembl-retrieval",
        )
    )

    assert resumed.run_context.run.run_id == run_id
    assert resumed.run_context.run.workflow_contract["contract_sha256"] == pinned_hash
