"""One durable workflow run per in-process chat."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cs_copilot.agents.execution_binding import PROCESS_OWNER, get_binding
from cs_copilot.agents.session_runs import (
    ensure_agno_session_run,
    finalize_agno_turn,
    reconcile_interrupted_work,
    restore_team_session_state,
    store_chat_upload,
)
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext, RunStatus, TaskRecord, TaskStatus


def _team(state=None):
    shared = state if state is not None else {"agentic_contracts": {}, "resource_profile": {}}
    member = SimpleNamespace(name="member", tools=[], session_state=shared)
    return SimpleNamespace(session_state=shared, members=[member], tools=[], run_context=None)


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/chat-1")
    return tmp_path


def test_first_message_creates_and_binds_the_chat_run(storage):
    team = _team()

    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")

    assert run.run.workflow_slug == "agno-session"
    assert run.run.status is RunStatus.RUNNING
    state = team.session_state
    assert state["output_context"]["run_id"] == run.run.run_id
    assert state["agentic_contracts"]["active_run"]["run_id"] == run.run.run_id
    assert team.run_context is run
    assert get_binding(team).run_context is run
    assert ensure_agno_session_run(team, session_id="chat-1", mode="observe") is run


def test_off_mode_creates_no_run(storage):
    team = _team()
    assert ensure_agno_session_run(team, session_id="chat-1", mode="off") is None
    assert "output_context" not in team.session_state


def test_a_resumed_chat_reuses_its_run(storage):
    first = _team()
    run = ensure_agno_session_run(first, session_id="chat-1", mode="observe")

    resumed = _team()
    restore_team_session_state(resumed, dict(first.session_state))
    again = ensure_agno_session_run(resumed, session_id="chat-1", mode="observe")

    assert again.run.run_id == run.run.run_id
    assert resumed.session_state["output_context"]["run_id"] == run.run.run_id


def test_legacy_output_contexts_are_kept_but_replaced(storage):
    legacy = {
        "session_id": "chat-1",
        "run_id": "chemical_space-old",
        "workflow_slug": "chemical_space",
    }
    team = _team({"agentic_contracts": {}, "output_context": dict(legacy)})

    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")

    assert run.run.run_id != "chemical_space-old"
    assert team.session_state["legacy_output_contexts"] == [legacy]


def test_restore_updates_the_shared_dict_in_place(storage):
    team = _team()
    shared = team.session_state
    team.session_state["resource_profile"] = {"gpu": False}

    restore_team_session_state(team, {"uploaded_files": {"a.csv": "p"}, "resource_profile": {}})

    assert team.session_state is shared
    assert team.members[0].session_state is shared
    assert shared["uploaded_files"] == {"a.csv": "p"}
    assert shared["resource_profile"] == {"gpu": False}


def test_interrupted_spans_and_tasks_are_reconciled(storage):
    team = _team()
    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")
    for task_id in ("t1", "t2"):
        run.add_task(
            TaskRecord(task_id=task_id, role="gtm_agent", profile="gtm-analysis", step="Map")
        )
        run.transition_task(task_id, TaskStatus.RUNNING)
    attempt = run.run.tasks["t1"].attempts
    for span_id, owner, task_id in (
        ("orphan", {"pid": -1}, "t2"),
        ("mine", PROCESS_OWNER, "t1"),
    ):
        run.append_event(
            "tool_progress",
            {
                "runtime": "agno",
                "tool_name": "agno.GTMToolkit.gtm_optimization",
                "span_id": span_id,
                "stage": "started",
                "owner": owner,
                "task_id": task_id,
                "task_attempt": attempt,
            },
        )

    abandoned = reconcile_interrupted_work(run)

    assert abandoned == ["orphan"]
    assert "mine" in " ".join(run.pending_tool_invocations())
    tasks = run.refresh().tasks
    assert tasks["t1"].status is TaskStatus.RUNNING  # its span belongs to this process
    assert tasks["t2"].status is TaskStatus.FAILED  # interrupted


def test_turn_finalization_completes_delegated_tasks(storage):
    team = _team()
    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")
    run.add_task(TaskRecord(task_id="t1", role="gtm_agent", profile="gtm-analysis", step="Map"))
    run.transition_task("t1", TaskStatus.RUNNING)
    get_binding(team).note_task_started("gtm_agent", "t1")

    finalize_agno_turn(team)

    assert run.refresh().tasks["t1"].status is TaskStatus.COMPLETED
    assert get_binding(team).active_tasks == {}


def test_failed_turns_fail_their_tasks(storage):
    team = _team()
    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")
    run.add_task(TaskRecord(task_id="t2", role="gtm_agent", profile="gtm-analysis", step="Map"))
    run.transition_task("t2", TaskStatus.RUNNING)
    get_binding(team).note_task_started("gtm_agent", "t2")

    finalize_agno_turn(team, failed=True)

    assert run.refresh().tasks["t2"].status is TaskStatus.FAILED


def test_uploads_are_stored_in_the_run_as_untrusted_artifacts(storage):
    team = _team()
    run = ensure_agno_session_run(team, session_id="chat-1", mode="observe")

    path, artifact_id = store_chat_upload(
        run, filename="../ligands set.csv", content=b"smiles\nCCO\n"
    )
    again_path, again_id = store_chat_upload(
        run, filename="ligands set.csv", content=b"smiles\nCCO\n"
    )

    record = run.run.artifacts[artifact_id]
    assert (again_path, again_id) == (path, artifact_id)
    assert record.relative_path.startswith("inputs/uploads/ligands_set-")
    assert record.artifact_type == "user_upload"
    assert record.trust.value == "untrusted"
    with S3.open(path, "rb") as handle:
        assert handle.read() == b"smiles\nCCO\n"
    reloaded = RunContext.load(run.run.run_id, session_id="chat-1", verify_artifacts=True)
    assert artifact_id in reloaded.run.artifacts


def test_the_production_team_is_routed_through_the_kernel(storage, monkeypatch):
    from agno.models.base import Model

    from cs_copilot.agents import teams
    from cs_copilot.tools.chemistry.autoencoder_toolkit import AutoencoderToolkit
    from cs_copilot.tools.chemistry.peptide_designer_toolkit import PeptideDesignerToolkit

    class _ConstructionModel(Model):
        def invoke(self, *args, **kwargs):
            raise NotImplementedError

        async def ainvoke(self, *args, **kwargs):
            raise NotImplementedError

        def invoke_stream(self, *args, **kwargs):
            raise NotImplementedError
            yield

        async def ainvoke_stream(self, *args, **kwargs):
            raise NotImplementedError
            yield

        def _parse_provider_response(self, response, **kwargs):
            raise NotImplementedError

        def _parse_provider_response_delta(self, response):
            raise NotImplementedError

    for cls in (AutoencoderToolkit, PeptideDesignerToolkit):
        for name in ("_ensure_model_exists", "_load_model"):
            if hasattr(cls, name):
                monkeypatch.setattr(cls, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(teams, "analyze_resources", lambda: {"cpu": "test"})
    team = teams.get_cs_copilot_agent_team(
        _ConstructionModel(id="construction"),
        enable_memory=False,
        enable_mlflow_tracking=False,
        execution_mode="observe",
    )

    run = ensure_agno_session_run(team, session_id="chat-1")

    assert team.run_context is run
    state = team.session_state
    assert state["agentic_contracts"]["active_run"]["run_id"] == run.run.run_id
    assert all(member.session_state is state for member in team.members)
    wrapped = unwrapped = 0
    for entity in (team, *team.members):
        for tool in entity.tools:
            functions = getattr(tool, "functions", None)
            entrypoints = (
                [function.entrypoint for function in functions.values()]
                if isinstance(functions, dict)
                else [tool]
            )
            for entrypoint in entrypoints:
                if getattr(entrypoint, "__cs_execution_binding__", None) is get_binding(team):
                    wrapped += 1
                else:
                    unwrapped += 1
    assert wrapped > 100 and unwrapped == 0
