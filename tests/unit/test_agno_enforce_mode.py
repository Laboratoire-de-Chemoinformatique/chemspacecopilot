"""Enforce mode: the full kernel for in-process Agno tool calls."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import pytest
from agno.agent import Agent
from agno.tools import Toolkit
from agno.tools.function import FunctionCall

from cs_copilot.agents.execution_binding import _in_process_contract, attach_execution
from cs_copilot.agents.session_runs import adopt_session_uploads, ensure_agno_session_run
from cs_copilot.execution.spec import ToolSpec
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext


class OutputToolkit(Toolkit):
    def __init__(self) -> None:
        super().__init__("outputs")
        self.register(self.save_table)
        self.register(self.save_then_fail)
        self.register(self.remember)

    def save_table(self, output_path: str) -> str:
        """Save a table.

        Args:
            output_path: Where to save it.
        """
        with S3.open(output_path, "w") as handle:
            handle.write("x\n1\n")
        return output_path

    def save_then_fail(self, output_path: str) -> str:
        """Save a table, then fail.

        Args:
            output_path: Where to save it.
        """
        with S3.open(output_path, "w") as handle:
            handle.write("partial")
        raise RuntimeError("analysis failed after writing")

    def remember(self, key: str, session_state: Optional[Dict[str, Any]] = None) -> str:
        """Store a key in session state.

        Args:
            key: Key to store.
        """
        session_state["remembered"] = key
        return key


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/enforce-chat")
    state: dict[str, Any] = {}
    run = RunContext.create("agno-session", session_state=state, session_id="enforce-chat")
    run.transition_run("running")
    toolkit = OutputToolkit()
    member = Agent(name="member", tools=[toolkit], session_state=state, telemetry=False)
    member.agentic_role = "report_generator"
    team = SimpleNamespace(session_state=state, members=[member], tools=[], run_context=run)
    attach_execution(team, run_context=run, mode="enforce")
    return SimpleNamespace(run=run, toolkit=toolkit, member=member, state=state)


def _call(chat, name: str, **arguments: Any):
    function = chat.toolkit.functions[name]
    function.process_entrypoint()
    function._agent = chat.member
    function._session_state = chat.state
    return FunctionCall(function=function, arguments=arguments).execute()


def test_outputs_are_rewritten_into_the_run_and_registered(chat):
    result = _call(chat, "save_table", output_path="results.csv")

    assert result.status == "success"
    run_root = chat.run.layout.run_root
    assert result.result == f"{run_root}/results.csv"
    [artifact] = chat.run.refresh().artifacts.values()
    assert artifact.relative_path == "results.csv"
    assert artifact.producer_tool == "agno.OutputToolkit.save_table"
    assert artifact.provenance["result_field"] == "published_write"


def test_rewriting_a_registered_output_is_denied_as_a_tool_error(chat):
    _call(chat, "save_table", output_path="results.csv")

    second = _call(chat, "save_table", output_path="results.csv")

    assert second.status == "failure"
    assert "immutable registered workflow artifact" in (second.error or "")
    assert len(chat.run.refresh().artifacts) == 1


def test_a_failing_call_rolls_back_its_files(chat):
    result = _call(chat, "save_then_fail", output_path="partial.csv")

    assert result.status == "failure"
    assert "analysis failed after writing" in (result.error or "")
    assert not S3.exists(f"{chat.run.layout.run_root}/partial.csv")
    assert chat.run.refresh().artifacts == {}


def test_enforce_passes_the_live_session_state(chat):
    _call(chat, "remember", key="kept")

    assert chat.state["remembered"] == "kept"


def test_forces_apply_only_to_declared_parameters():
    def design(seed: str, _source_tool: str = "default") -> str:
        return seed

    contract = ToolSpec(
        mcp_name="mol_design_molecules",
        toolkit_factory=object,
        method="design",
        summary="Design.",
        forces={"_source_tool": "design_molecules", "undeclared": 1},
        max_retries=1,
        idempotent=True,
        timeout_s=5,
    )

    adapted = _in_process_contract(contract, design)

    assert adapted.forces == {"_source_tool": "design_molecules"}
    assert (adapted.timeout_s, adapted.max_retries, adapted.run_in_worker_process) == (
        None,
        0,
        False,
    )


def test_enforce_refuses_catalog_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/catalog-chat")
    run = RunContext.create("chembl-to-gtm-report", session_id="catalog-chat")
    team = SimpleNamespace(session_state={}, members=[], tools=[], run_context=run)

    with pytest.raises(ValueError, match="ad-hoc chat runs only"):
        attach_execution(team, run_context=run, mode="enforce")


def test_legacy_uploads_are_adopted_into_the_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/legacy-chat")
    with S3.open("ligands.csv", "w") as handle:
        handle.write("smiles\nCCO\n")
    state = {"uploaded_files": {"ligands.csv": S3.path("ligands.csv")}, "agentic_contracts": {}}
    team = SimpleNamespace(session_state=state, members=[], tools=[], run_context=None)

    run = ensure_agno_session_run(team, session_id="legacy-chat", mode="enforce")

    artifact_id = state["uploaded_artifacts"]["ligands.csv"]
    assert run.run.artifacts[artifact_id].artifact_type == "user_upload"
    assert state["uploaded_files"]["ligands.csv"].startswith(S3.path(run.layout.run_root))
    assert adopt_session_uploads(run, state) == []  # idempotent
