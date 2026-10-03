"""Toolkit writers keep working when a step is repeated under enforce mode.

Enforce mode makes writes create-only and registered outputs immutable, so a
writer with a deterministic file name must pick a new version on a re-run
instead of failing. Each canary runs one writer twice through the real Agno
execution binding and checks both calls succeed with distinct, registered
files.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import pandas as pd
import pytest
from agno.agent import Agent
from agno.tools import Toolkit
from agno.tools.function import FunctionCall

from cs_copilot.agents.execution_binding import attach_execution
from cs_copilot.storage import S3
from cs_copilot.tools.analysis.robustness_toolkit import RobustnessAnalysisToolkit
from cs_copilot.tools.chemistry import clean_dataset
from cs_copilot.tools.chemistry.peptide_designer_toolkit import _save_peptide_design_artifact
from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerToolkit
from cs_copilot.tools.chemography import gtm_operations
from cs_copilot.tools.databases.chembl import ChemblToolkit
from cs_copilot.tools.io import report_export
from cs_copilot.tools.io.session_memory import (
    save_candidate_set_artifact,
    save_candidate_set_dataset,
)
from cs_copilot.workflows import RunContext

TABLE = pd.DataFrame({"smiles": ["CCO", "CCN"], "value": [1.0, 2.0]})
FILTERING_SUMMARY = {"filtered_row_count": 1, "decision_counts": {"drop": 1}}


def _clean_dataset_outputs(state: Dict[str, Any]) -> list[str]:
    def path(name: str, *folders: str) -> str:
        return clean_dataset._chemical_space_artifact_path(name, *folders, session_state=state)

    return [
        clean_dataset._write_csv(TABLE, path("ligands_clean.csv", "datasets", "clean")),
        clean_dataset._write_parquet(
            TABLE, path("ligands_descriptors.parquet", "datasets", "descriptors")
        ),
        clean_dataset._write_report(
            {"clean_dataset_path": "ligands_clean.csv"},
            path("ligands_report.md", "standardization"),
        ),
    ]


def _chembl_outputs(state: Dict[str, Any]) -> list[str]:
    toolkit = ChemblToolkit()
    return [
        toolkit._save_filtered_rows(TABLE, query_slug="chembl234", session_state=state),
        toolkit._write_retrieval_filtering_only_report("chembl234", FILTERING_SUMMARY, state),
    ]


def _gtm_outputs(state: Dict[str, Any]) -> list[str]:
    state["ligands"] = TABLE
    state["ligand_map"] = {"weights": [1, 2, 3]}
    message = gtm_operations.save_gtm_and_dataset(
        "ligands", "ligand_map", SimpleNamespace(session_state=state)
    )
    return [part.split(": ", 1)[1] for part in message.split("; ")]


def _candidate_set_outputs(state: Dict[str, Any]) -> list[str]:
    rows = [{"smi": "CCO", "rank": 1, "candidate_set_id": "cset_1", "source": "design"}]
    return [
        save_candidate_set_dataset("cset_1", rows, session_state=state),
        save_candidate_set_artifact("cset_1", [{"smiles": "CCO"}], session_state=state),
    ]


def _peptide_outputs(state: Dict[str, Any]) -> list[str]:
    # A counter restored from an older snapshot must not reuse an existing id.
    state.pop("_peptide_design_run_counter", None)
    saved = _save_peptide_design_artifact(
        state, session_key="peptides", candidates=[{"sequence": "ACDE"}], metadata={}
    )
    return [saved["artifact_path"]]


def _report_outputs(state: Dict[str, Any]) -> list[str]:
    return [
        report_export._write_text_report(
            "# Report\n", report_export._report_rel_path("summary.md", "analysis", state)
        ),
        report_export._write_binary_report(
            b"%PDF-1.4\n", report_export._report_rel_path("summary.pdf", "analysis", state)
        ),
    ]


def _robustness_outputs(state: Dict[str, Any]) -> list[str]:
    run_root = f"workflows/{state['output_context']['run_id']}"
    return [
        RobustnessAnalysisToolkit().export_analysis_report(
            {"runs": [{"score": 0.9}]}, format="json", output_path=f"{run_root}/robustness"
        )
    ]


def _synplanner_outputs(state: Dict[str, Any]) -> list[str]:
    toolkit = SynPlannerToolkit()
    plan = {"query": "ethanol", "smiles": "CCO", "routes": [{"steps": []}]}
    toolkit._persist_route_artifacts(plan, agent=None, session_state=state)
    return [
        toolkit._persist_plan_artifact(plan, agent=None, session_state=state),
        plan["routes"][0]["route_json_path"],
    ]


WRITERS: dict[str, Callable[[Dict[str, Any]], list[str]]] = {
    "clean_dataset": _clean_dataset_outputs,
    "chembl": _chembl_outputs,
    "gtm": _gtm_outputs,
    "candidate_sets": _candidate_set_outputs,
    "peptide": _peptide_outputs,
    "report_export": _report_outputs,
    "robustness": _robustness_outputs,
    "synplanner": _synplanner_outputs,
}


class CanaryToolkit(Toolkit):
    def __init__(self) -> None:
        super().__init__("canaries")
        self.register(self.write_outputs)

    def write_outputs(self, writer: str, session_state: Optional[Dict[str, Any]] = None) -> str:
        """Run one toolkit writer.

        Args:
            writer: Name of the writer to run.
        """
        return json.dumps(WRITERS[writer](session_state))


@pytest.fixture
def chat(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix("sessions/canary-chat")
    state: dict[str, Any] = {}
    run = RunContext.create("agno-session", session_state=state, session_id="canary-chat")
    run.transition_run("running")
    toolkit = CanaryToolkit()
    member = Agent(name="member", tools=[toolkit], session_state=state, telemetry=False)
    member.agentic_role = "single_agent"
    team = SimpleNamespace(session_state=state, members=[member], tools=[], run_context=run)
    attach_execution(team, run_context=run, mode="enforce")
    return SimpleNamespace(run=run, toolkit=toolkit, member=member, state=state)


def _write(chat, writer: str) -> list[str]:
    function = chat.toolkit.functions["write_outputs"]
    function.process_entrypoint()
    function._agent = chat.member
    function._session_state = chat.state
    call = FunctionCall(function=function, arguments={"writer": writer}).execute()
    assert call.status == "success", call.error
    return json.loads(call.result)


@pytest.mark.parametrize("writer", sorted(WRITERS))
def test_repeating_a_step_writes_new_versions(chat, writer):
    first = _write(chat, writer)
    second = _write(chat, writer)

    keys = [S3.session_relative(path) for path in first + second]
    assert len(set(keys)) == len(first) * 2
    assert all(key.startswith(f"{chat.run.layout.run_root}/") for key in keys)
    assert all(S3.exists(key) for key in keys)
    registered = {
        chat.run.layout.artifact_rel_path(record.relative_path)
        for record in chat.run.refresh().artifacts.values()
    }
    assert set(keys) <= registered
