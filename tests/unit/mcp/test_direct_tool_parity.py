"""Behavior tests for direct MCP parity facades."""

from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import build_tool
from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.storage import S3, ensure_output_context


def _spec(name: str):
    return next(spec for spec in all_specs() if spec.mcp_name == name)


def test_skill_tools_work_directly():
    skills = _spec("skill_list").toolkit_factory()

    listed = skills.list()
    searched = skills.search("molecular design")
    fetched = skills.fetch("molecular-design")

    assert any(skill["slug"] == "molecular-design" for skill in listed)
    assert searched[0]["slug"] == "molecular-design"
    assert fetched["slug"] == "molecular-design"
    assert "#" in fetched["skill_md"]


def test_workflow_tools_work_directly():
    workflows = _spec("workflow_list").toolkit_factory()

    listed = workflows.list()
    searched = workflows.search("chembl gtm report")
    fetched = workflows.fetch("chembl-to-gtm-report")

    assert any(workflow["slug"] == "chembl-to-gtm-report" for workflow in listed)
    assert searched[0]["slug"] == "chembl-to-gtm-report"
    assert fetched["slug"] == "chembl-to-gtm-report"
    assert "report_save_rich" in fetched["workflow_md"]


def test_molecular_validate_and_rank_work_without_model_access():
    tools = _spec("mol_validate_design_candidates").toolkit_factory()

    validated = tools.validate_design_candidates(["CCO", "c1ccccc1", "bad"])
    ranked = tools.rank_design_candidates(validated, seed_smiles="CCO")

    assert ranked[0]["smiles"] == "CCO"
    assert ranked[0]["properties"]["seed_tanimoto"] == 1.0
    assert ranked[-1]["valid"] is False


def test_peptide_validate_and_rank_work_without_model_access():
    tools = _spec("peptide_validate_design_candidates").toolkit_factory()

    validated = tools.validate_design_candidates(["ACD", "A C E", "B"])
    ranked = tools.rank_design_candidates(validated, seed_sequence="A C D")

    assert ranked[0]["sequence"] == "A C D"
    assert ranked[0]["properties"]["seed_sequence_similarity"] == 1.0
    assert ranked[-1]["valid"] is False


def test_llm_design_engines_fail_clearly_in_default_mcp():
    ctx = MCPAgentContext()
    mol_spec = _spec("mol_design_molecules")
    pep_spec = _spec("peptide_design_peptides")

    mol_tool = build_tool(mol_spec, mol_spec.toolkit_factory(), ctx)
    pep_tool = build_tool(pep_spec, pep_spec.toolkit_factory(), ctx)

    with pytest.raises(MCPToolError, match="agno_team_run"):
        asyncio.run(mol_tool(goal="design molecules", engine="llm"))
    with pytest.raises(MCPToolError, match="agno_team_run"):
        asyncio.run(pep_tool(goal="design peptides", engine="llm"))


def test_pandas_facade_schema_hides_injected_parameters():
    ctx = MCPAgentContext()
    spec = _spec("pandas_normalize_for_analysis")
    tool = build_tool(spec, spec.toolkit_factory(), ctx)
    signature = inspect.signature(tool)

    assert "agent" not in signature.parameters
    assert "session_state" not in signature.parameters
    assert "df_path" in signature.parameters


def test_synplanner_identify_input_handles_smiles_and_fallback_names():
    tools = _spec("synplanner_identify_input").toolkit_factory()

    smiles = tools.identify_input("CCO")
    aspirin = tools.identify_input("aspirin")

    assert smiles["source"] == "smiles"
    assert smiles["smiles"] == "CCO"
    assert aspirin["source"] == "name"
    assert aspirin["smiles"]


def _manifest_payloads(tmp_path, session_name: str):
    manifest_root = (
        tmp_path
        / "data"
        / "sessions"
        / session_name
        / "workflows"
        / session_name
        / "manifests"
        / "mcp"
    )
    return [json.loads(path.read_text(encoding="utf-8")) for path in manifest_root.glob("*.json")]


def test_new_direct_tools_write_mcp_manifests(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    session_name = "direct-tool-manifest"
    S3.set_session_prefix(f"sessions/{session_name}")
    ctx = MCPAgentContext()
    ensure_output_context(ctx.session_state, workflow_slug="smoke")
    spec = _spec("skill_fetch")
    tool = build_tool(spec, spec.toolkit_factory(), ctx)

    result = asyncio.run(tool(slug="molecular-design", include_content=False))

    assert result["slug"] == "molecular-design"
    payloads = _manifest_payloads(tmp_path, session_name)
    assert len(payloads) == 1
    assert payloads[0]["tool_name"] == "skill_fetch"
    assert payloads[0]["status"] == "success"


def test_synplanner_backend_failure_surfaces_as_call_time_tool_error(monkeypatch):
    from cs_copilot.tools.chemistry.synplanner_toolkit import SynPlannerError

    spec = _spec("synplanner_plan_synthesis")
    instance = spec.toolkit_factory()

    def fail_backend():
        raise SynPlannerError("SynPlanner backend missing")

    monkeypatch.setattr(instance, "_load_synplanner_components", fail_backend)
    tool = build_tool(spec, instance, MCPAgentContext())

    with pytest.raises(MCPToolError, match="SynPlanner backend missing"):
        asyncio.run(tool(query="CCO"))
