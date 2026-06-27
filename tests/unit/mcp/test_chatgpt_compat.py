"""Tests for ChatGPT-compatible MCP search/fetch tools."""

from __future__ import annotations

import pytest

from cs_copilot.mcp.chatgpt_compat import fetch, search
from cs_copilot.storage import S3


@pytest.fixture
def isolated_session(tmp_path, monkeypatch):
    monkeypatch.setenv("USE_S3", "false")
    monkeypatch.chdir(tmp_path)
    S3.set_session_prefix(f"sessions/test-{tmp_path.name}")
    return tmp_path


def test_search_returns_catalog_entries(isolated_session):
    result = search("cs_copilot tools")
    ids = {item.id for item in result.results}
    assert "catalog:overview" in ids
    assert "catalog:tools" in ids


def test_search_can_find_tool_documentation(isolated_session):
    result = search("chembl fetch compounds")
    ids = [item.id for item in result.results]
    assert "tool:chembl_fetch_compounds" in ids
    assert "tool:chembl_prepare_retrieval" in ids

    fetched = fetch("tool:chembl_fetch_compounds")
    assert fetched.title == "Tool: chembl_fetch_compounds"
    assert "LLM-as-judge" in fetched.text
    assert fetched.metadata["kind"] == "tool"

    preflight = fetch("tool:chembl_prepare_retrieval")
    assert preflight.title == "Tool: chembl_prepare_retrieval"
    assert "preflight" in preflight.text.lower()


def test_fetch_renders_cs_copilot_workflow_prompt(isolated_session):
    result = search("cs_copilot workflow prompt")
    ids = [item.id for item in result.results]
    assert "prompt:cs_copilot_workflow" in ids

    fetched = fetch("prompt:cs_copilot_workflow")
    assert fetched.title == "Prompt: cs_copilot_workflow"
    assert fetched.metadata["kind"] == "prompt"
    assert "external reasoner" in fetched.text
    assert "agent selection" in fetched.text
    assert "session_state" in fetched.text


def test_fetch_renders_mcp_native_workflow_prompt(isolated_session):
    result = search("mcp workflow prompt")
    ids = [item.id for item in result.results]
    assert "prompt:cs_copilot_mcp_workflow" in ids

    fetched = fetch("prompt:cs_copilot_mcp_workflow")
    assert fetched.title == "Prompt: cs_copilot_mcp_workflow"
    assert fetched.metadata["kind"] == "prompt"
    assert "external MCP reasoner" in fetched.text
    assert "mcp_bootstrap" in fetched.text


def test_fetch_round_trips_text_session_artifact(isolated_session):
    with S3.open("notes.md", "w") as handle:
        handle.write("# Notes\ncs_copilot MCP artifact\n")

    result = search("notes")
    ids = [item.id for item in result.results]
    assert "resource:notes.md" in ids

    fetched = fetch("resource:notes.md")
    assert fetched.url == "cscopilot://session/notes.md"
    assert fetched.text.startswith("# Notes")
    assert fetched.metadata["mime_type"] == "text/markdown"


def test_search_and_fetch_skill_documentation(isolated_session):
    result = search("gtm activity landscape skill")
    ids = [item.id for item in result.results]
    assert "skill:gtm-activity-landscape" in ids

    fetched = fetch("skill:gtm-activity-landscape")
    assert fetched.title == "Skill: GTM activity landscape"
    assert fetched.metadata["kind"] == "skill"
    assert "gtm_create_activity_landscapes" in fetched.text
    assert "# GTM Activity Landscape" in fetched.text

    density = search("gtm density landscape skill")
    density_ids = [item.id for item in density.results]
    assert "skill:gtm-density-landscape" in density_ids

    fetched_density = fetch("skill:gtm-density-landscape")
    assert fetched_density.title == "Skill: GTM density landscape"
    assert fetched_density.metadata["kind"] == "skill"
    assert "gtm_save_density_plot" in fetched_density.text
    assert "# GTM Density Landscape" in fetched_density.text


def test_fetch_skill_catalog(isolated_session):
    fetched = fetch("catalog:skills")
    assert fetched.metadata["kind"] == "catalog"
    assert "chembl-target-retrieval" in fetched.text
    assert "retrosynthesis-planning" in fetched.text


def test_search_and_fetch_workflow_documentation(isolated_session):
    result = search("chembl gtm report workflow")
    ids = [item.id for item in result.results]
    assert "workflow:chembl-to-gtm-report" in ids

    fetched = fetch("workflow:chembl-to-gtm-report")
    assert fetched.title == "Workflow: ChEMBL to GTM report"
    assert fetched.metadata["kind"] == "workflow"
    assert "report_save_rich" in fetched.text
    assert "# ChEMBL To GTM Report" in fetched.text


def test_fetch_workflow_catalog(isolated_session):
    fetched = fetch("catalog:workflows")
    assert fetched.metadata["kind"] == "catalog"
    assert "chembl-to-gtm-report" in fetched.text
    assert "robustness-report" in fetched.text


def test_fetch_rejects_unknown_id(isolated_session):
    with pytest.raises(ValueError, match="Unknown cs_copilot MCP"):
        fetch("missing")


def test_search_finds_new_direct_tool_namespaces(isolated_session):
    molecular = [item.id for item in search("molecular design").results]
    peptide = [item.id for item in search("peptide design").results]
    synplanner = [item.id for item in search("synplanner").results]

    assert "tool:mol_validate_design_candidates" in molecular
    assert "tool:peptide_validate_design_candidates" in peptide
    assert "tool:synplanner_identify_input" in synplanner
    assert "tool:mcp_bootstrap" in [item.id for item in search("mcp bootstrap").results]


def test_fetch_renders_new_direct_tool_documentation(isolated_session):
    skill_doc = fetch("tool:skill_fetch")
    mol_doc = fetch("tool:mol_validate_design_candidates")

    assert skill_doc.title == "Tool: skill_fetch"
    assert "Group: skills" in skill_doc.text
    assert "SKILL.md" in skill_doc.text
    assert mol_doc.title == "Tool: mol_validate_design_candidates"
    assert "Group: molecular_design" in mol_doc.text
    assert "Validate" in mol_doc.text
