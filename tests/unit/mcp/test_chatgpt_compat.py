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
    result = search("ChemSpace tools")
    ids = {item.id for item in result.results}
    assert "catalog:overview" in ids
    assert "catalog:tools" in ids


def test_search_can_find_tool_documentation(isolated_session):
    result = search("chembl fetch compounds")
    ids = [item.id for item in result.results]
    assert "tool:chembl_fetch_compounds" in ids

    fetched = fetch("tool:chembl_fetch_compounds")
    assert fetched.title == "Tool: chembl_fetch_compounds"
    assert "LLM-as-judge" in fetched.text
    assert fetched.metadata["kind"] == "tool"


def test_fetch_renders_chemspace_workflow_prompt(isolated_session):
    result = search("chemspace workflow prompt")
    ids = [item.id for item in result.results]
    assert "prompt:chemspace_workflow" in ids

    fetched = fetch("prompt:chemspace_workflow")
    assert fetched.title == "Prompt: chemspace_workflow"
    assert fetched.metadata["kind"] == "prompt"
    assert "external reasoner" in fetched.text
    assert "agent selection" in fetched.text
    assert "session_state" in fetched.text


def test_fetch_round_trips_text_session_artifact(isolated_session):
    with S3.open("notes.md", "w") as handle:
        handle.write("# Notes\nChemSpace MCP artifact\n")

    result = search("notes")
    ids = [item.id for item in result.results]
    assert "resource:notes.md" in ids

    fetched = fetch("resource:notes.md")
    assert fetched.url == "cscopilot://session/notes.md"
    assert fetched.text.startswith("# Notes")
    assert fetched.metadata["mime_type"] == "text/markdown"


def test_fetch_rejects_unknown_id(isolated_session):
    with pytest.raises(ValueError, match="Unknown ChemSpace MCP"):
        fetch("missing")
