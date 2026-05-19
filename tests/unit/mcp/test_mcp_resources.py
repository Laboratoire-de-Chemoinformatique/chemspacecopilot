"""Tests for the MCP resource layer (session artifact listing/reading)."""

from __future__ import annotations

import json

import pytest

from cs_copilot.mcp import resources as mcp_resources
from cs_copilot.storage import S3


@pytest.fixture
def isolated_session(tmp_path, monkeypatch):
    """Re-root the local storage prefix into a tmp dir for the duration of the test."""

    monkeypatch.setenv("USE_S3", "false")
    prefix = f"sessions/test-{tmp_path.name}"
    S3.set_session_prefix(prefix)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_list_entries_includes_manifest_only_on_empty_session(isolated_session):
    entries = mcp_resources.list_entries()
    uris = [e.uri for e in entries]
    assert "cscopilot://session/manifest.json" in uris


def test_round_trip_text_artifact(isolated_session):
    with S3.open("notes.md", "w") as handle:
        handle.write("# example\nhello\n")

    entries = mcp_resources.list_entries()
    uris = [e.uri for e in entries]
    assert "cscopilot://session/notes.md" in uris

    text = mcp_resources.read_text("cscopilot://session/notes.md")
    assert text.startswith("# example")


def test_manifest_is_valid_json(isolated_session):
    text = mcp_resources.read_text("cscopilot://session/manifest.json")
    payload = json.loads(text)
    assert "layout_version" in payload
    assert "session_prefix" in payload


def test_unknown_uri_scheme_rejected():
    with pytest.raises(ValueError):
        mcp_resources.read_text("http://example.com/x.txt")
