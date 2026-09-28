"""Tests for canonical MCP workflow-run resources."""

from __future__ import annotations

import json

import pytest

from cs_copilot.mcp import resources as mcp_resources
from cs_copilot.storage import S3
from cs_copilot.workflows import RunContext


@pytest.fixture
def isolated_session(tmp_path, monkeypatch):
    """Re-root the local storage prefix into a tmp dir for the duration of the test."""

    monkeypatch.setenv("USE_S3", "false")
    old_prefix = S3.current_prefix()
    prefix = f"sessions/test-{tmp_path.name}"
    S3.set_session_prefix(prefix)
    monkeypatch.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        S3.set_session_prefix(old_prefix)


def test_empty_session_has_no_synthetic_resources(isolated_session):
    assert mcp_resources.list_entries() == []


def test_list_and_round_trip_canonical_run_resources(isolated_session):
    context = RunContext.create("pilot", run_id="resource-run")
    artifact_path = "workflows/resource-run/reports/notes.md"
    with S3.open(artifact_path, "w") as handle:
        handle.write("# example\nhello\n")
    context.register_artifact(
        artifact_path,
        artifact_id="report-notes",
        artifact_type="markdown_report",
        mime_type="text/markdown",
        producer_tool="save_markdown_report",
    )

    entries = mcp_resources.list_entries()
    uris = {entry.uri for entry in entries}
    assert "cscopilot://runs/resource-run/manifest.json" in uris
    assert "cscopilot://runs/resource-run/artifacts/report-notes" in uris
    assert sum(uri.startswith("cscopilot://runs/resource-run/events/") for uri in uris) == 2

    text = mcp_resources.read_text("cscopilot://runs/resource-run/artifacts/report-notes")
    assert text.startswith("# example")
    assert (
        mcp_resources.resource_mime("cscopilot://runs/resource-run/artifacts/report-notes")
        == "text/markdown"
    )


def test_manifest_and_event_resources_are_valid_json(isolated_session):
    context = RunContext.create("pilot", run_id="json-run")
    manifest_uri = "cscopilot://runs/json-run/manifest.json"

    payload = json.loads(mcp_resources.read_text(manifest_uri))
    assert payload["schema_version"] == 2
    assert payload["session_id"] == context.run.session_id
    assert payload["run_id"] == "json-run"
    assert payload["workflow_slug"] == "pilot"

    event_uri = next(
        entry.uri
        for entry in mcp_resources.list_entries(run_id="json-run")
        if "/events/" in entry.uri
    )
    event = json.loads(mcp_resources.read_text(event_uri))
    assert event["event_type"] == "run_created"
    assert event["schema_version"] == 2


def test_artifact_read_rejects_checksum_mismatch(isolated_session):
    context = RunContext.create("pilot", run_id="tamper-run")
    artifact_path = "workflows/tamper-run/reports/result.txt"
    with S3.open(artifact_path, "w") as handle:
        handle.write("trusted")
    context.register_artifact(
        artifact_path,
        artifact_id="result",
        artifact_type="text_result",
        mime_type="text/plain",
    )
    with S3.open(artifact_path, "w") as handle:
        handle.write("tampered")

    with pytest.raises(ValueError, match="checksum"):
        mcp_resources.read_text("cscopilot://runs/tamper-run/artifacts/result")


def test_resources_replay_events_when_cross_process_snapshots_are_stale(
    isolated_session,
):
    context = RunContext.create("pilot", run_id="stale-resource-run")
    stale_writer = RunContext.load("stale-resource-run")
    artifact_path = "workflows/stale-resource-run/reports/result.txt"
    with S3.open(artifact_path, "w") as handle:
        handle.write("authoritative event-backed result")
    context.register_artifact(
        artifact_path,
        artifact_id="event-backed-result",
        artifact_type="text_result",
        mime_type="text/plain",
    )

    # Simulate an older process replacing the derived snapshots after the
    # artifact event committed.
    stale_writer._write_snapshots()
    root = isolated_session / "data" / S3.current_prefix() / "workflows" / "stale-resource-run"
    stale_index = json.loads((root / "artifacts" / "index.json").read_text())
    assert stale_index["artifacts"] == []

    artifact_uri = "cscopilot://runs/stale-resource-run/artifacts/event-backed-result"
    assert artifact_uri in {entry.uri for entry in mcp_resources.list_entries()}
    assert mcp_resources.read_text(artifact_uri) == "authoritative event-backed result"
    manifest = json.loads(
        mcp_resources.read_text("cscopilot://runs/stale-resource-run/manifest.json")
    )
    assert manifest["event_count"] == 2
    assert {artifact["artifact_id"] for artifact in manifest["artifacts"]} == {
        "event-backed-result"
    }


def test_resources_do_not_require_replaceable_snapshots(isolated_session):
    context = RunContext.create("pilot", run_id="snapshotless-resource-run")
    artifact_path = "workflows/snapshotless-resource-run/reports/result.txt"
    with S3.open(artifact_path, "w") as handle:
        handle.write("still available")
    context.register_artifact(
        artifact_path,
        artifact_id="snapshotless-result",
        artifact_type="text_result",
        mime_type="text/plain",
    )
    root = (
        isolated_session / "data" / S3.current_prefix() / "workflows" / "snapshotless-resource-run"
    )
    (root / "manifest.json").unlink()
    (root / "artifacts" / "index.json").unlink()

    uris = {entry.uri for entry in mcp_resources.list_entries()}
    assert "cscopilot://runs/snapshotless-resource-run/manifest.json" in uris
    assert (
        mcp_resources.read_text(
            "cscopilot://runs/snapshotless-resource-run/artifacts/snapshotless-result"
        )
        == "still available"
    )


@pytest.mark.parametrize(
    "uri",
    (
        "http://example.com/x.txt",
        "cscopilot://session/manifest.json",
        "cscopilot://runs/../manifest.json",
        "cscopilot://runs/run/events/../secret.jsonl",
        "cscopilot://runs/run/artifacts/a/b",
    ),
)
def test_unknown_or_traversing_uri_rejected(uri):
    with pytest.raises(ValueError):
        mcp_resources.read_text(uri)
