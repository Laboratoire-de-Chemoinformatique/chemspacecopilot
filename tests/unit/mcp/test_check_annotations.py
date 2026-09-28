"""Tests for MCP readiness-check metadata validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cs_copilot.mcp.check import (
    CheckError,
    CheckReport,
    _report_payload,
    _required_tools_for_profile,
    _validate_server_instructions,
    _validate_tool_annotations,
)
from cs_copilot.mcp.server import SERVER_INSTRUCTIONS


def _tool(name: str, *, read_only: bool, destructive: bool = False, open_world: bool = False):
    return SimpleNamespace(
        name=name,
        annotations=SimpleNamespace(
            readOnlyHint=read_only,
            destructiveHint=destructive,
            openWorldHint=open_world,
        ),
    )


def test_validate_tool_annotations_counts_read_and_write_tools():
    total, read_only, write = _validate_tool_annotations(
        [
            _tool("search", read_only=True),
            _tool("fetch", read_only=True),
            _tool("chembl_fetch_compounds", read_only=False, open_world=True),
            _tool("gtm_optimization", read_only=False, open_world=True),
            _tool("report_save_markdown", read_only=False),
        ]
    )

    assert total == 5
    assert read_only == 2
    assert write == 3


def test_required_tools_follow_the_selected_profile():
    standard = _required_tools_for_profile("standard")
    bootstrap = _required_tools_for_profile("bootstrap")
    reporting = _required_tools_for_profile("reporting")

    assert "chembl_fetch_compounds" in standard
    assert "chembl_fetch_compounds" not in bootstrap
    assert "gtm_optimization" not in bootstrap
    assert "report_save_markdown" in reporting
    assert {"search", "fetch", "mcp_bootstrap"} <= set(bootstrap)


def test_explicit_required_tool_is_never_silently_filtered():
    assert "site_specific_probe" in _required_tools_for_profile(
        "bootstrap",
        ("site_specific_probe",),
    )


def test_validate_tool_annotations_rejects_missing_annotations():
    with pytest.raises(CheckError, match="without annotations"):
        _validate_tool_annotations([SimpleNamespace(name="search", annotations=None)])


def test_validate_tool_annotations_rejects_missing_read_only_hint():
    with pytest.raises(CheckError, match="without boolean readOnlyHint"):
        _validate_tool_annotations(
            [
                SimpleNamespace(
                    name="search",
                    annotations=SimpleNamespace(
                        readOnlyHint=None,
                        destructiveHint=False,
                        openWorldHint=False,
                    ),
                )
            ]
        )


def test_validate_tool_annotations_rejects_core_hint_mismatch():
    with pytest.raises(CheckError, match="incorrect readOnlyHint"):
        _validate_tool_annotations(
            [
                _tool("search", read_only=False),
                _tool("fetch", read_only=True),
                _tool("chembl_fetch_compounds", read_only=False),
                _tool("gtm_optimization", read_only=False),
                _tool("report_save_markdown", read_only=False),
            ]
        )


def test_validate_tool_annotations_rejects_destructive_hint_mismatch():
    with pytest.raises(CheckError, match="incorrect destructiveHint"):
        _validate_tool_annotations([_tool("search", read_only=True, destructive=True)])


def test_validate_tool_annotations_accepts_declared_destructive_recovery_tool():
    assert _validate_tool_annotations(
        [
            _tool(
                "workflow_abandon_tool_invocation",
                read_only=False,
                destructive=True,
            )
        ]
    ) == (1, 0, 1)


def test_validate_tool_annotations_accepts_declared_open_world_tool():
    assert _validate_tool_annotations(
        [_tool("chembl_fetch_compounds", read_only=False, open_world=True)]
    ) == (1, 0, 1)


def test_validate_tool_annotations_rejects_open_world_hint_mismatch():
    with pytest.raises(CheckError, match="incorrect openWorldHint"):
        _validate_tool_annotations(
            [_tool("chembl_fetch_compounds", read_only=False, open_world=False)]
        )

    missing_hint = _tool("site_specific_probe", read_only=True)
    missing_hint.annotations.openWorldHint = None
    with pytest.raises(CheckError, match="without boolean openWorldHint"):
        _validate_tool_annotations([missing_hint])


def test_validate_server_instructions_accepts_contract():
    assert _validate_server_instructions(SERVER_INSTRUCTIONS) == len(SERVER_INSTRUCTIONS)


def test_validate_server_instructions_rejects_missing_contract_parts():
    with pytest.raises(CheckError, match="incomplete server instructions"):
        _validate_server_instructions("cs_copilot MCP")


def test_validate_server_instructions_rejects_long_contract():
    long_contract = SERVER_INSTRUCTIONS + " " + ("extra " * 20)

    with pytest.raises(CheckError, match="512-character"):
        _validate_server_instructions(long_contract)


def test_report_payload_is_machine_readable():
    report = CheckReport(
        endpoint_url="https://mcp.example.com/mcp",
        session_id="session-1",
        tool_count=55,
        prompt_count=12,
        resource_count=1,
        required_tools=("search", "fetch"),
        fetch_id="tool:chembl_fetch_compounds",
        workflow_prompt_id="prompt:cs_copilot_mcp_workflow",
        skill_id="skill:gtm-activity-landscape",
        auth_enabled=True,
        mode="existing-url",
        annotated_tool_count=55,
        read_only_tool_count=31,
        write_tool_count=24,
        instructions_length=445,
    )

    payload = _report_payload(report)

    assert payload["status"] == "passed"
    assert payload["auth"] == "bearer-token"
    assert payload["required_tools"] == ["search", "fetch"]
    assert payload["workflow_prompt_id"] == "prompt:cs_copilot_mcp_workflow"
    assert payload["skill_id"] == "skill:gtm-activity-landscape"
    assert payload["chatgpt_connector_name"] == "cs_copilot"
    assert "chembl_fetch_compounds" in str(payload["chatgpt_expected_evidence"])
    assert "fetch prompt:cs_copilot_mcp_workflow" in str(payload["chatgpt_smoke_prompt"])
    assert "ChatGPT app" in str(payload["chatgpt_next"])
