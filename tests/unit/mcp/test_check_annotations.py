"""Tests for MCP readiness-check metadata validation."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cs_copilot.mcp.check import (
    CheckError,
    CheckReport,
    _report_payload,
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
            _tool("chembl_fetch_compounds", read_only=False),
            _tool("gtm_optimization", read_only=False),
            _tool("report_save_markdown", read_only=False),
        ]
    )

    assert total == 5
    assert read_only == 2
    assert write == 3


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


def test_validate_tool_annotations_rejects_destructive_or_open_world_tools():
    with pytest.raises(CheckError, match="unexpected destructive"):
        _validate_tool_annotations([_tool("search", read_only=True, destructive=True)])

    with pytest.raises(CheckError, match="unexpected open-world"):
        _validate_tool_annotations([_tool("search", read_only=True, open_world=True)])


def test_validate_server_instructions_accepts_contract():
    assert _validate_server_instructions(SERVER_INSTRUCTIONS) == len(SERVER_INSTRUCTIONS)


def test_validate_server_instructions_rejects_missing_contract_parts():
    with pytest.raises(CheckError, match="incomplete server instructions"):
        _validate_server_instructions("ChemSpace Copilot MCP")


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
        workflow_prompt_id="prompt:chemspace_workflow",
        auth_enabled=True,
        mode="existing-url",
        annotated_tool_count=55,
        read_only_tool_count=31,
        write_tool_count=24,
        instructions_length=485,
    )

    payload = _report_payload(report)

    assert payload["status"] == "passed"
    assert payload["auth"] == "bearer-token"
    assert payload["required_tools"] == ["search", "fetch"]
    assert payload["workflow_prompt_id"] == "prompt:chemspace_workflow"
    assert payload["chatgpt_connector_name"] == "ChemSpace Copilot"
    assert "chembl_fetch_compounds" in str(payload["chatgpt_expected_evidence"])
    assert "fetch prompt:chemspace_workflow" in str(payload["chatgpt_smoke_prompt"])
    assert "ChatGPT app" in str(payload["chatgpt_next"])
