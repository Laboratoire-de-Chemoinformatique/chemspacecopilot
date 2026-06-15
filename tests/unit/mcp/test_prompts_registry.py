"""Tests for the MCP prompts registry."""

from __future__ import annotations

import re

from cs_copilot.agents import prompts as agent_prompts
from cs_copilot.mcp.prompts_registry import all_specs

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_every_prompt_has_valid_name():
    bad = [spec.mcp_name for spec in all_specs() if not _NAME_RE.match(spec.mcp_name)]
    assert not bad, f"Invalid MCP prompt names: {bad!r}"


def test_prompt_names_are_unique():
    names = [spec.mcp_name for spec in all_specs()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"Duplicate prompt names: {duplicates!r}"


def test_agent_prompts_render_non_empty_text():
    spec = next(s for s in all_specs() if s.mcp_name == "chembl_agent")
    rendered = spec.render()
    assert rendered.strip(), "rendered prompt was empty"


def test_mcp_workflow_prompt_is_external_reasoner_native():
    spec = next(s for s in all_specs() if s.mcp_name == "cs_copilot_mcp_workflow")
    rendered = spec.render()

    assert "external MCP reasoner" in rendered
    assert "mcp_bootstrap" in rendered
    assert "Call MCP tools directly" in rendered


def test_chembl_retrieval_judge_template_renders():
    spec = next(s for s in all_specs() if s.mcp_name == "chembl_retrieval_judge")
    rendered = spec.render(
        target_query="EGFR",
        keywords="EGFR,HER1",
        organism_filter="Homo sapiens",
        items="[]",
    )
    assert "EGFR" in rendered
    assert "Homo sapiens" in rendered
    assert "Items:" in rendered


def test_chembl_metadata_judge_template_renders():
    spec = next(s for s in all_specs() if s.mcp_name == "chembl_metadata_judge")
    rendered = spec.render(
        target_query="ABL1",
        keywords="ABL1",
        organism_filter="Homo sapiens",
        items="[]",
    )
    assert "target metadata" in rendered.lower()
    assert "ABL1" in rendered


def test_cs_copilot_workflow_prompt_uses_team_instructions():
    spec = next(s for s in all_specs() if s.mcp_name == "cs_copilot_workflow")
    rendered = spec.render()
    assert agent_prompts.AGENT_TEAM_INSTRUCTIONS, "team instructions constant is empty"
    # The rendered prompt should contain the first line of the team instructions.
    assert agent_prompts.AGENT_TEAM_INSTRUCTIONS[0] in rendered
