#!/usr/bin/env python
# coding: utf-8
"""Tests for report-generation instruction requirements."""

from cs_copilot.agents.prompts import REPORT_GENERATOR_INSTRUCTIONS


def _report_instructions_text() -> str:
    return "\n".join(REPORT_GENERATOR_INSTRUCTIONS)


def test_report_instructions_require_named_captioned_inline_figures():
    """Report Generator instructions should enforce named, captioned inline figures."""
    instructions = _report_instructions_text()

    assert "Every available static PNG must be included as an inline report figure" in instructions
    assert "Every figure object MUST include name and caption" in instructions
    assert "Number figures sequentially across the whole report" in instructions
    assert "GTM landscape figures MUST appear in the section that describes them" in instructions
    assert (
        "do not defer density or activity landscapes to a final Visualizations section"
        in instructions
    )
    assert "structure_smiles or smiles" in instructions
    assert (
        "save_rich_report will generate a section-local compound image automatically"
        in instructions
    )
    assert "Do not render every valid SMILES" in instructions


def test_report_instructions_define_required_report_structures():
    """Workflow-specific reports should have stable required structures."""
    instructions = _report_instructions_text()

    assert "GTM analysis report required structure" in instructions
    assert "User Request and Data Source" in instructions
    assert "Retrieved and Standardized Data" in instructions
    assert "Descriptors" in instructions
    assert "GTM Construction or Loading" in instructions
    assert "Map Analysis" in instructions

    assert "Analog generation report required structure" in instructions
    assert "Reference Maps" in instructions
    assert "Generated Compound Analysis" in instructions

    assert "SynPlanner Routes and Attempts" in instructions
    assert "Route Analysis" in instructions
