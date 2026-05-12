#!/usr/bin/env python
# coding: utf-8
"""Tests for report-generation instruction requirements."""

from cs_copilot.agents.prompts import REPORT_GENERATOR_INSTRUCTIONS


def _report_instructions_text() -> str:
    return "\n".join(REPORT_GENERATOR_INSTRUCTIONS)


def test_report_instructions_require_named_captioned_inline_figures():
    """Report Generator instructions should enforce named, captioned inline figures."""
    instructions = _report_instructions_text()

    assert "Every available non-Plotly static PNG must be included" in instructions
    assert "Every figure object MUST include name and caption" in instructions
    assert "Number figures sequentially across the whole report" in instructions
    assert (
        "GTM density landscape figures MUST appear directly after density analysis" in instructions
    )
    assert (
        "GTM activity landscape figures MUST appear directly after activity analysis"
        in instructions
    )
    assert (
        "do not defer density or activity landscapes to a final Visualizations section"
        in instructions
    )
    assert "pass mark_nodes for every GTM node discussed in the density text" in instructions
    assert "pass mark_nodes for every GTM node discussed in the activity text" in instructions
    assert "Every GTM node discussed in the report text MUST be explicitly labeled" in instructions
    assert "Density plots use a grayscale legend" in instructions
    assert "do not describe density as red/orange vs blue" in instructions
    assert "Include only the static PNG" in instructions
    assert "do not include Plotly PNGs, Plotly HTML, or Plotly artifact_path" in instructions
    assert "Do not put Plotly paths or GTM interactive .html artifact_path values" in instructions
    assert "structure_smiles or smiles" in instructions
    assert (
        "save_rich_report will generate a section-local compound image automatically"
        in instructions
    )
    assert "a scaffold/SAR paragraph contains an untagged valid SMILES" in instructions
    assert "Scaffold_1, Scaffold_2" in instructions
    assert "Molecule_1, Molecule_2" in instructions
    assert "If any dataset/source identifier already exists" in instructions
    assert (
        "structure_id, source_id, external_id, compound_id, molecule_id, scaffold_id"
        in instructions
    )
    assert "dataset-provided display names" in instructions
    assert "ChEMBL ID" in instructions
    assert "Only when no source ID exists" in instructions
    assert "MUST reference the matching figure" in instructions
    assert "CMPD-123, top potency source analog (Figure 4)" in instructions
    assert "Scaffold_1, Piperidine urea phenyl scaffold (Figure 3)" in instructions
    assert "Piperidine urea phenyl scaffold" in instructions
    assert "Top potency piperidine urea analog" in instructions
    assert "structure_type ('scaffold' or 'molecule')" in instructions
    assert "after_paragraph_index" in instructions
    assert "Scaffold ID / Scaffold / SMILES / Name / Node / Description" in instructions
    assert "Molecule ID / Molecule / SMILES / Name / Node / Description" in instructions
    assert "Scaffold inventory table rows with scaffold SMILES" in instructions
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
