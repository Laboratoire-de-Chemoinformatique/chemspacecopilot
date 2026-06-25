"""Tests for the reusable ChemSpace skill registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.skills import SkillRegistry, get_skill, list_skills, search_skills


def test_default_registry_loads_initial_skill_catalog():
    skills = list_skills()
    slugs = {skill.slug for skill in skills}

    assert {
        "chembl-target-retrieval",
        "gtm-activity-landscape",
        "molecular-design",
        "peptide-design",
        "report-generation",
        "retrosynthesis-planning",
        "chembl-to-gtm-report",
        "candidate-design-to-gtm",
        "retrosynthesis-for-candidates",
        "dataset-normalization",
        "robustness-report",
    }.issubset(slugs)

    gtm = get_skill("gtm-activity-landscape")
    assert "gtm_create_activity_landscapes" in gtm.required_tools
    assert "# GTM Activity Landscape" in gtm.skill_md


def test_default_registry_searches_metadata_and_tools():
    results = search_skills("chembl standardization")
    assert results[0].slug == "chembl-target-retrieval"

    results = search_skills("report-generation")
    assert results[0].slug == "report-generation"

    results = search_skills("report_save_rich")
    assert any(result.slug == "report-generation" for result in results)


def test_default_skill_tool_references_exist_in_mcp_registry():
    tool_names = {spec.mcp_name for spec in all_specs()}

    missing = {}
    for skill in list_skills():
        referenced = set(skill.required_tools) | set(skill.optional_tools)
        absent = sorted(tool for tool in referenced if tool not in tool_names)
        if absent:
            missing[skill.slug] = absent

    assert not missing


def test_updated_mcp_skills_reference_current_direct_tools():
    molecular = get_skill("molecular-design")
    peptide = get_skill("peptide-design")
    retrosynthesis = get_skill("retrosynthesis-planning")

    assert "mol_design_molecules" in molecular.skill_md
    assert "mol_register_design_candidates" in molecular.required_tools
    assert 'engine="autoencoder"' in molecular.skill_md

    assert "peptide_design_peptides" in peptide.skill_md
    assert "peptide_validate_model_loaded" in peptide.optional_tools
    assert 'engine="wae"' in peptide.skill_md

    assert "synplanner_plan_synthesis" in retrosynthesis.skill_md
    assert "synplanner_identify_input" in retrosynthesis.required_tools


def test_custom_registry_rejects_missing_skill_md(tmp_path: Path):
    skill_dir = tmp_path / "broken"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "slug: broken\ntitle: Broken\nsummary: Broken skill\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)
    with pytest.raises(FileNotFoundError, match="SKILL.md"):
        registry.list_skills()


def test_custom_registry_rejects_slug_directory_mismatch(tmp_path: Path):
    skill_dir = tmp_path / "actual"
    skill_dir.mkdir()
    (skill_dir / "skill.yaml").write_text(
        "slug: other\ntitle: Other\nsummary: Other skill\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("# Other\n", encoding="utf-8")

    registry = SkillRegistry(tmp_path)
    with pytest.raises(ValueError, match="must match directory"):
        registry.list_skills()
