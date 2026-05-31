"""Tests for the reusable ChemSpace skill registry."""

from __future__ import annotations

from pathlib import Path

import pytest

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
    }.issubset(slugs)

    gtm = get_skill("gtm-activity-landscape")
    assert "gtm_create_activity_landscapes" in gtm.required_tools
    assert "# GTM Activity Landscape" in gtm.skill_md


def test_default_registry_searches_metadata_and_tools():
    results = search_skills("chembl standardization")
    assert results[0].slug == "chembl-target-retrieval"

    results = search_skills("report_save_rich")
    assert results[0].slug == "report-generation"


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
