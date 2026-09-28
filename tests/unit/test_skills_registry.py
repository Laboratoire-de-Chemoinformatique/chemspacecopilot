"""Tests for the reusable ChemSpace skill registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from cs_copilot.mcp.tools_registry import all_specs
from cs_copilot.skills import SkillRegistry, get_skill, list_skills, search_skills
from cs_copilot.skills.registry import SKILLS_ENV, discover_skill_root


def test_default_registry_loads_initial_skill_catalog():
    skills = list_skills()
    slugs = {skill.slug for skill in skills}

    assert {
        "chembl-target-retrieval",
        "gtm-density-landscape",
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

    density = get_skill("gtm-density-landscape")
    assert "gtm_save_density_plot" in density.required_tools
    assert "# GTM Density Landscape" in density.skill_md

    assert all(skill.version == "2.0.0" for skill in skills)
    assert all(skill.profiles for skill in skills)
    assert all(skill.permissions for skill in skills)
    assert all(skill.input_artifacts for skill in skills)
    assert all(skill.output_artifacts for skill in skills)
    assert density.artifact_outputs == tuple(item.name for item in density.output_artifacts)


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
    (tmp_path / "broken").mkdir()

    registry = SkillRegistry(tmp_path)
    with pytest.raises(FileNotFoundError, match="SKILL.md"):
        registry.list_skills()


def test_explicit_missing_catalog_does_not_fall_back_to_bundled_skills(
    tmp_path: Path,
    monkeypatch,
):
    selected = tmp_path / "missing-skills"
    monkeypatch.setenv(SKILLS_ENV, str(selected))

    assert discover_skill_root() == selected.resolve()
    with pytest.raises(FileNotFoundError, match="Skill catalog root does not exist"):
        SkillRegistry().list_skills()


def test_custom_registry_rejects_skill_md_without_frontmatter(tmp_path: Path):
    skill_dir = tmp_path / "plain"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text("# Plain\n", encoding="utf-8")

    registry = SkillRegistry(tmp_path)
    with pytest.raises(ValueError, match="frontmatter"):
        registry.list_skills()


def test_custom_registry_rejects_slug_directory_mismatch(tmp_path: Path):
    skill_dir = tmp_path / "actual"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: other\ndescription: Other skill\n---\n\n# Other\n",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path)
    with pytest.raises(ValueError, match="must match directory"):
        registry.list_skills()


def test_stdlib_yaml_fallback_parses_nested_frontmatter():
    """The no-PyYAML fallback must read the name/description + metadata shape."""
    from cs_copilot.skills.registry import _parse_yaml_block

    text = (
        "name: demo\n"
        "description: Demo skill\n"
        "metadata:\n"
        "  title: Demo\n"
        "  keywords:\n"
        "    - alpha\n"
        "    - beta gamma\n"
        "  permissions:\n"
        "    - artifact:read\n"
        "  input_artifacts:\n"
        "    - name: source\n"
        "      kind: dataset\n"
        "      required: false\n"
    )
    data = _parse_yaml_block(text.splitlines(), Path("demo"))
    assert data["name"] == "demo"
    assert data["description"] == "Demo skill"
    assert data["metadata"]["title"] == "Demo"
    assert data["metadata"]["keywords"] == ["alpha", "beta gamma"]
    assert data["metadata"]["permissions"] == ["artifact:read"]
    assert data["metadata"]["input_artifacts"] == [
        {"name": "source", "kind": "dataset", "required": False}
    ]


def _write_skill(root: Path, slug: str, *, version: str = "1.0.0", depends_on=()):
    directory = root / slug
    directory.mkdir()
    dependencies = "\n".join(f"    - {item}" for item in depends_on) or "[]"
    if dependencies != "[]":
        dependencies = "\n" + dependencies
    (directory / "SKILL.md").write_text(
        "---\n"
        f"name: {slug}\n"
        f"description: {slug} description\n"
        "metadata:\n"
        f"  version: {version}\n"
        f"  depends_on: {dependencies}\n"
        "  profiles:\n"
        "    - standard\n"
        "  permissions:\n"
        "    - artifact:read\n"
        "  input_artifacts:\n"
        "    - name: source\n"
        "      kind: dataset\n"
        "      required: true\n"
        "  output_artifacts:\n"
        "    - name: result\n"
        "      kind: report\n"
        "      required: true\n"
        "---\n\n"
        f"# {slug}\n",
        encoding="utf-8",
    )


def test_registry_rejects_invalid_semver(tmp_path: Path):
    _write_skill(tmp_path, "invalid-version", version="1.0")

    with pytest.raises(ValueError, match="semantic versioning"):
        SkillRegistry(tmp_path).list_skills()


def test_registry_rejects_missing_and_cyclic_dependencies(tmp_path: Path):
    missing_root = tmp_path / "missing"
    missing_root.mkdir()
    _write_skill(missing_root, "consumer", depends_on=("absent",))
    with pytest.raises(ValueError, match="unknown skill dependencies"):
        SkillRegistry(missing_root).list_skills()

    cycle_root = tmp_path / "cycle"
    cycle_root.mkdir()
    _write_skill(cycle_root, "alpha", depends_on=("beta",))
    _write_skill(cycle_root, "beta", depends_on=("alpha",))
    with pytest.raises(ValueError, match="Skill dependency cycle"):
        SkillRegistry(cycle_root).list_skills()
