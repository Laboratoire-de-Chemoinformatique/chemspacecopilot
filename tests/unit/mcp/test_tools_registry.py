"""Tests for the MCP tools registry — naming and uniqueness invariants."""

from __future__ import annotations

import re

from cs_copilot.mcp.tools_registry import all_specs

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def test_every_spec_has_valid_mcp_name():
    bad = [spec.mcp_name for spec in all_specs() if not _NAME_RE.match(spec.mcp_name)]
    assert not bad, f"Invalid MCP tool names: {bad!r}"


def test_mcp_names_are_unique():
    names = [spec.mcp_name for spec in all_specs()]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"Duplicate MCP tool names: {duplicates!r}"


def test_chembl_fetch_forces_judge_disabled():
    spec = next(s for s in all_specs() if s.mcp_name == "chembl_fetch_compounds")
    assert spec.forces.get("enable_retrieval_judge") is False
    assert spec.forces.get("enable_metadata_judge") is False


def test_every_method_resolves_against_its_toolkit():
    for spec in all_specs():
        instance = spec.toolkit_factory()
        bound = getattr(instance, spec.method, None)
        assert callable(bound), f"{spec.mcp_name}: missing method {spec.method!r}"


def test_every_spec_has_explicit_safety_hints():
    for spec in all_specs():
        assert isinstance(spec.read_only, bool), spec.mcp_name
        assert isinstance(spec.destructive, bool), spec.mcp_name
        assert isinstance(spec.open_world, bool), spec.mcp_name


def test_review_sensitive_tools_are_classified_conservatively():
    specs = {spec.mcp_name: spec for spec in all_specs()}

    assert specs["chembl_describe_dataset"].read_only is True
    assert specs["chem_calculate_tanimoto_similarity"].read_only is True
    assert specs["session_resolve_candidate_set"].read_only is True
    assert specs["robustness_generate_insights"].read_only is True

    assert specs["chembl_fetch_compounds"].read_only is False
    assert specs["gtm_optimization"].read_only is False
    assert specs["gtm_sample_nodes"].read_only is False
    assert specs["session_select_session_object"].read_only is False
    assert specs["session_summarize_session_memory"].read_only is False
    assert specs["report_save_markdown"].read_only is False
    assert specs["robustness_export_analysis_report"].read_only is False
