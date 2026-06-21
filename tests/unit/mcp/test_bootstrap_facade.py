"""Tests for MCP bootstrap recommendations."""

from __future__ import annotations

from cs_copilot.mcp.facades.bootstrap import mcp_bootstrap_facade


def test_bootstrap_recommends_chembl_preflight_for_chembl_request():
    result = mcp_bootstrap_facade().bootstrap(
        "Fetch ChEMBL human EGFR binding data, full name epidermal growth "
        "factor receptor, any mechanism"
    )

    assert result["recommended_prompt"] == "cs_copilot_mcp_workflow"
    assert result["recommended_workflow"] == "chembl-target-retrieval"
    assert "chembl-target-retrieval" in result["relevant_skills"]
    assert result["preflight_tools"] == ["chembl_prepare_retrieval"]
    assert result["status"] == "ok"
    assert result["next_action"]["type"] == "fetch_context"
    assert "prompt:cs_copilot_mcp_workflow" in result["next_action"]["ids"]
    assert result["action_plan"][1]["type"] == "call_tool"
    assert result["action_plan"][1]["phase"] == "preflight"
    assert result["action_plan"][1]["tool"] == "chembl_prepare_retrieval"
    assert "policy_results" not in result
    assert (
        "target_specificity"
        in result["clarification_contract"]["do_not_infer_missing_requirements"]
    )
    assert (
        "chembl_fetch_compounds"
        in result["clarification_contract"]["blocked_tools_until_clarified"]
    )


def test_bootstrap_recommends_gtm_workflow_for_activity_landscape():
    result = mcp_bootstrap_facade().bootstrap(
        "Create a GTM activity landscape for the current clean dataset"
    )

    assert result["recommended_workflow"] == "gtm-activity-landscape"
    assert "gtm-activity-landscape" in result["relevant_skills"]
    assert "chemspace_plan_analysis" in result["preflight_tools"]
    assert "gtm_optimization" not in result["preflight_tools"]
    assert "gtm_optimization" in result["recommended_next_tools"]
    assert "workflow:gtm-activity-landscape" in result["fetch_ids"]
    assert result["action_plan"][0]["type"] == "fetch_context"
    assert result["action_plan"][1]["tool"] == "chemspace_plan_analysis"


def test_bootstrap_recommends_peptide_skill_for_peptide_design():
    result = mcp_bootstrap_facade().bootstrap(
        "Design antimicrobial peptide candidates and rank them"
    )

    assert result["recommended_workflow"] is None
    assert "peptide-design" in result["relevant_skills"]
    assert "skill:peptide-design" in result["fetch_ids"]


def test_bootstrap_routes_broad_request_to_preflight_not_bootstrap_questions():
    result = mcp_bootstrap_facade().bootstrap("help me analyze compounds")

    assert result["status"] == "ok"
    assert result["bootstrap_questions"] == []
    assert "chemspace_plan_analysis" in result["preflight_tools"]
    assert result["next_action"]["type"] == "fetch_context"
    assert result["next_action"]["ids"] == ["prompt:cs_copilot_mcp_workflow"]
    assert result["action_plan"][1]["tool"] == "chemspace_plan_analysis"
    assert result["clarification_contract"]["source"] == "preflight_tools"


def test_bootstrap_questions_are_limited_to_bootstrap_blockers():
    result = mcp_bootstrap_facade().bootstrap("")

    assert result["status"] == "needs_clarification"
    assert result["next_action"]["type"] == "ask_clarification"
    assert result["next_action"]["phase"] == "bootstrap"
    assert result["bootstrap_questions"] == ["What cs_copilot task should be planned?"]


def test_bootstrap_explicit_workflow_slug_wins():
    result = mcp_bootstrap_facade().bootstrap(
        "use this dataset",
        workflow_slug="dataset-normalization",
    )

    assert result["recommended_workflow"] == "dataset-normalization"
    assert result["recommended_workflow_id"] == "workflow:dataset-normalization"
    assert "Explicit workflow_slug" in " ".join(result["rationale"])
