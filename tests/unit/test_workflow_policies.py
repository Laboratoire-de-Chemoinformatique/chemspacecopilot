"""Unit tests for import-safe workflow preflight policies."""

from __future__ import annotations

from cs_copilot.workflows import plan_chemical_space_analysis, prepare_chembl_retrieval


def test_chembl_preflight_rejects_bare_kinase_request():
    result = prepare_chembl_retrieval("Fetch kinase inhibitors")

    assert result["needs_clarification"] is True
    assert result["can_proceed"] is False
    assert "target_specificity" in result["missing_requirements"]
    assert result["recommended_next_tool"] is None


def test_chembl_preflight_rejects_kinase_index_request():
    result = prepare_chembl_retrieval("Fetch kinase 2 inhibitor data")

    assert result["needs_clarification"] is True
    assert "target_specificity" in result["missing_requirements"]
    assert result["target"] is None


def test_chembl_preflight_sparse_abbreviation_requires_retrieval_scope():
    result = prepare_chembl_retrieval("Fetch CDK2 data")

    assert result["needs_clarification"] is True
    assert result["target"] == "CDK2"
    assert set(result["missing_requirements"]) == {
        "target_confirmation",
        "organism",
        "assay_types",
        "mechanism",
    }


def test_chembl_preflight_allows_fully_scoped_abbreviation_request():
    result = prepare_chembl_retrieval("Fetch human CDK2 binding data, any mechanism")

    assert result["can_proceed"] is True
    assert result["needs_clarification"] is False
    assert result["target"] == "CDK2"
    assert result["organism"] == "Homo sapiens"
    assert result["assay_types"] == ["B"]
    assert result["mechanism"] is None
    assert result["mechanism_preference"] == "any"
    assert result["recommended_next_tool"] == "chembl_convert_to_chembl_query"


def test_chembl_preflight_does_not_treat_all_species_as_mechanism_preference():
    result = prepare_chembl_retrieval("Fetch CDK2 binding data for all species")

    assert result["needs_clarification"] is True
    assert result["organism"] == "all species"
    assert "mechanism" in result["missing_requirements"]


def test_chemical_space_preflight_asks_for_broad_request_details():
    result = plan_chemical_space_analysis("Analyze chemical space")

    assert result["can_proceed"] is False
    assert result["needs_clarification"] is True
    assert set(result["missing_requirements"]) == {"analysis_intent", "data_source"}


def test_chemical_space_preflight_allows_activity_landscape_with_session_dataset():
    result = plan_chemical_space_analysis(
        "Make an activity landscape",
        session_summary="current clean dataset is available",
    )

    assert result["can_proceed"] is True
    assert result["needs_clarification"] is False
    assert result["analysis_intents"] == ["activity_landscape"]
    assert result["dataset_source"] == "session_clean_dataset"
    assert "gtm_create_activity_landscapes" in result["recommended_next_tools"]
