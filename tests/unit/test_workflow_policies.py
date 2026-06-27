"""Unit tests for import-safe workflow preflight policies.

These gates validate the structured retrieval / analysis dimensions an external
MCP reasoner supplies; they do not parse free text. The reasoning engine (the
connected LLM) decides the fields, and the gate enforces the completeness
checklist and the "ask the user, don't infer" contract.
"""

from __future__ import annotations

from cs_copilot.workflows import plan_chemical_space_analysis, prepare_chembl_retrieval

# --- ChEMBL preflight ------------------------------------------------------


def test_chembl_preflight_requires_all_fields_when_empty():
    result = prepare_chembl_retrieval()

    assert result["can_proceed"] is False
    assert result["needs_clarification"] is True
    assert set(result["missing_requirements"]) == {
        "target_specificity",
        "organism",
        "assay_types",
        "mechanism",
    }
    assert result["recommended_next_tool"] is None


def test_chembl_preflight_clears_with_named_protein_and_full_scope():
    # Regression for the closed allow-list bug: any specific target the caller
    # supplies must clear the gate, not just a hardcoded list.
    result = prepare_chembl_retrieval(
        target="soluble epoxide hydrolase",
        organism="Homo sapiens",
        assay_types=["binding"],
        mechanism="any",
    )

    assert result["can_proceed"] is True
    assert result["needs_clarification"] is False
    assert result["target"] == "soluble epoxide hydrolase"
    assert result["target_type"] == "protein"
    assert result["organism"] == "Homo sapiens"
    assert result["assay_types"] == ["B"]
    assert result["mechanism"] is None
    assert result["mechanism_preference"] == "any"
    assert result["recommended_next_tool"] == "chembl_convert_to_chembl_query"


def test_chembl_preflight_accepts_gene_symbol_target():
    result = prepare_chembl_retrieval(
        target="EPHX2",
        organism="Homo sapiens",
        assay_types=["B"],
        mechanism="any",
    )

    assert result["can_proceed"] is True
    assert result["target"] == "EPHX2"


def test_chembl_preflight_named_target_still_requires_assay_and_mechanism():
    result = prepare_chembl_retrieval(
        target="soluble epoxide hydrolase",
        organism="Homo sapiens",
    )

    assert result["can_proceed"] is False
    assert "target_specificity" not in result["missing_requirements"]
    assert set(result["missing_requirements"]) == {"assay_types", "mechanism"}


def test_chembl_preflight_organism_target_skips_organism_requirement():
    result = prepare_chembl_retrieval(
        target="HIV-1",
        target_type="organism",
        assay_types=["binding"],
        mechanism="any",
    )

    assert result["can_proceed"] is True
    assert "organism" not in result["missing_requirements"]


def test_chembl_preflight_maps_assay_words_to_codes():
    result = prepare_chembl_retrieval(
        target="EGFR",
        organism="Homo sapiens",
        assay_types=["functional", "admet"],
        mechanism="any",
    )

    assert result["assay_types"] == ["F", "A"]


def test_chembl_preflight_records_specific_mechanism():
    result = prepare_chembl_retrieval(
        target="EGFR",
        organism="Homo sapiens",
        assay_types=["B"],
        mechanism="ATP-competitive",
    )

    assert result["can_proceed"] is True
    assert result["mechanism"] == "ATP-competitive"
    assert result["mechanism_preference"] == "ATP-competitive"


def test_chembl_preflight_all_species_counts_as_organism():
    result = prepare_chembl_retrieval(
        target="CDK2",
        organism="all species",
        assay_types=["B"],
        mechanism="any",
    )

    assert result["can_proceed"] is True
    assert result["organism"] == "all species"


# --- chemical-space preflight ----------------------------------------------


def test_chemical_space_preflight_requires_intent_and_source():
    result = plan_chemical_space_analysis()

    assert result["can_proceed"] is False
    assert result["needs_clarification"] is True
    assert set(result["missing_requirements"]) == {"analysis_intent", "data_source"}


def test_chemical_space_preflight_clears_with_intent_and_session_dataset():
    result = plan_chemical_space_analysis(
        analysis_intents=["activity_landscape"],
        dataset_source="session_clean_dataset",
    )

    assert result["can_proceed"] is True
    assert result["needs_clarification"] is False
    assert result["analysis_intents"] == ["activity_landscape"]
    assert result["dataset_source"] == "session_clean_dataset"
    assert "gtm_create_activity_landscapes" in result["recommended_next_tools"]


def test_chemical_space_preflight_recommends_density_plot_tools():
    result = plan_chemical_space_analysis(
        analysis_intents=["density_landscape"],
        dataset_source="session_clean_dataset",
    )

    assert result["can_proceed"] is True
    assert "gtm_save_density_plot" in result["recommended_next_tools"]
    assert "gtm_get_density_summary" in result["recommended_next_tools"]
    assert "gtm_create_activity_landscapes" not in result["recommended_next_tools"]


def test_chemical_space_preflight_chembl_intent_defaults_source():
    result = plan_chemical_space_analysis(analysis_intents=["chembl_retrieval"])

    assert result["can_proceed"] is True
    assert result["dataset_source"] == "chembl_retrieval"
    assert "chembl_prepare_retrieval" in result["recommended_next_tools"]
    assert any("chembl_prepare_retrieval" in note for note in result["notes"])


def test_chemical_space_preflight_accepts_comma_separated_intents():
    result = plan_chemical_space_analysis(
        analysis_intents="gtm_build, report_generation",
        dataset_source="explicit_path",
    )

    assert result["can_proceed"] is True
    assert result["analysis_intents"] == ["gtm_build", "report_generation"]
    assert "gtm_optimization" in result["recommended_next_tools"]
    assert "report_save_rich" in result["recommended_next_tools"]
