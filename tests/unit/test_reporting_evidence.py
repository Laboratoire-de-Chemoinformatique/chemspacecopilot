"""Evidence counts entities and preserves scope without expanding session context."""

import json

import pandas as pd
import pytest

from cs_copilot.tools.io import reporting_evidence as evidence


def test_atomic_coverage_distinguishes_combinations_missing_and_conflicts():
    df = pd.DataFrame(
        {
            "assay_chembl_ids": [" A|B ", "B|A", "A", None, "|nan|"],
            "document_chembl_ids": ["D1|D2", "D2", "D1", "", None],
        }
    )
    counts = evidence.coverage_counts(df)
    assert counts["distinct_assays"]["value"] == 2
    assert counts["distinct_documents"]["value"] == 2
    assert counts["distinct_assays"]["missing_rows"] == 2
    assert (
        evidence.coverage_counts(pd.DataFrame({"x": [1]}))["distinct_assays"]["status"]
        == "unresolved"
    )
    df["assay_chembl_id"] = ["X"] * 5
    conflict = evidence.coverage_counts(df)["distinct_assays"]
    assert conflict["status"] == "unresolved"
    assert conflict["value"] is None


def test_scaffolds_count_unique_structures_in_explicit_population():
    df = pd.DataFrame(
        {"smiles": ["c1ccccc1", "Cc1ccccc1", "CC", "CC", "C1CCCCC1"], "node_index": [1, 1, 2, 2, 3]}
    )
    full = evidence.scaffold_summary(df)
    region = evidence.scaffold_summary(df, nodes=[1])
    assert full["population_size"] == 4
    assert full["rows"][0] == {"scaffold_smiles": "c1ccccc1", "count": 2}
    assert region["population_size"] == 2
    assert region["selection"] == {"node_index": [1]}
    assert region["rows"] == [{"scaffold_smiles": "c1ccccc1", "count": 2}]
    assert "exact" in region["definition"]
    assert evidence.scaffold_summary(df, nodes=[99])["population_size"] == 0


def test_selective_read_checks_sources_and_paginates_before_returning(tmp_path):
    source = tmp_path / "clean.csv"
    df = pd.DataFrame(
        {
            "smiles": ["CC", "c1ccccc1", "C1CCCCC1"],
            "assay_chembl_ids": ["A|B", "B", "A"],
            "document_chembl_ids": ["D", "D", "D"],
        }
    )
    df.to_csv(source, index=False)
    reference = evidence.write_dataset_evidence(
        str(source), {"clean": (df, str(source), "All standardized molecules")}
    )
    assert set(reference) == {"dataset_version", "evidence_path", "status"}
    assert len(json.dumps(reference)) < 600
    result = json.loads(evidence.get_report_evidence(reference["evidence_path"]))
    assert result["coverage"]["distinct_assays"]["value"] == 2
    assert "identifiers" not in result and "scaffolds" not in result
    assert len(json.dumps(result)) < 2200
    page = json.loads(
        evidence.get_report_evidence(reference["evidence_path"], section="scaffolds", limit=1)
    )
    assert page["total_rows"] == 3 and len(page["rows"]) == 1
    assert page["population_size"] == 3 and page["next_offset"] == 1
    assert page["partial"] is True
    source.write_text("smiles\nCCC\n")
    stale = json.loads(evidence.get_report_evidence(reference["evidence_path"]))
    assert stale["status"] == "unresolved"
    assert "stale" in stale["reason"]
    assert "coverage" not in stale


def test_missing_artifact_and_invalid_selection_are_explicit(tmp_path):
    missing = json.loads(evidence.get_report_evidence(str(tmp_path / "absent.json")))
    assert missing["status"] == "unresolved"
    with pytest.raises(ValueError):
        evidence.get_report_evidence("unused", limit=0)


def test_preparation_persists_populations_without_expanding_state(monkeypatch, tmp_path):
    from cs_copilot.storage import client as storage_client
    from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset

    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    state = {}
    result = prepare_clean_dataset(
        pd.DataFrame(
            {
                "smiles": ["CC", "CC", "c1ccccc1", "invalid"],
                "assay_chembl_id": ["A", "B", "A", "X"],
                "document_chembl_id": ["D", "E", "D", "Z"],
            }
        ),
        source_name="evidence",
        session_state=state,
    )
    ref = result.standardization_summary["reporting_evidence"]
    clean = json.loads(evidence.get_report_evidence(ref["evidence_path"]))
    raw = json.loads(evidence.get_report_evidence(ref["evidence_path"], population="retained"))
    contributing = json.loads(
        evidence.get_report_evidence(ref["evidence_path"], population="contributing")
    )
    assert clean["coverage"]["distinct_assays"]["value"] == 2
    assert raw["coverage"]["distinct_assays"]["value"] == 3
    assert contributing["coverage"]["distinct_assays"]["value"] == 2
    assert state["data_file_paths"]["evidence_path"] == ref["evidence_path"]
    assert "identifiers" not in json.dumps(state)


def test_describe_dataset_labels_atomic_counts(tmp_path):
    from cs_copilot.tools.databases.chembl import ChemblToolkit

    path = tmp_path / "clean.csv"
    pd.DataFrame(
        {"assay_chembl_ids": ["A|B", "B|A", "A"], "document_chembl_ids": ["D|E", "D", "E"]}
    ).to_csv(path, index=False)
    result = json.loads(ChemblToolkit().describe_dataset(str(path)))
    assert result["coverage"]["distinct_assays"]["value"] == 2
    assert result["population_label"] == "All rows in the supplied dataset"


def test_node_tool_returns_scoped_paginated_evidence(monkeypatch, tmp_path):
    from cs_copilot.storage import client as storage_client
    from cs_copilot.tools.chemography.gtm_operations import analyze_scaffolds_in_nodes

    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    frame = pd.DataFrame({"smi": ["CC", "c1ccccc1", "C1CCCCC1"], "node_index": [1, 1, 2]})
    result = json.loads(analyze_scaffolds_in_nodes(frame, [1], limit=1))
    assert result["population_size"] == 2
    assert result["selection"] == {"node_index": [1]}
    assert result["total_rows"] == 2 and result["partial"]
    more = json.loads(
        evidence.get_report_evidence(
            result["evidence_path"], section="scaffolds", population="selected", offset=1
        )
    )
    assert len(more["rows"]) == 1
    assert more["dataset_version"] == result["dataset_version"]


def test_contributing_and_clean_identifier_disagreement_is_unresolved(tmp_path):
    clean = pd.DataFrame({"smiles": ["CC"], "assay_chembl_ids": ["A"]})
    contributing = pd.DataFrame({"smiles": ["CC", "CC"], "assay_chembl_id": ["A", "B"]})
    clean_path, contributing_path = tmp_path / "clean.csv", tmp_path / "contributing.csv"
    clean.to_csv(clean_path, index=False)
    contributing.to_csv(contributing_path, index=False)
    ref = evidence.write_dataset_evidence(
        str(clean_path),
        {
            "clean": (clean, str(clean_path), "All clean molecules"),
            "contributing": (contributing, str(contributing_path), "Contributing rows"),
        },
    )
    result = json.loads(evidence.get_report_evidence(ref["evidence_path"]))
    assert result["coverage"]["distinct_assays"]["status"] == "unresolved"
    assert result["coverage"]["distinct_assays"]["value"] is None


def test_clean_coverage_exposes_structure_denominator(tmp_path):
    path = tmp_path / "clean.csv"
    df = pd.DataFrame({"smiles": ["CC", "CCC"]})
    df.to_csv(path, index=False)
    ref = evidence.write_dataset_evidence(
        str(path), {"clean": (df, str(path), "All clean molecules")}
    )
    result = json.loads(evidence.get_report_evidence(ref["evidence_path"]))
    assert result["coverage"]["distinct_structures"]["value"] == 2
    assert "SMILES" in result["coverage"]["distinct_structures"]["method"]


def test_retrieval_separates_search_activity_and_clean_coverage(monkeypatch, tmp_path):
    from cs_copilot.storage import client as storage_client
    from cs_copilot.tools.databases.chembl import ChemblToolkit

    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    toolkit = ChemblToolkit()
    monkeypatch.setattr(
        toolkit._fetcher,
        "fetch_assays",
        lambda *args, **kwargs: [
            {"assay_chembl_id": "A"},
            {"assay_chembl_id": "B"},
            {"assay_chembl_id": "NO_ACTIVITIES"},
        ],
    )
    monkeypatch.setattr(
        toolkit._fetcher,
        "fetch_activities",
        lambda *args, **kwargs: [
            {
                "activity_id": 1,
                "assay_chembl_id": "A",
                "canonical_smiles": "CC",
                "document_chembl_id": "D",
            },
            {
                "activity_id": 2,
                "assay_chembl_id": "B",
                "canonical_smiles": "invalid",
                "document_chembl_id": "E",
            },
        ],
    )
    state = {}
    message = toolkit.fetch_compounds(
        "kinase", session_state=state, enable_retrieval_judge=False, enable_metadata_judge=False
    )
    path = state["data_file_paths"]["evidence_path"]
    actual = {
        population: json.loads(evidence.get_report_evidence(path, population=population))[
            "coverage"
        ]["distinct_assays"]["value"]
        for population in ("search", "retrieved", "retained", "clean")
    }
    assert actual == {"search": 3, "retrieved": 2, "retained": 2, "clean": 1}
    assert "Search-matched assays: 3" in message
    assert '"assay_count":' not in json.dumps(state)


@pytest.mark.parametrize("artifact", [[], {"schema_version": 1, "sources": []}])
def test_malformed_evidence_is_unresolved_instead_of_crashing(tmp_path, artifact):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(artifact))
    result = json.loads(evidence.get_report_evidence(str(path)))
    assert result["status"] == "unresolved"


def test_preparation_preserves_id_aliases_and_upstream_conflicts(monkeypatch, tmp_path):
    from cs_copilot.storage import client as storage_client
    from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset

    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    result = prepare_clean_dataset(
        pd.DataFrame(
            {
                "smiles": ["CC"],
                "assay_chembl_id": ["A"],
                "assay_chembl_ids": ["B"],
                "document_id": ["D"],
            }
        ),
        source_name="aliases",
    )
    ref = result.standardization_summary["reporting_evidence"]
    clean = json.loads(evidence.get_report_evidence(ref["evidence_path"]))["coverage"]
    assert clean["distinct_assays"]["status"] == "unresolved"
    assert clean["distinct_documents"]["value"] == 1


def test_prepare_without_raw_source_keeps_missing_population_unresolved(monkeypatch, tmp_path):
    from cs_copilot.storage import client as storage_client
    from cs_copilot.tools.chemistry.clean_dataset import prepare_clean_dataset

    monkeypatch.setattr(storage_client, "LOCAL_STORAGE_ROOT", tmp_path / "data")
    result = prepare_clean_dataset(
        pd.DataFrame({"smiles": ["CC"]}), source_name="no_raw", save_raw=False
    )
    assert result.raw_dataset_path == ""
    ref = result.standardization_summary["reporting_evidence"]
    retained = json.loads(evidence.get_report_evidence(ref["evidence_path"], population="retained"))
    assert retained["status"] == "unresolved"
    assert "retained" not in retained["available_populations"]


@pytest.mark.parametrize(
    "section,broken",
    [
        ("coverage", None),
        ("coverage", {"row_count": 1}),
        ("scaffolds", {"status": "verified", "rows": []}),
    ],
)
def test_incomplete_section_stays_unresolved_at_report_boundary(tmp_path, section, broken):
    path = tmp_path / "clean.csv"
    frame = pd.DataFrame({"smiles": ["CC"], "assay_chembl_ids": ["A"]})
    frame.to_csv(path, index=False)
    ref = evidence.write_dataset_evidence(
        str(path), {"clean": (frame, str(path), "All clean molecules")}
    )
    artifact = json.loads(open(ref["evidence_path"]).read())
    artifact["populations"]["clean"][section] = broken
    with open(ref["evidence_path"], "w") as handle:
        json.dump(artifact, handle)
    result = json.loads(evidence.get_report_evidence(ref["evidence_path"], section=section))
    assert result["status"] == "unresolved"
    rendered = evidence.evidence_report_section(
        {"evidence_path": ref["evidence_path"], "section": section}
    )
    assert rendered["paragraphs"][0].startswith("Unresolved:")
