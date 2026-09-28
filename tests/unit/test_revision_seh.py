"""Scientific selection and snapshot integrity for the prospective sEH dataset."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "prepare_revision_seh", Path(__file__).parents[2] / "scripts" / "prepare_revision_seh.py"
)
seh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seh)


def target():
    return {
        "target_chembl_id": seh.TARGET,
        "organism": "Homo sapiens",
        "target_type": "SINGLE PROTEIN",
        "target_components": [{"accession": "P34913"}],
    }


def assay(identifier="CHEMBL_A", **updates):
    return {
        "assay_chembl_id": identifier,
        "target_chembl_id": seh.TARGET,
        "confidence_score": 9,
        "assay_type": "B",
        "assay_organism": None,
        **updates,
    }


def activity(identifier=1, value="100", **updates):
    return {
        "activity_id": identifier,
        "target_chembl_id": seh.TARGET,
        "target_organism": "Homo sapiens",
        "assay_chembl_id": "CHEMBL_A",
        "canonical_smiles": "CCO",
        "standard_type": "IC50",
        "standard_units": "nM",
        "standard_relation": "=",
        "standard_value": value,
        "molecule_chembl_id": "CHEMBL_M",
        "data_validity_comment": None,
        "potential_duplicate": 0,
        **updates,
    }


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        ({"standard_type": "Ki"}, "endpoint_not_IC50"),
        ({"standard_units": "uM"}, "units_not_nM"),
        ({"standard_relation": "<"}, "relation_not_equal"),
        ({"standard_value": "0"}, "value_not_finite_positive"),
        ({"standard_value": "NaN"}, "value_not_finite_positive"),
        ({"standard_value": "-2"}, "value_not_finite_positive"),
        ({"data_validity_comment": "Outside typical range"}, "data_validity_comment"),
        ({"potential_duplicate": 1}, "potential_duplicate"),
        ({"target_organism": "Mus musculus"}, "other_target_or_organism"),
        ({"canonical_smiles": "invalid"}, "invalid_or_empty_standardized_structure"),
    ],
)
def test_measurement_filters_are_explicit_and_do_not_mix_endpoints(updates, reason):
    result = seh.prepare_dataset(target(), [assay()], [activity(**updates)])
    assert not result["compounds"]
    assert result["counts"]["excluded_by_first_failed_filter"] == {reason: 1}


def test_assay_confidence_and_type_are_read_from_assay_metadata():
    result = seh.prepare_dataset(
        target(),
        [assay(confidence_score=8), assay("CHEMBL_B", assay_type="A")],
        [activity(), activity(2, assay_chembl_id="CHEMBL_B")],
    )
    assert result["counts"]["excluded_by_first_failed_filter"] == {
        "assay_confidence_not_9": 1,
        "assay_type_not_B_or_F": 1,
    }


def test_aggregation_collapses_duplicate_measurements_but_retains_conflicts():
    result = seh.prepare_dataset(
        target(),
        [assay(), assay("CHEMBL_B", assay_type="F")],
        [
            activity(1, "100", canonical_smiles="CCO.[Na+]"),
            activity(2, "100.0", canonical_smiles="OCC", molecule_chembl_id="CHEMBL_ALIAS"),
            activity(3, "10000", assay_chembl_id="CHEMBL_B"),
        ],
    )
    compound = result["compounds"][0]
    assert compound["canonical_smiles"] == "CCO"
    assert compound["n_measurements"] == 2
    assert compound["n_source_records"] == 3
    assert compound["activity_ids"] == [1, 2, 3]
    assert compound["pIC50"] == 6.0  # Median on log scale; not mean or median nM.
    assert compound["standard_value"] == 1000.0
    assert compound["activity_class"] == "intermediate"
    assert compound["activity_binary"] is None
    assert compound["activity_class_conflict"] is True
    assert compound["active_inactive_conflict"] is True
    assert result["counts"]["exact_within_assay_duplicate_measurements_collapsed"] == 1


def test_equal_measurements_in_distinct_assays_are_retained():
    result = seh.prepare_dataset(
        target(),
        [assay(), assay("CHEMBL_B")],
        [activity(), activity(2, assay_chembl_id="CHEMBL_B")],
    )
    assert result["compounds"][0]["n_measurements"] == 2


@pytest.mark.parametrize(("pic50", "label"), [(7, "active"), (6, "intermediate"), (5, "inactive")])
def test_activity_threshold_boundaries(pic50, label):
    assert seh.activity_class(pic50) == label


def test_wrong_target_and_duplicate_activity_ids_are_rejected():
    with pytest.raises(ValueError, match="Target metadata"):
        seh.prepare_dataset({**target(), "organism": "Mus musculus"}, [], [])
    with pytest.raises(ValueError, match="repeated activity"):
        seh.prepare_dataset(target(), [assay()], [activity(), activity()])


def fake_api(*, end_release="ChEMBL_37", total_activities=2):
    status_reads = 0

    def fetch(url):
        nonlocal status_reads
        if url.endswith("status.json"):
            status_reads += 1
            payload = {"chembl_db_version": "ChEMBL_37" if status_reads == 1 else end_release}
        elif "/target/" in url:
            payload = target()
        elif "/assay.json?" in url:
            payload = {
                "assays": [assay()],
                "page_meta": {"offset": 0, "total_count": 1, "next": None},
            }
        else:
            second = "offset=1" in url
            payload = {
                "activities": [activity(2 if second else 1)],
                "page_meta": {
                    "offset": int(second),
                    "total_count": total_activities,
                    "next": (
                        None
                        if second
                        else "/chembl/api/data/activity.json?target_chembl_id=CHEMBL2409&offset=1"
                    ),
                },
            }
        return json.dumps(payload).encode()

    return fetch


def test_snapshot_retains_pages_queries_and_hashes_and_detects_tampering(tmp_path):
    manifest = seh.download_snapshot(tmp_path, "ChEMBL_37", fetch=fake_api())
    assert manifest["complete"] is True
    assert len(manifest["collections"]["activities"]["paths"]) == 2
    assert len(manifest["artifacts"]) == 6
    saved, verified_target, assays, records = seh.read_snapshot(tmp_path)
    assert saved == manifest
    assert verified_target == target()
    assert len(assays) == 1 and len(records) == 2
    assert all(a["request_url"].startswith(seh.API) for a in manifest["artifacts"])
    (tmp_path / "raw/activity_0001.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        seh.read_snapshot(tmp_path)


def test_release_change_and_incomplete_pagination_cannot_be_certified(tmp_path):
    with pytest.raises(ValueError, match="release changed"):
        seh.download_snapshot(
            tmp_path / "release", "ChEMBL_37", fetch=fake_api(end_release="ChEMBL_38")
        )
    with pytest.raises(ValueError, match="Incomplete activity"):
        seh.download_snapshot(tmp_path / "count", "ChEMBL_37", fetch=fake_api(total_activities=3))
    with pytest.raises(ValueError, match="Incomplete raw"):
        seh.read_snapshot(tmp_path / "release")


def test_api_pagination_is_limited_to_public_chembl_origin():
    with pytest.raises(ValueError, match="origin"):
        seh._api_url("https://example.com/page.json")
    with pytest.raises(ValueError, match="path"):
        seh._api_url("https://www.ebi.ac.uk/other/page.json")


def test_phosphatase_domain_ic50_is_excluded_from_hydrolase_dataset():
    result = seh.prepare_dataset(
        target(),
        [assay("CHEMBL4415272")],
        [activity(assay_chembl_id="CHEMBL4415272")],
    )
    assert result["counts"]["excluded_by_first_failed_filter"] == {"phosphatase_domain_assay": 1}
    assert not result["compounds"]
