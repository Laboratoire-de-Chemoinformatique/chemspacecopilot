"""Offline regression cases for the historical reporting reconciliation."""

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def reconciliation():
    path = Path(__file__).parents[2] / "scripts/reconcile_reporting_evidence.py"
    assert path.is_file(), "The offline reconciliation entry point is missing"
    spec = importlib.util.spec_from_file_location("reconcile_reporting_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reconciliation_separates_populations_and_serialized_combinations():
    module = reconciliation()
    raw = pd.DataFrame(
        {
            "activity_id": [1, 2, 3],
            "assay_chembl_id": ["A", "B", "A"],
            "document_chembl_id": ["D", "D", "E"],
        }
    )
    filtered = pd.DataFrame(
        {"activity_id": [4], "assay_chembl_id": ["C"], "document_chembl_id": ["F"]}
    )
    clean = pd.DataFrame(
        {
            "smiles": ["c1ccccc1", "Cc1ccccc1", "CC"],
            "assay_chembl_ids": ["A|B", "B|A", " A | B |"],
            "document_chembl_ids": ["D|E", "E|D", None],
        }
    )
    result = module.reconcile_tables(raw, filtered, clean)
    assert result["retained"]["distinct_assays"] == 2
    assert result["full_retrieved"]["distinct_assays"] == 3
    assert result["clean"]["distinct_assays"] == 2
    assert result["clean"]["distinct_documents"] == 2
    assert result["clean"]["serialized_assay_combinations"] == 3
    assert result["scaffolds"]["counts"] == {"c1ccccc1": 2, "": 1}
    assert result["scaffolds"]["denominator"] == 3


def test_reconciliation_rejects_ambiguous_retrieval_partition():
    module = reconciliation()
    raw = pd.DataFrame({"activity_id": [1], "assay_chembl_id": ["A"], "document_chembl_id": ["D"]})
    clean = pd.DataFrame(
        {"smiles": ["CC"], "assay_chembl_ids": ["A"], "document_chembl_ids": ["D"]}
    )
    with pytest.raises(ValueError, match="overlap"):
        module.reconcile_tables(raw, raw, clean)


def test_invalid_structures_are_explicit_and_not_acyclic_scaffolds():
    module = reconciliation()
    result = module.scaffold_summary(pd.Series(["CC", "not a molecule", None]))
    assert result["denominator"] == 3
    assert result["valid_molecules"] == 1
    assert result["invalid_molecules"] == 2
    assert result["counts"] == {"": 1}


def test_activity_subset_requires_exact_source_population_and_excludes_missing_activity():
    module = reconciliation()
    clean = pd.DataFrame({"smiles": ["CC", "c1ccccc1", "CCC"], "activity_final": [7.0, 5.0, None]})
    active = clean.iloc[[0]].copy()
    assert module.reconcile_activity_subset(clean, active, "active")["denominator"] == 1
    with pytest.raises(ValueError, match="activity subset"):
        module.reconcile_activity_subset(clean, clean.iloc[[1, 2]], "inactive")


def test_truncated_node_scope_is_never_inferred_as_all_nodes():
    module = reconciliation()
    calls = [
        {
            "tool_name": "analyze_scaffolds_in_nodes",
            "error": False,
            "tool_args": {"list_of_nodes": [0, 1, {"_truncated": 898}]},
            "result_preview": "scaffold_smi count Scaffold ID\nc1ccccc1 9 0",
        }
    ]
    scope = module.scaffold_call_evidence(calls)[0]
    assert scope["node_selection_complete"] is False
    assert scope["subset_recomputation_status"] == "unresolved"
    assert scope["recorded_counts"] == {"c1ccccc1": 9}


def test_six_run_replay_preserves_inputs_and_rejects_changed_evidence(tmp_path):
    module = reconciliation()
    source_root, audit_dir, output = (tmp_path / name for name in ("original", "audit", "output"))
    audit_dir.mkdir()

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    for relative in [
        "src/cs_copilot/tools/databases/chembl.py",
        "src/cs_copilot/tools/chemography/gtm_operations.py",
    ]:
        write(source_root / relative, "# Frozen fixture source\n")
    runs, datasets, prior = [], [], []
    for arm in ["team", "single_agent"]:
        for repetition in range(1, 4):
            root = source_root / f"{arm}_{repetition}"
            files = {}
            for name, value in {
                "raw": "activity_id,assay_chembl_id,document_chembl_id\n1,A,D\n2,B,E\n",
                "filtered": "activity_id,assay_chembl_id,document_chembl_id\n3,C,F\n",
                "clean": "smiles,assay_chembl_ids,document_chembl_ids\nc1ccccc1,A,D\nCC,B,E\n",
            }.items():
                path = root / "datasets" / name / f"{name}.csv"
                write(path, value)
                files[name] = {"path": str(path), "file_sha256": module.sha256(path)}
            state = {
                "data_file_paths": {
                    f"{name}_dataset_path": meta["path"] for name, meta in files.items()
                },
                "session_objects": {"datasets": {"ds_001": {"assay_count": 525}}},
            }
            write(root / "run/session_state.json", json.dumps({"session_state": state}))
            write(root / "run/response.txt", "525 assays; two scaffold examples.\n")
            bundle = {
                "case_name": "case_1_seh_analysis",
                "response_path": "run/response.txt",
                "task_success": True,
                "tool_calls": [
                    {
                        "tool_name": "run_dataframe_operation",
                        "error": True,
                        "tool_args": {
                            "operation": "filter",
                            "operation_parameters": "activity > 6",
                        },
                    }
                ],
                "generated_files": {},
            }
            bundle_path = root / "reliability/runs.jsonl"
            write(bundle_path, json.dumps(bundle) + "\n")
            runs.append(
                {
                    "system_under_test": arm,
                    "global_repetition": repetition - 1,
                    "case_name": bundle["case_name"],
                    "source_bundle": str(bundle_path),
                }
            )
            datasets.append(
                {
                    "arm": arm,
                    "repetition": repetition,
                    "files": files,
                    "retrieved_activity_records": 3,
                }
            )
            prior.append(
                {
                    "arm": arm,
                    "repetition": repetition,
                    "case": bundle["case_name"],
                    "clean_count": 2,
                    "full_retrieval_unique_assays": 3,
                    "full_retrieval_unique_documents": 3,
                    "raw_unique_assay_count": 2,
                    "raw_unique_document_count": 2,
                    "murcko_scaffold_count_including_acyclic": 2,
                }
            )
    with (audit_dir / "live_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    write(audit_dir / "live_dataset_comparison.json", json.dumps({"runs": datasets}))
    write(audit_dir / "live_scientific_audit.json", json.dumps({"records": prior}))
    write(audit_dir / "scientific_validity_review.txt", "Frozen fixture audit\n")
    write(audit_dir / "audit_live_science.py", "# Original fixture script\n")
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = module.reconcile(audit_dir, source_root, output)
    assert len(result["runs"]) == 6
    assert all(r["original_task_success"] is True for r in result["runs"])
    assert all(path.read_bytes() == data for path, data in originals.items())
    first = (output / "reconciliation.json").read_bytes()
    module.reconcile(audit_dir, source_root, output)
    assert (output / "reconciliation.json").read_bytes() == first
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for name, digest in result["original_source_sha256"].items()
        for path in [Path(name)]
    )
    Path(datasets[0]["files"]["raw"]["path"]).write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.reconcile(audit_dir, source_root, output)
    assert (output / "reconciliation.json").read_bytes() == first
