"""The offline replay verifies real report outputs against independent counts."""

import importlib.util
from pathlib import Path

import pandas as pd


def test_replay_checks_counts_and_reports_context_cost(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "replay_reporting_evidence.py"
    assert script.exists()
    spec = importlib.util.spec_from_file_location("reporting_replay", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean = tmp_path / "clean.csv"
    pd.DataFrame(
        {
            "smiles": ["CC", "c1ccccc1"],
            "assay_chembl_ids": ["A|B", "B"],
            "document_chembl_ids": ["D", "D"],
        }
    ).to_csv(clean, index=False)
    record = {
        "run": "example",
        "sources": {"clean": {"path": str(clean), "sha256": module.source_sha256(str(clean))}},
        "clean": {"rows": 2, "distinct_assays": 2, "distinct_documents": 1},
        "scaffolds": {
            "unique_scaffolds_including_acyclic": 2,
            "top_scaffolds": [["", 1], ["c1ccccc1", 1]],
        },
    }
    result = module.replay_run(record, tmp_path / "out")
    assert result["counts_match_independent_reconciliation"]
    assert result["evidence_tool_calls"] == 2
    assert result["coverage_payload_bytes"] < 3000
    assert result["metadata_reference_bytes"] < 600
    assert "| distinct assays | 2 |" in Path(result["report_path"]).read_text()
