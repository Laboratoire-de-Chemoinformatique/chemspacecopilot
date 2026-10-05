#!/usr/bin/env python3
"""Replay deterministic report facts against the independently reconciled frozen runs.

This exercises each arm's registered reader and shared renderer, not LLM inference.
The full evidence remains in output artifacts; measured payloads contain only the
requested coverage and first scaffold page. Original source bytes are unchanged.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from cs_copilot.tools.io.report_export import _normalize_sections, _render_markdown_report
from cs_copilot.tools.io.reporting_evidence import (
    evidence_report_section,
    get_report_evidence,
    source_sha256,
    write_dataset_evidence,
)


def replay_run(record, output_dir, reader=get_report_evidence):
    output_dir.mkdir(parents=True, exist_ok=True)
    source = record["sources"]["clean"]
    if source_sha256(source["path"]) != source["sha256"]:
        raise ValueError(f"Changed original source: {source['path']}")
    df = pd.read_csv(source["path"])
    ref = write_dataset_evidence(
        source["path"],
        {
            "clean": (df, source["path"], "All clean standardized molecules in " + record["run"]),
        },
        evidence_path=str(output_dir / "evidence.json"),
    )
    coverage_text = reader(ref["evidence_path"])
    scaffold_text = reader(ref["evidence_path"], section="scaffolds", limit=10)
    coverage, scaffolds = json.loads(coverage_text), json.loads(scaffold_text)
    for metric in ("distinct_assays", "distinct_documents"):
        assert coverage["coverage"][metric]["value"] == record["clean"][metric], metric
    assert coverage["coverage"]["row_count"] == record["clean"]["rows"]
    assert scaffolds["population_size"] == record["clean"]["rows"]
    assert scaffolds["total_rows"] == record["scaffolds"]["unique_scaffolds_including_acyclic"]
    # Inspect full artifacts locally, without sending them through the reader.
    stored = json.loads(Path(ref["evidence_path"]).read_text())
    computed = {
        row["scaffold_smiles"]: row["count"]
        for row in stored["populations"]["clean"]["scaffolds"]["rows"]
    }
    for smi, count in record["scaffolds"]["top_scaffolds"]:
        assert computed[smi] == count
    if "all_scaffold_counts" in record:
        assert computed == record["all_scaffold_counts"]
    sections = [
        evidence_report_section({"evidence_path": ref["evidence_path"], "section": section})
        for section in ("coverage", "scaffolds")
    ]
    report = _render_markdown_report(
        "Verified evidence replay: " + record["run"],
        [],
        _normalize_sections(sections),
        [],
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report)
    if source_sha256(source["path"]) != source["sha256"]:
        raise ValueError("Original source changed during replay")
    return {
        "run": record["run"],
        "source": source,
        "report_path": str(report_path),
        "report_sha256": source_sha256(str(report_path)),
        "counts_match_independent_reconciliation": True,
        "metadata_reference_bytes": len(json.dumps(ref).encode()),
        "coverage_payload_bytes": len(coverage_text.encode()),
        "scaffold_page_payload_bytes": len(scaffold_text.encode()),
        "full_evidence_bytes": Path(ref["evidence_path"]).stat().st_size,
        "evidence_tool_calls": 2,
        "repeated_evidence_tool_calls": 0,
        "renderer_revalidates_sources": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corrections-dir",
        type=Path,
        default=Path("docs/manuscript/results/reporting-corrections"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/reporting_evidence_replay")
    )
    args = parser.parse_args()
    from cs_copilot.agents.factories import ReportGeneratorFactory, SingleAgentFactory

    readers = {
        arm: next(
            tool
            for tool in factory().get_agent_config().tools
            if getattr(tool, "__name__", "") == "get_report_evidence"
        )
        for arm, factory in (("team", ReportGeneratorFactory), ("single_agent", SingleAgentFactory))
    }
    oracle = json.loads((args.corrections_dir / "reconciliation.json").read_text())
    all_counts = {}
    with (args.corrections_dir / "murcko_scaffold_frequencies.csv").open() as handle:
        for row in csv.DictReader(handle):
            all_counts.setdefault(row["run"], {})[row["scaffold_smiles"]] = int(row["count"])
    results = []
    for record in oracle["runs"]:
        record["all_scaffold_counts"] = all_counts[record["run"]]
        results.append(
            replay_run(record, (args.output_dir / record["run"]).resolve(), readers[record["arm"]])
        )
    manifest = {
        "scope": "Offline deterministic replay through registered team/single reporting readers and shared renderer; no LLM inference or benchmark rescoring",
        "script_sha256": source_sha256(str(Path(__file__).resolve())),
        "runtime_sha256": source_sha256(
            str(
                Path(__file__).resolve().parents[1]
                / "src/cs_copilot/tools/io/reporting_evidence.py"
            )
        ),
        "runs": results,
    }
    (args.corrections_dir / "reporting_replay.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "runs_verified": len(results),
                "max_coverage_payload_bytes": max(r["coverage_payload_bytes"] for r in results),
                "max_metadata_reference_bytes": max(r["metadata_reference_bytes"] for r in results),
            }
        )
    )


if __name__ == "__main__":
    main()
