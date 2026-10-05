#!/usr/bin/env python3
"""Reconcile the six retained sEH reports offline, without changing evidence or scores.

Usage: python scripts/reconcile_reporting_evidence.py --audit-dir ANALYSIS_DIRECTORY
       --source-root RETAINED_V4_CHECKOUT --output-dir CORRECTIONS_DIRECTORY
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

ADAMANTANE = "C1C2CC3CC1CC(C2)C3"
PHENYL_UREA_ADAMANTYL = "O=C(Nc1ccccc1)NC12CC3CC(CC(C3)C1)C2"
SCAFFOLD_METHOD = (
    "RDKit GetScaffoldForMol on each original clean SMILES; canonical isomeric "
    "SMILES (atom and bond types retained, no generic scaffold conversion); "
    "count every valid clean row once, including empty acyclic scaffolds."
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_ids(values):
    return {
        token.strip()
        for value in values.dropna()
        for token in str(value).split("|")
        if token.strip()
    }


def coverage(frame, clean=False):
    suffix = "s" if clean else ""
    assays = frame[f"assay_chembl_id{suffix}"]
    documents = frame[f"document_chembl_id{suffix}"]
    result = {
        "rows": len(frame),
        "distinct_assays": len(atomic_ids(assays)),
        "distinct_documents": len(atomic_ids(documents)),
    }
    if clean:
        result.update(
            serialized_assay_combinations=int(assays.nunique()),
            serialized_document_combinations=int(documents.nunique()),
        )
    return result


def scaffold_summary(smiles):
    counts = Counter()
    invalid = 0
    for value in smiles:
        mol = Chem.MolFromSmiles(value) if isinstance(value, str) and value.strip() else None
        if mol is None:
            invalid += 1
            continue
        counts[Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))] += 1
    return {
        "method": SCAFFOLD_METHOD,
        "population": "all clean standardized compound rows",
        "denominator": len(smiles),
        "valid_molecules": sum(counts.values()),
        "invalid_molecules": invalid,
        "unique_scaffolds_including_acyclic": len(counts),
        "acyclic_molecules": counts[""],
        "counts": dict(sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))),
    }


def reconcile_tables(raw, filtered, clean):
    for name, table in [("retained", raw), ("filtered", filtered)]:
        if table.activity_id.isna().any() or table.activity_id.duplicated().any():
            raise ValueError(f"Missing or duplicate activity IDs in {name} table")
    if set(raw.activity_id) & set(filtered.activity_id):
        raise ValueError("Retained and filtered activity IDs overlap")
    full = pd.concat([raw, filtered], ignore_index=True)
    return {
        "retained": coverage(raw),
        "filtered": coverage(filtered),
        "full_retrieved": coverage(full),
        "clean": coverage(clean, clean=True),
        "reconstruction": "disjoint union of retained raw and excluded activity rows by activity_id",
        "scaffolds": scaffold_summary(clean.smiles),
    }


def reconcile_activity_subset(clean, subset, activity_class):
    mask = clean.activity_final >= 6 if activity_class == "active" else clean.activity_final < 6
    expected = clean.loc[mask, ["smiles", "activity_final"]]
    if Counter(expected.itertuples(index=False, name=None)) != Counter(
        subset[["smiles", "activity_final"]].itertuples(index=False, name=None)
    ):
        raise ValueError(f"Original {activity_class} activity subset disagrees with clean dataset")
    return scaffold_summary(subset.smiles)


def scaffold_call_evidence(calls):
    evidence = []
    loaded_datasets = {}
    for index, call in enumerate(calls):
        agent = call.get("agent_name")
        if call["tool_name"] == "load_and_prep_data" and not call.get("error"):
            loaded_datasets[agent] = call["tool_args"].get("dataset")
        if call["tool_name"] != "analyze_scaffolds_in_nodes" or call.get("error"):
            continue
        nodes = call["tool_args"].get("list_of_nodes", [])
        counts = {}
        for line in call.get("result_preview", "").splitlines()[1:]:
            match = re.fullmatch(r"\s*(.*?)\s+(\d+)\s+\d+\s*", line)
            if match:
                counts[match[1]] = int(match[2])
        evidence.append(
            {
                "tool_call_index": index,
                "tool_name": call["tool_name"],
                "original_loaded_dataset": loaded_datasets.get(agent),
                "recorded_node_selection": nodes,
                "node_selection_complete": bool(nodes) and all(isinstance(n, int) for n in nodes),
                "recorded_counts": counts,
                "origin": "original stored tool-call preview; may be truncated",
                "subset_recomputation_status": "unresolved",
                "subset_denominator": None,
                "reason": "Original per-molecule node membership is not available in these previews; no projection inference rerun.",
            }
        )
    return evidence


def evidence_lines(text, pattern):
    return [line.strip() for line in text.splitlines() if re.search(pattern, line, re.I)]


def reconcile(audit_dir, source_root, output_dir):
    audit_dir, source_root, output_dir = map(Path, (audit_dir, source_root, output_dir))
    sources = {}

    def source(path, expected=None):
        path = Path(path).resolve()
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Historical input hash mismatch: {path}")
        sources[str(path)] = digest
        return {"path": str(path), "sha256": digest, "kind": "original"}

    def original_path(path):
        path = Path(path)
        historical_root = Path("/home/aorlov/Programs/agents/chemspacecopilot-v4")
        if path.is_relative_to(historical_root):
            return source_root / path.relative_to(historical_root)
        return path if path.is_absolute() else source_root / path

    index_path = audit_dir / "live_runs.csv"
    comparison_path = audit_dir / "live_dataset_comparison.json"
    for path in [
        index_path,
        comparison_path,
        audit_dir / "live_scientific_audit.json",
        audit_dir / "scientific_validity_review.txt",
        audit_dir / "audit_live_science.py",
    ]:
        source(path)
    comparison = json.loads(comparison_path.read_text())
    original_audit = json.loads((audit_dir / "live_scientific_audit.json").read_text())
    original_records = {
        (r["arm"], r["repetition"]): r
        for r in original_audit["records"]
        if r["case"] == "case_1_seh_analysis"
    }
    datasets = {(r["arm"], r["repetition"]): r for r in comparison["runs"]}
    retrieval_code = source(source_root / "src/cs_copilot/tools/databases/chembl.py")
    scaffold_code = source(source_root / "src/cs_copilot/tools/chemography/gtm_operations.py")
    records, ledger, scaffold_rows = [], [], []
    with index_path.open(newline="") as handle:
        run_rows = [r for r in csv.DictReader(handle) if r["case_name"] == "case_1_seh_analysis"]
    if len(run_rows) != 6:
        raise ValueError("Expected exactly six retained sEH analysis runs")
    for row in run_rows:
        arm, repetition = row["system_under_test"], int(row["global_repetition"]) + 1
        run_id = f"{arm}_rep{repetition}"
        bundle_path = original_path(row["source_bundle"])
        bundle_source = source(bundle_path)
        bundle = next(
            r
            for r in map(json.loads, bundle_path.read_text().splitlines())
            if r["case_name"] == row["case_name"]
        )
        run_dir = bundle_path.parent.parent / Path(bundle["response_path"]).parent
        state_source = source(run_dir / "session_state.json")
        response_source = source(run_dir / "response.txt")
        state = json.loads(Path(state_source["path"]).read_text())["session_state"]
        response = Path(response_source["path"]).read_text()
        files = datasets[arm, repetition]["files"]
        table_sources = {
            name: source(original_path(meta["path"]), meta["file_sha256"])
            for name, meta in files.items()
        }
        tables = {name: pd.read_csv(meta["path"]) for name, meta in table_sources.items()}
        for name in ["raw", "filtered", "clean"]:
            pointer = (
                state.get(f"chembl_{name}_dataset_path")
                or state["data_file_paths"][f"{name}_dataset_path"]
            )
            if original_path(pointer).resolve() != Path(table_sources[name]["path"]):
                raise ValueError(f"Audit manifest/state dataset mismatch for {run_id}/{name}")
        result = reconcile_tables(tables["raw"], tables["filtered"], tables["clean"])
        prior = original_records[arm, repetition]
        checks = {
            "clean_count": result["clean"]["rows"] == prior["clean_count"],
            "full_assays": result["full_retrieved"]["distinct_assays"]
            == prior["full_retrieval_unique_assays"],
            "full_documents": result["full_retrieved"]["distinct_documents"]
            == prior["full_retrieval_unique_documents"],
            "retained_assays": result["retained"]["distinct_assays"]
            == prior["raw_unique_assay_count"],
            "retained_documents": result["retained"]["distinct_documents"]
            == prior["raw_unique_document_count"],
            "scaffold_count": result["scaffolds"]["unique_scaffolds_including_acyclic"]
            == prior["murcko_scaffold_count_including_acyclic"],
            "retrieval_rows": result["full_retrieved"]["rows"]
            == datasets[arm, repetition]["retrieved_activity_records"],
        }
        if not all(checks.values()):
            raise ValueError(
                f"Independent recomputation disagrees with historical audit: {run_id}: {checks}"
            )
        report_sources, report_texts = [], []
        for path in sorted(set(bundle["generated_files"].values())):
            resolved = original_path(path)
            if resolved.suffix == ".md" and "report" in resolved.name and resolved.is_file():
                report_sources.append(source(resolved))
                report_texts.append(resolved.read_text())
        calls = scaffold_call_evidence(bundle["tool_calls"])
        identifier_calls = []
        for index, call in enumerate(bundle["tool_calls"]):
            args = call["tool_args"]
            if (
                call["tool_name"] != "run_dataframe_operation"
                or args.get("operation") != "nunique"
                or call.get("error")
            ):
                continue
            parameters = args.get("operation_parameters", {})
            if isinstance(parameters, str):
                parameters = json.loads(parameters)
            column = parameters.get("column")
            if column in {"assay_chembl_ids", "document_chembl_ids"}:
                identifier_calls.append(
                    {
                        "tool_call_index": index,
                        "column": column,
                        "original_result": call["result_preview"],
                        "serialized_combinations": int(tables["clean"][column].nunique()),
                        "distinct_atomic_ids": len(atomic_ids(tables["clean"][column])),
                        "source": bundle_source,
                    }
                )
        artifact_inventory = []
        base = Path(table_sources["clean"]["path"]).parents[2]
        for path in sorted(base.rglob("*.csv")):
            with path.open() as handle:
                delimiter = "\t" if "\t" in handle.readline() else ","
            columns = pd.read_csv(path, sep=delimiter, nrows=0).columns.tolist()
            artifact_inventory.append({**source(path), "columns": columns})
        activity_subsets = {}
        for activity_class in ["active", "inactive"]:
            pointer = (
                state.get("chemotype_analysis", {})
                .get("output_paths", {})
                .get(f"{activity_class}_subset_csv")
            )
            if pointer:
                subset_source = source(original_path(pointer))
                summary = reconcile_activity_subset(
                    tables["clean"], pd.read_csv(subset_source["path"]), activity_class
                )
                activity_subsets[activity_class] = {
                    "status": "verified",
                    "source": subset_source,
                    "rule": "activity_final >= 6"
                    if activity_class == "active"
                    else "activity_final < 6",
                    **summary,
                }
        original_claims = evidence_lines(response, r"assay|document|scaffold|adamant|urea")
        counts = result["scaffolds"].pop("counts")
        result["scaffolds"]["top_scaffolds"] = list(counts.items())[:10]
        result["scaffolds"]["adamantane"] = counts.get(ADAMANTANE, 0)
        result["scaffolds"]["phenyl_urea_adamantyl"] = counts.get(PHENYL_UREA_ADAMANTYL, 0)
        scaffold_rows.extend(
            {
                "run": run_id,
                "population": "all_clean",
                "denominator": len(tables["clean"]),
                "scaffold_smiles": smi,
                "count": count,
            }
            for smi, count in counts.items()
        )
        metadata_count = state["session_objects"]["datasets"]["ds_001"].get("assay_count")
        result.update(
            run=run_id,
            arm=arm,
            repetition=repetition,
            original_task_success=bundle["task_success"],
            original_score_unchanged=True,
            computation_kind="offline reconstruction from original artifacts",
            sources={
                "bundle": bundle_source,
                "state": state_source,
                "response": response_source,
                **table_sources,
                "reports": report_sources,
            },
            original_claim_excerpts=original_claims,
            original_report_claim_excerpts=[
                evidence_lines(t, r"assay|document|scaffold|adamant|urea") for t in report_texts
            ],
            search_assay_metadata={
                "recorded_count": metadata_count,
                "scope": "search-matched assay IDs before activity retrieval",
                "scope_status": "verified_from_original_code_and_state",
                "independent_id_count_status": "unresolved",
                "reason": "The matched assay ID list was not retained; 525 is metadata, not verified retained coverage.",
                "code_source": retrieval_code,
            },
            scaffold_calls=calls,
            original_saved_activity_subsets=activity_subsets,
            serialized_identifier_count_calls=identifier_calls,
            csv_artifact_inventory=artifact_inventory,
            historical_audit_agreement=checks,
        )
        records.append(result)
        common_sources = [
            response_source,
            state_source,
            bundle_source,
            *table_sources.values(),
            *report_sources,
        ]
        ledger.append(
            {
                "run": run_id,
                "claim_type": "assay_document_coverage",
                "original_claim": evidence_lines(response, r"assay|document"),
                "source_artifacts": common_sources,
                "population": "retained raw; reconstructed full retrieval; clean compounds",
                "counting_method": "Split pipe-delimited IDs, trim blanks, count distinct nonempty atomic IDs; never count serialized combinations as IDs.",
                "corrected_value": {k: result[k] for k in ["retained", "full_retrieved", "clean"]},
                "original_count_operation_evidence": identifier_calls,
                "status": "verified",
                "evidence_kind": "reconstructed",
                "originals_preserved": True,
            }
        )
        for activity_class, summary in activity_subsets.items():
            ledger.append(
                {
                    "run": run_id,
                    "claim_type": "saved_activity_subset_scaffolds",
                    "original_claim": state["chemotype_analysis"]
                    .get("scaffolds_per_activity_class", {})
                    .get(activity_class),
                    "source_artifacts": [state_source, table_sources["clean"], summary["source"]],
                    "population": f"{activity_class}: {summary['rule']}; {summary['denominator']} molecules",
                    "counting_method": SCAFFOLD_METHOD
                    + " Verify exact SMILES/activity row multiset against clean threshold subset.",
                    "corrected_value": summary,
                    "status": "verified",
                    "evidence_kind": "reconstructed from original saved activity subset; not a node-membership replay",
                    "originals_preserved": True,
                }
            )
        ledger.append(
            {
                "run": run_id,
                "claim_type": "search_matched_assays",
                "original_claim": f"Original ds_001.assay_count = {metadata_count}",
                "source_artifacts": [state_source, retrieval_code],
                "population": "search-matched assay IDs before activity retrieval",
                "counting_method": "Original fetch_compounds builds all_assay_ids from search results, then total_assays=len(all_assay_ids).",
                "corrected_value": result["search_assay_metadata"],
                "status": "unresolved",
                "evidence_kind": "original metadata scope verified; numeric ID set not independently retained",
                "originals_preserved": True,
            }
        )
        ledger.append(
            {
                "run": run_id,
                "claim_type": "whole_dataset_exact_murcko",
                "original_claim": evidence_lines(response, r"scaffold|adamant|urea"),
                "source_artifacts": common_sources,
                "population": f"all {len(tables['clean'])} clean compound rows",
                "counting_method": SCAFFOLD_METHOD,
                "corrected_value": result["scaffolds"],
                "status": "verified",
                "evidence_kind": "reconstructed",
                "originals_preserved": True,
            }
        )
        for call in calls:
            ledger.append(
                {
                    "run": run_id,
                    "claim_type": "selected_node_scaffolds",
                    "original_claim": {
                        "tool_call_index": call["tool_call_index"],
                        "counts": call["recorded_counts"],
                    },
                    "source_artifacts": [bundle_source, scaffold_code],
                    "population": {
                        "selected_nodes": call["recorded_node_selection"],
                        "denominator": call["subset_denominator"],
                    },
                    "counting_method": "Original tool selects source_mols.node_index membership then computes Murcko scaffold frequencies.",
                    "corrected_value": call,
                    "status": call["subset_recomputation_status"],
                    "evidence_kind": "original subset result; scope recovered; whole-dataset extrapolation invalid",
                    "originals_preserved": True,
                }
            )
    if len({r["run"] for r in records}) != 6:
        raise ValueError("Duplicate or missing run identities")
    for path, digest in sources.items():
        if sha256(path) != digest:
            raise ValueError(f"Input changed during reconciliation: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        "reconciliation.json",
        "correction_ledger.json",
        "murcko_scaffold_frequencies.csv",
        "README.md",
    }
    if any(str((output_dir / name).resolve()) in sources for name in output_names):
        raise ValueError("Correction output would overwrite original evidence")
    payload = {
        "schema_version": 1,
        "scope": "six sEH analysis reports only; no model inference, retrieval, rescoring or original-artifact edits",
        "rdkit_version": rdBase.rdkitVersion,
        "pandas_version": pd.__version__,
        "original_source_sha256": sources,
        "inputs_unchanged_after_reconciliation": True,
        "script_sha256": sha256(__file__),
        "runs": records,
    }
    (output_dir / "reconciliation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "correction_ledger.json").write_text(
        json.dumps(ledger, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "murcko_scaffold_frequencies.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["run", "population", "denominator", "scaffold_smiles", "count"]
        )
        writer.writeheader()
        writer.writerows(scaffold_rows)
    (output_dir / "README.md").write_text(render_addendum(records, audit_dir, source_root))
    return payload


def render_addendum(records, audit_dir, source_root):
    text = [
        "# Historical sEH reporting corrections",
        "",
        "This additive, post-hoc correction covers the six retained sEH analysis runs. Original responses, reports, datasets, audit artifacts and task-success scores are preserved. Every run is independently recomputed and checked against the historical audit, with SHA-256 hashes in `reconciliation.json` and claim-level provenance in `correction_ledger.json`.",
        "",
        "## Reproduce offline",
        "",
        "```sh",
        "PYTHONPATH=src python scripts/reconcile_reporting_evidence.py \\",
        f"  --audit-dir {audit_dir} \\",
        f"  --source-root {source_root} \\",
        "  --output-dir docs/manuscript/results/reporting-corrections",
        "```",
        "",
        "The original artifacts must be available at the recorded paths. This command reads them and writes only this correction directory. No ChEMBL retrieval, model inference, GTM projection, or benchmark rescoring is performed. Python, pandas and RDKit are required; versions and the reconciliation script hash are recorded in the JSON.",
        "",
        "## Counting rules",
        "",
        "Retained raw means activity rows that survived retrieval filtering. Full retrieved means the disjoint union of retained raw and excluded rows, validated with activity IDs. Clean means standardized compound rows. Identifier counts split pipe-joined cells and count distinct nonempty atomic IDs. Distinct serialized combinations are reported only to explain the earlier error.",
        "",
        SCAFFOLD_METHOD
        + " Invalid structures are counted separately and never treated as acyclic. All six historical clean datasets have zero invalid structures under this replay.",
        "",
        "| Run | Full rows | Full assays/docs | Retained rows | Retained assays/docs | Clean compounds | Clean assays/docs | Exact Murcko scaffolds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        f, a, c = r["full_retrieved"], r["retained"], r["clean"]
        text.append(
            f"| {r['run']} | {f['rows']} | {f['distinct_assays']}/{f['distinct_documents']} | {a['rows']} | {a['distinct_assays']}/{a['distinct_documents']} | {c['rows']} | {c['distinct_assays']}/{c['distinct_documents']} | {r['scaffolds']['unique_scaffolds_including_acyclic']} |"
        )
    text.extend(
        [
            "",
            "The original `ds_001.assay_count=525` is traced to `fetch_compounds`: it unions search-matched assay IDs before fetching activity rows. That scope is verified from the retained source code and state in each run. The original matched-ID lists were not retained, so 525 cannot be independently recounted. It is neither the 508 assays represented in retrieved activities nor retained coverage. Single-agent repetition 1's 237 documents, repetition 2's 167 assays / 129 documents, and repetition 3's 129 documents trace to stored `nunique` calls on pipe-joined clean cells. Recomputing these serialized combinations reproduces those numbers; splitting their atomic IDs gives the corrected coverage above.",
            "",
        ]
    )
    for r in records:
        c, s = r["clean"], r["scaffolds"]
        text.extend(
            [
                f"## {r['run']}",
                "",
                f"Original task success remains `{str(r['original_task_success']).lower()}`. Coverage: {r['retained']['distinct_assays']} assays / {r['retained']['distinct_documents']} documents in retained activity rows; {c['distinct_assays']} / {c['distinct_documents']} atomic IDs in {c['rows']} clean compounds. Clean-cell serialization has {c['serialized_assay_combinations']} assay combinations / {c['serialized_document_combinations']} document combinations; these are not assay/document counts.",
                "",
                f"Whole-set exact Murcko frequencies: adamantane **{s['adamantane']}/{c['rows']}**; phenyl-urea-adamantyl **{s['phenyl_urea_adamantyl']}/{c['rows']}**; acyclic **{s['acyclic_molecules']}/{c['rows']}**; **{s['unique_scaffolds_including_acyclic']}** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.",
                "",
            ]
        )
        if r["arm"] == "team" and r["repetition"] == 1:
            text.append(
                "The original report explicitly used approximate substructure-defined chemotypes, not exact Murcko scaffolds. Its four motif categories must not be interpreted as four unique scaffolds; the exact Murcko enumeration here is a separate measurement. Substructure prevalence claims are not revalidated by exact scaffold frequencies."
            )
        if r["arm"] == "team" and r["repetition"] == 2:
            text.append(
                "The original state labels 1,563 molecules as scaffold-analyzed, while its summary frequencies match the full clean population. The explicit denominator for this correction is all 1,580 valid clean structures; the truncated original node selection cannot independently verify the 1,563 scope."
            )
        for activity_class, summary in r["original_saved_activity_subsets"].items():
            text.append(
                f"Saved {activity_class} subset verified independently: **{summary['denominator']}** compounds satisfy `{summary['rule']}` in the original clean table, with **{summary['unique_scaffolds_including_acyclic']}** exact scaffolds. Adamantane={summary['counts'].get(ADAMANTANE, 0)}; phenyl-urea-adamantyl={summary['counts'].get(PHENYL_UREA_ADAMANTYL, 0)}. Missing activity is excluded. This verifies the activity-defined population without recovering original node membership."
            )
        for call in r["scaffold_calls"]:
            a = call["recorded_counts"].get(ADAMANTANE)
            p = call["recorded_counts"].get(PHENYL_UREA_ADAMANTYL)
            nodes = call["recorded_node_selection"]
            node_text = (
                str(nodes)
                if call["node_selection_complete"]
                else f"truncated original node list ({nodes[:3]} …); full selection unresolved"
            )
            text.append(
                f"Original scaffold call {call['tool_call_index']} selected nodes {node_text}. Its stored preview reports adamantane={a}, phenyl-urea-adamantyl={p} (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **{call['subset_recomputation_status']}**; denominator={call['subset_denominator']}. {call['reason']}"
            )
            text.append("")
        if r["arm"] == "single_agent":
            text.append(
                "The scaffold numbers in the response match the first selected-node call above. They must be labeled as selected-region observations, not whole-dataset frequencies. The whole-dataset ranking is shown above."
            )
            if r["repetition"] == 3:
                text.append(
                    "The response also calls the 42-count scaffold most frequent although its same stored preview has adamantane at 55."
                )
        text.extend(
            [
                "",
                "Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.",
                "",
            ]
        )
    text.extend(
        [
            "## Implications and remaining limits",
            "",
            "These corrections reduce the reported breadth of retained assay/document coverage and restrict selected-node scaffold conclusions to their sampled regions. The retained sets still contain many exact scaffolds; local adamantyl-urea enrichment does not establish that one series represents the entire dataset. Corresponding team/single-agent repetitions have the same recomputed coverage and whole-set scaffold counts, so these metrics do not support a between-architecture scientific advantage.",
            "",
            "Mixed endpoints, filtering differences between repetitions, GTM density and node activity remain exploratory context. The reporting correction does not establish improved potency, independent lead discovery, causal SAR, or experimental validation. Generation, retrosynthesis, peptide analyses and noncoverage/non-scaffold narrative claims are outside this correction's scope. Original benchmark outcomes remain unchanged; no replacement success percentage is assigned.",
            "",
            "Selected-node scope is recoverable from the stored tool calls, but the single-agent runs retain no CSV with molecule-level node membership. Some result previews and node lists are truncated, so complete subset denominators and scaffold distributions cannot be independently verified without additional original artifacts. Reprojecting would produce reconstructed evidence and is deliberately not substituted for missing historical membership. Inventory and truncation flags are in the JSON. The original search-matched 525 assay IDs remain unresolved at the ID level.",
            "",
        ]
    )
    return "\n".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = reconcile(args.audit_dir, args.source_root, args.output_dir)
    print(
        f"Reconciled {len(result['runs'])} runs; verified {len(result['original_source_sha256'])} original source hashes."
    )


if __name__ == "__main__":
    main()
