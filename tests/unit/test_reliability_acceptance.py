"""Adversarial and successful scientific acceptance examples (no model calls)."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[1] / "robustness"))
from reliability.validators import _SEH_PARENT, evaluate_run  # noqa: E402


def output(state=None, **kwargs):
    return {
        "status": "success",
        "response": "",
        "session_state": state or {},
        "generated_files": {},
        "telemetry": {"tool_calls": []},
        **kwargs,
    }


def call(name, *, error=False, result=""):
    return {"tool_name": name, "error": error, "result_preview": result}


def write_json(path, payload):
    path.write_text(json.dumps(payload))
    return str(path)


def molecular_output(tmp_path, smiles=("CCO", "CCN")):
    path = write_json(
        tmp_path / "candidates.json",
        {
            "metadata": {"seed_smiles": _SEH_PARENT, "seed_compound_id": "CHEMBL3327073"},
            "candidates": [{"smiles": value} for value in smiles],
        },
    )
    state = {"session_objects": {"candidate_sets": {"set1": {"artifact_path": path}}}}
    return output(
        state,
        prompt="Generate 2 analogues of CHEMBL3327073.",
        telemetry={"tool_calls": [call("generate_analogs")]},
    )


def failures(result):
    return {
        check["name"]
        for check in result["checks"]
        if not check["passed"] and check["severity"] == "required"
    }


def test_counts_pointers_and_preview_are_not_valid_molecules(tmp_path):
    item = {
        "count_returned": 100,
        "seed_smiles": _SEH_PARENT,
        "seed_compound_id": "CHEMBL3327073",
        "artifact_path": str(tmp_path / "absent.json"),
        "preview": [{"smiles": "CCO"}],
    }
    run = output(
        {"session_objects": {"candidate_sets": {"set1": item}}},
        prompt="Generate 10 analogues of CHEMBL3327073.",
        telemetry={"tool_calls": [call("generate_analogs")]},
    )
    assert "valid_candidates_returned" in failures(evaluate_run("molecular_generation", run))


def test_parse_candidate_structures_and_enforce_requested_count(tmp_path):
    run = molecular_output(tmp_path, ("CCO", "not-a-smiles"))
    result = evaluate_run("molecular_generation", run)
    assert "valid_candidates_returned" in failures(result)
    evidence = next(
        check["evidence"]
        for check in result["checks"]
        if check["name"] == "valid_candidates_returned"
    )
    assert "'raw_artifact_count': 2" in evidence
    assert "'valid_count': 1" in evidence
    run = molecular_output(tmp_path)
    assert evaluate_run("molecular_generation", run)["task_success"]


def test_provenance_must_match_named_parent_not_arbitrary_seed(tmp_path):
    run = molecular_output(tmp_path)
    path = Path(run["session_state"]["session_objects"]["candidate_sets"]["set1"]["artifact_path"])
    payload = json.loads(path.read_text())
    payload["metadata"]["seed_smiles"] = "CCO"
    write_json(path, payload)
    assert "parent_provenance_present" in failures(evaluate_run("molecular_generation", run))


def test_inherited_candidates_are_not_new_generation(tmp_path):
    run = molecular_output(tmp_path)
    run["initial_session_state"] = copy.deepcopy(run["session_state"])
    path = run["session_state"]["session_objects"]["candidate_sets"]["set1"]["artifact_path"]
    run["artifact_baseline"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    assert not evaluate_run("molecular_generation", run)["task_success"]


def test_recovered_tool_error_does_not_override_completed_task(tmp_path):
    run = molecular_output(tmp_path)
    run["telemetry"]["tool_calls"].insert(0, call("generate_analogs", error=True))
    result = evaluate_run("molecular_generation", run)
    assert result["task_success"]
    diagnostic = next(
        check for check in result["checks"] if check["name"] == "no_failed_tool_calls"
    )
    assert diagnostic["severity"] == "diagnostic"
    assert not diagnostic["passed"]
    assert "tool_exception" in result["failure_categories"]


def test_gtm_keyword_and_successful_attempt_are_not_map_evidence():
    state = {
        "gtm_path": "missing.pkl",
        "session_objects": {
            "maps": {"map1": {}},
            "figures": {"fig1": {}, "fig2": {}},
            "reports": {"report1": {"path": "absent.md"}},
        },
    }
    run = output(
        state,
        response="Assays, active and inactive scaffolds and density activity.",
        telemetry={"tool_calls": [call("gtm_optimization"), call("save_rich_report")]},
    )
    result = evaluate_run("seh_analysis", run)
    assert {"gtm_map_registered", "density_and_activity_evidence", "report_available"} <= failures(
        result
    )


def seh_output(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "assay_chembl_id": ["CHEMBL123", "CHEMBL124"],
            "activity": ["active", "inactive"],
        }
    ).to_csv(dataset, index=False)
    descriptors = tmp_path / "descriptors.parquet"
    pd.DataFrame(
        {"smiles": ["CCO", "CCN"], "autoencoder_embedding": [[0.2, 0.8], [0.3, 0.7]]}
    ).to_parquet(descriptors)
    density = tmp_path / "density.csv"
    pd.DataFrame({"nodes": [0, 1], "density": [0.6, 0.4]}).to_csv(density, index=False)
    activity = tmp_path / "activity.csv"
    pd.DataFrame({"nodes": [0, 1], "active_prob": [0.9, 0.2], "inactive_prob": [0.1, 0.8]}).to_csv(
        activity, index=False
    )
    model = tmp_path / "map.pkl"
    model.write_bytes(b"cchemographykit.gtm\nGTM\n.")  # Type-marker fixture; never unpickled.
    report = tmp_path / "report.md"
    report.write_text(
        "# sEH chemical space\nThe dataset contains 2 compounds from assays CHEMBL123 and CHEMBL124. "
        "The density and activity landscapes separate active and inactive compounds. "
        "Scaffold frequencies were inspected in populated nodes.\n"
    )
    state = {
        "data_file_paths": {
            "dataset_path": str(dataset),
            "descriptor_parquet_path": str(descriptors),
            "density_path": str(density),
            "activity_path": str(activity),
        },
        "session_objects": {
            "maps": {"map1": {"model_path": str(model)}},
            "reports": {"report1": {"paths": {"Markdown": str(report)}}},
        },
    }
    return output(
        state,
        tier="frozen",
        response="The report covers 2 compounds and their assays, active and inactive classes.",
        telemetry={
            "tool_calls": [
                call("analyze_scaffolds_in_nodes", result="scaffold_smi count\nC1CCCCC1 2")
            ]
        },
    )


def test_readable_gtm_outputs_and_grounded_report_pass(tmp_path):
    run = seh_output(tmp_path)
    result = evaluate_run("seh_analysis", run)
    assert result["task_success"], result


def test_frozen_complete_analysis_does_not_count_as_new_work(tmp_path):
    run = seh_output(tmp_path)
    run["initial_session_state"] = copy.deepcopy(run["session_state"])
    run["artifact_baseline"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()
    }
    result = evaluate_run("seh_analysis", run)
    assert {"density_and_activity_evidence", "report_available"} <= failures(result)


@pytest.mark.parametrize(
    "attempt", [{"stop_reason": "error", "error": "model unavailable"}, {"max_iterations": 100}]
)
def test_failed_or_unexecuted_search_is_not_a_valid_no_route(attempt):
    run = output(
        {"synplanner_plan": {"smiles": "CCO", "routes": [], "attempts": [attempt]}},
        response="No route was found.",
        telemetry={"tool_calls": [call("plan_synthesis")]},
    )
    assert not evaluate_run("retrosynthesis", run)["task_success"]


def test_completed_no_route_search_and_target_match(tmp_path):
    run = molecular_output(tmp_path)
    run["prompt"] = "Run SynPlanner for the first valid generated analogue."
    run["session_state"]["synplanner_plan"] = {
        "smiles": "CCO",
        "routes": [],
        "attempts": [{"stop_reason": "no_routes", "route_count": 0, "iterations": 100}],
    }
    run["response"] = "SynPlanner completed the search: no route was found."
    run["telemetry"]["tool_calls"] = [call("plan_synthesis")]
    assert evaluate_run("retrosynthesis", run)["task_success"]
    run["session_state"]["synplanner_plan"]["smiles"] = "CCC"
    assert "target_structure_resolved" in failures(evaluate_run("retrosynthesis", run))


def test_peptide_counts_and_keyword_report_do_not_pass(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("Similarity and uniqueness are good. " * 5)
    run = output(
        {
            "designed_peptides": {"count_returned": 10},
            "session_objects": {
                "reports": {"report1": {"path": str(report)}},
                "figures": {"logo": {"path": str(tmp_path / "logo.png")}},
            },
        },
        response="The sequences are similar and unique.",
        telemetry={"tool_calls": [call("create_peptide_activity_landscapes")]},
    )
    assert {
        "peptide_candidates_available",
        "similarity_and_uniqueness_analyzed",
        "sequence_logo_available",
    } <= failures(evaluate_run("peptide_design", run))


def test_peptide_artifact_sequences_metrics_logo_and_report_pass(tmp_path):
    candidate_path = write_json(
        tmp_path / "peptides.json",
        {"candidates": [{"sequence": "K L L K"}, {"sequence": "R L L R"}]},
    )
    activity = tmp_path / "activity.csv"
    pd.DataFrame({"nodes": [0, 1], "active_prob": [0.8, 0.1]}).to_csv(activity, index=False)
    logo = tmp_path / "logo.svg"
    logo.write_text('<svg xmlns="http://www.w3.org/2000/svg"><text>KLLK</text></svg>')
    report = tmp_path / "report.md"
    report.write_text(
        "# Peptide report\nGenerated KLLK and RLLR from active regions of the E. coli landscape. "
        "Pairwise similarity: 0.5. Unique sequence fraction: 1.0. The sequence logo is attached.\n"
    )
    run = output(
        {
            "designed_peptides": {
                "peptide_candidate_set_id": "pep1",
                "artifact_path": candidate_path,
            },
            "data_file_paths": {"activity_path": str(activity), "logo_path": str(logo)},
            "session_objects": {"reports": {"report1": {"path": str(report)}}},
        }
    )
    assert evaluate_run("peptide_design", run)["task_success"]


def test_duplicate_smiles_do_not_inflate_requested_candidate_count(tmp_path):
    run = molecular_output(tmp_path, ("CCO", "C(C)O"))
    assert "valid_candidates_returned" in failures(evaluate_run("molecular_generation", run))


def test_wrong_generation_engine_does_not_satisfy_autoencoder_request(tmp_path):
    run = molecular_output(tmp_path)
    run["prompt"] = "Generate 2 analogues of CHEMBL3327073 with the autoencoder."
    assert "requested_generation_engine_used" in failures(evaluate_run("molecular_generation", run))


def test_non_model_file_cannot_satisfy_gtm_evidence(tmp_path):
    run = seh_output(tmp_path)
    (tmp_path / "map.pkl").write_text("GTM optimization completed successfully")
    assert "gtm_map_registered" in failures(evaluate_run("seh_analysis", run))


def test_separate_descriptor_table_and_large_model_are_supported(tmp_path):
    run = seh_output(tmp_path)
    pd.DataFrame({"autoencoder_embedding": [[0.2, 0.8], [0.3, 0.7]]}).to_parquet(
        tmp_path / "descriptors.parquet"
    )
    with (tmp_path / "map.pkl").open("ab") as handle:
        handle.truncate(70 * 1024 * 1024)
    assert evaluate_run("seh_analysis", run)["task_success"]


def test_unavailable_telemetry_is_not_error_free_execution():
    result = evaluate_run("execution_only", output(telemetry_status="unavailable"))
    diagnostic = next(
        check for check in result["checks"] if check["name"] == "no_failed_tool_calls"
    )
    assert diagnostic["passed"] is False
    assert diagnostic["severity"] == "diagnostic"
    assert result["task_success"] is True


def test_same_content_recomputed_with_writer_evidence_is_new_output(tmp_path):
    run = seh_output(tmp_path)
    run["initial_session_state"] = copy.deepcopy(run["session_state"])
    run["artifact_baseline"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in tmp_path.iterdir()
    }
    run["artifact_baseline_mtime_ns"] = {
        str(path): path.stat().st_mtime_ns - 1 for path in tmp_path.iterdir()
    }
    assert not evaluate_run("seh_analysis", run)["task_success"]
    run["telemetry"]["tool_calls"].extend(
        [call("create_activity_landscapes"), call("save_rich_report")]
    )
    assert evaluate_run("seh_analysis", run)["task_success"]


def test_pdf_report_text_is_supported(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import patch

    run = seh_output(tmp_path)
    text = (tmp_path / "report.md").read_text()
    (tmp_path / "report.md").unlink()
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(b"%PDF-report")
    run["session_state"]["session_objects"]["reports"]["report1"]["paths"] = {"PDF": str(pdf)}
    with patch(
        "pypdf.PdfReader",
        return_value=SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: text)]),
    ):
        assert evaluate_run("seh_analysis", run)["task_success"]


@pytest.mark.parametrize(
    "status,passed", [("complete", True), ("partial", False), ("unavailable", False)]
)
def test_error_free_diagnostic_uses_nested_runner_telemetry(status, passed):
    run = output(
        telemetry={"telemetry_status": status, "tool_calls": []},
        telemetry_status="complete" if status != "complete" else "unavailable",
    )
    result = evaluate_run("execution_only", run)
    check = next(item for item in result["checks"] if item["name"] == "no_failed_tool_calls")
    assert check["passed"] is passed
    assert f"telemetry={status}" in check["evidence"]


def test_error_free_diagnostic_retains_top_level_compatibility():
    result = evaluate_run("execution_only", output(telemetry_status="complete"))
    check = next(item for item in result["checks"] if item["name"] == "no_failed_tool_calls")
    assert check["passed"] is True


def test_all_manuscript_generation_prompts_require_ten_candidates(tmp_path):
    config_path = Path(__file__).parents[1] / "robustness" / "manuscript_reliability.yaml"
    config = yaml.safe_load(config_path.read_text())
    cases = config["tests"]
    prompts = list(cases["frozen_case_2_seh_generation"]["prompt_variants"])
    prompts.extend(
        step["prompt"]
        for step in cases["live_seh_workflow"]["steps"]
        if step.get("validator") == "molecular_generation"
    )
    assert len(prompts) == 4
    run = molecular_output(tmp_path)
    for prompt in prompts:
        run["prompt"] = prompt
        result = evaluate_run("molecular_generation", run)
        assert result["scientific_outcome"]["requested_candidate_count"] == 10, prompt
        assert "valid_candidates_returned" in failures(result), prompt
