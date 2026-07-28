"""Objective acceptance validators for the manuscript reliability benchmark."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

from .models import ValidationResult

Validator = Callable[[Mapping[str, Any]], List[ValidationResult]]


def _check(
    name: str,
    passed: bool,
    evidence: str,
    *,
    category: str | None = None,
    severity: str = "required",
) -> ValidationResult:
    return ValidationResult(
        name=name,
        passed=bool(passed),
        evidence=evidence,
        category=category if not passed else None,
        severity=severity,
    )


def _state(output: Mapping[str, Any]) -> Dict[str, Any]:
    state = output.get("session_state")
    return state if isinstance(state, dict) else {}


def _response(output: Mapping[str, Any]) -> str:
    return str(output.get("response") or "")


def _tool_calls(output: Mapping[str, Any]) -> List[Dict[str, Any]]:
    telemetry = output.get("telemetry")
    if isinstance(telemetry, dict):
        calls = telemetry.get("tool_calls")
        if isinstance(calls, list):
            return [call for call in calls if isinstance(call, dict)]
    calls = output.get("tool_calls")
    return [call for call in calls or [] if isinstance(call, dict)]


def _tool_names(output: Mapping[str, Any]) -> List[str]:
    return [str(call.get("tool_name") or "") for call in _tool_calls(output)]


def _has_tool(output: Mapping[str, Any], names: Sequence[str]) -> bool:
    actual = _tool_names(output)
    return any(name in actual for name in names)


def _memory_collection(output: Mapping[str, Any], collection: str) -> List[Dict[str, Any]]:
    memory = _state(output).get("session_objects")
    if not isinstance(memory, dict):
        return []
    records = memory.get(collection)
    if not isinstance(records, dict):
        return []
    return [record for record in records.values() if isinstance(record, dict)]


def _has_path(output: Mapping[str, Any], keys: Sequence[str]) -> bool:
    state = _state(output)
    paths = state.get("data_file_paths")
    path_values = paths if isinstance(paths, dict) else {}
    generated = output.get("generated_files")
    generated_values = generated if isinstance(generated, dict) else {}
    for key in keys:
        if path_values.get(key):
            return True
        if any(key in generated_key for generated_key in generated_values):
            return True
    return False


def _has_named_artifact(output: Mapping[str, Any], token: str) -> bool:
    generated = output.get("generated_files")
    if not isinstance(generated, dict):
        return False
    token_lower = token.lower()
    return any(
        token_lower in str(key).lower() or token_lower in str(value).lower()
        for key, value in generated.items()
    )


def _execution_checks(output: Mapping[str, Any]) -> List[ValidationResult]:
    status = str(output.get("status") or "unknown")
    checks = [
        _check(
            "execution_completed",
            status == "success",
            f"execution status={status}",
            category=(
                "timeout"
                if status == "timeout"
                else (
                    "fixture_failure"
                    if status == "fixture_error"
                    else "agent_exception" if status == "failed" else "execution_failure"
                )
            ),
        )
    ]
    failed_tools = [call for call in _tool_calls(output) if call.get("error")]
    checks.append(
        _check(
            "no_failed_tool_calls",
            not failed_tools,
            (
                "no structured tool failures"
                if not failed_tools
                else "failed tools: "
                + ", ".join(str(call.get("tool_name")) for call in failed_tools)
            ),
            category="tool_exception",
        )
    )
    return checks


def validate_execution_only(output: Mapping[str, Any]) -> List[ValidationResult]:
    return _execution_checks(output)


def validate_seh_analysis(output: Mapping[str, Any]) -> List[ValidationResult]:
    checks = _execution_checks(output)
    datasets = _memory_collection(output, "datasets")
    maps = _memory_collection(output, "maps")
    figures = _memory_collection(output, "figures")
    analyses = _memory_collection(output, "analyses")
    reports = _memory_collection(output, "reports")
    response = _response(output).lower()

    checks.extend(
        [
            _check(
                "seh_data_source_resolved",
                (
                    _has_tool(output, ("fetch_compounds",))
                    if str(output.get("tier") or "") == "live"
                    else bool(datasets) or _has_path(output, ("clean_dataset_path", "dataset_path"))
                ),
                (
                    "live tier requires ChEMBL retrieval; frozen tier requires "
                    "a registered fixture dataset"
                ),
                category="missing_required_call",
            ),
            _check(
                "clean_dataset_registered",
                bool(datasets) or _has_path(output, ("clean_dataset_path", "dataset_path")),
                f"datasets={len(datasets)}",
                category="missing_artifact",
            ),
            _check(
                "descriptor_artifact_registered",
                _has_path(output, ("descriptor_parquet_path",))
                or any(record.get("descriptor_parquet_path") for record in datasets),
                "descriptor pointer present in dataset state",
                category="missing_artifact",
            ),
            _check(
                "gtm_map_registered",
                bool(maps)
                or any("gtm" in str(key).lower() for key in _state(output))
                or _has_tool(output, ("gtm_optimization", "load_gtm_model_only")),
                f"maps={len(maps)}",
                category="missing_artifact",
            ),
            _check(
                "density_and_activity_evidence",
                (
                    len(figures) >= 2
                    or (
                        _has_tool(output, ("get_density_summary", "load_gtm_get_density_matrix"))
                        and _has_tool(
                            output,
                            ("create_activity_landscapes", "get_activity_landscape_summary"),
                        )
                    )
                ),
                f"figures={len(figures)}",
                category="missing_task_requirement",
            ),
            _check(
                "chemotype_or_scaffold_analysis",
                _has_tool(output, ("analyze_scaffolds_in_nodes",))
                or any(
                    token in str(record.get("analysis_type", "")).lower()
                    for record in analyses
                    for token in ("scaffold", "chemotype", "sar")
                )
                or any(token in response for token in ("chemotype", "scaffold")),
                f"analyses={len(analyses)}",
                category="missing_task_requirement",
            ),
            _check(
                "assay_and_class_separation_reported",
                "assay" in response
                and (
                    ("active" in response and "inactive" in response)
                    or "class separation" in response
                ),
                "response contains assay and active/inactive evidence",
                category="missing_task_requirement",
            ),
            _check(
                "report_available",
                bool(reports)
                or _has_path(output, ("report_path", "markdown_path", "html_path"))
                or _has_tool(output, ("save_markdown_report", "save_rich_report")),
                f"reports={len(reports)}",
                category="missing_artifact",
            ),
        ]
    )
    return checks


def _molecular_candidates(output: Mapping[str, Any]) -> List[Dict[str, Any]]:
    candidates = _memory_collection(output, "candidate_sets")
    if candidates:
        return candidates
    state = _state(output)
    pointers = []
    for key in ("designed_molecules", "generated_molecules", "sampled_molecules"):
        value = state.get(key)
        if isinstance(value, dict):
            pointers.append(value)
    return pointers


def validate_molecular_generation(output: Mapping[str, Any]) -> List[ValidationResult]:
    checks = _execution_checks(output)
    candidates = _molecular_candidates(output)
    counts = [
        int(
            item.get("count_returned")
            or item.get("artifact_count")
            or item.get("count")
            or len(item.get("compound_ids") or [])
            or len(item.get("candidates") or [])
        )
        for item in candidates
    ]
    tools = _tool_names(output)
    checks.extend(
        [
            _check(
                "molecular_design_tool_used",
                any(
                    name in tools
                    for name in (
                        "design_molecules",
                        "generate_analogs",
                        "sample_activity_landscape_nodes",
                        "sample_molecules",
                    )
                ),
                f"tools={tools}",
                category="missing_required_call",
            ),
            _check(
                "candidate_set_registered",
                bool(candidates),
                f"candidate_sets={len(candidates)}",
                category="missing_artifact",
            ),
            _check(
                "valid_candidates_returned",
                any(count > 0 for count in counts) or bool(output.get("smiles_generated")),
                f"returned_counts={counts}",
                category="missing_scientific_outcome",
            ),
            _check(
                "parent_provenance_present",
                "chembl3327073" in str(output.get("prompt", "")).lower()
                and any(
                    item.get("seed_smiles") or item.get("seed_compound_id") or item.get("parent")
                    for item in candidates
                ),
                "prompt identifies CHEMBL3327073 and candidate metadata stores a seed",
                category="missing_provenance",
            ),
        ]
    )
    return checks


def validate_retrosynthesis(output: Mapping[str, Any]) -> List[ValidationResult]:
    checks = _execution_checks(output)
    state = _state(output)
    plan = state.get("synplanner_plan")
    plan = plan if isinstance(plan, dict) else {}
    routes = plan.get("routes") if isinstance(plan.get("routes"), list) else []
    attempts = plan.get("attempts") if isinstance(plan.get("attempts"), list) else []
    route_records = _memory_collection(output, "routes")
    response = _response(output).lower()
    no_route_reported = "no route" in response or "did not return" in response

    checks.extend(
        [
            _check(
                "synplanner_search_executed",
                _has_tool(output, ("plan_synthesis",))
                or bool(attempts)
                or bool(routes)
                or bool(route_records),
                f"attempts={len(attempts)}, routes={len(routes) or len(route_records)}",
                category="missing_required_call",
            ),
            _check(
                "target_structure_resolved",
                bool(plan.get("smiles"))
                or any(record.get("target_smiles") for record in route_records)
                or _has_tool(output, ("identify_input", "convert_name_to_smiles")),
                "target SMILES is present in plan/route state",
                category="missing_provenance",
            ),
            _check(
                "route_outcome_reported",
                bool(routes) or bool(route_records) or no_route_reported,
                (
                    f"route_found={bool(routes or route_records)}"
                    if routes or route_records
                    else f"no_route_reported={no_route_reported}"
                ),
                category="missing_scientific_outcome",
            ),
        ]
    )
    return checks


def validate_peptide_design(output: Mapping[str, Any]) -> List[ValidationResult]:
    checks = _execution_checks(output)
    state = _state(output)
    analyses = _memory_collection(output, "analyses")
    figures = _memory_collection(output, "figures")
    reports = _memory_collection(output, "reports")
    peptide_pointers = [
        value
        for key, value in state.items()
        if isinstance(value, dict)
        and (
            any(
                token in str(key).lower()
                for token in ("designed_peptide", "generated_peptide", "peptide_candidate")
            )
            or value.get("count_returned")
        )
    ]
    response = _response(output).lower()
    tools = _tool_names(output)
    checks.extend(
        [
            _check(
                "peptide_activity_landscape_used",
                "create_peptide_activity_landscapes" in tools
                or any(
                    "activity" in str(record.get("figure_kind", "")).lower() for record in figures
                ),
                f"tools={tools}",
                category="missing_required_call",
            ),
            _check(
                "peptide_candidates_available",
                bool(peptide_pointers)
                or any(
                    record.get("analysis_type") == "peptide_design"
                    and int(record.get("count_returned") or 0) > 0
                    for record in analyses
                ),
                f"peptide_pointers={len(peptide_pointers)}",
                category="missing_artifact",
            ),
            _check(
                "similarity_and_uniqueness_analyzed",
                all(token in response for token in ("similar", "unique"))
                or any(
                    token in str(record.get("analysis_type", "")).lower()
                    for record in analyses
                    for token in ("similarity", "uniqueness")
                ),
                "response or analysis registry covers similarity and uniqueness",
                category="missing_task_requirement",
            ),
            _check(
                "sequence_logo_available",
                any(
                    "logo" in str(record.get("label", "")).lower()
                    or "logo" in str(record.get("path", "")).lower()
                    for record in figures
                )
                or _has_named_artifact(output, "logo"),
                f"figures={len(figures)}",
                category="missing_artifact",
            ),
            _check(
                "peptide_report_available",
                bool(reports) or _has_tool(output, ("save_markdown_report", "save_rich_report")),
                f"reports={len(reports)}",
                category="missing_artifact",
            ),
        ]
    )
    return checks


def _looks_like_clarification(response: str) -> bool:
    response_lower = response.lower()
    return "?" in response and any(
        token in response_lower
        for token in (
            "which",
            "what",
            "please specify",
            "clarify",
            "provide",
            "molecule",
            "peptide",
        )
    )


def _validate_recovery(
    output: Mapping[str, Any],
    *,
    name: str,
    forbidden_tools: Iterable[str],
    required_response: Callable[[str], bool],
) -> List[ValidationResult]:
    checks = _execution_checks(output)
    actual = set(_tool_names(output))
    forbidden = sorted(actual.intersection(forbidden_tools))
    response = _response(output)
    checks.extend(
        [
            _check(
                f"{name}_response",
                required_response(response),
                response[:300],
                category="missing_clarification",
            ),
            _check(
                "no_inappropriate_downstream_call",
                not forbidden,
                "no forbidden calls" if not forbidden else f"forbidden calls={forbidden}",
                category="incorrect_tool_selection",
            ),
            _check(
                "no_fabricated_artifacts",
                not output.get("generated_files"),
                f"generated_files={len(output.get('generated_files') or {})}",
                category="unsupported_claim",
            ),
        ]
    )
    return checks


def validate_clarification(output: Mapping[str, Any]) -> List[ValidationResult]:
    return _validate_recovery(
        output,
        name="clarification",
        forbidden_tools=(
            "fetch_compounds",
            "gtm_optimization",
            "design_molecules",
            "design_peptides",
            "plan_synthesis",
        ),
        required_response=_looks_like_clarification,
    )


def validate_missing_gtm(output: Mapping[str, Any]) -> List[ValidationResult]:
    return _validate_recovery(
        output,
        name="missing_gtm_prerequisite",
        forbidden_tools=("create_activity_landscapes", "save_gtm_landscape_plot"),
        required_response=lambda response: bool(
            re.search(r"(need|provide|load|build|missing|no).*(dataset|gtm|map)", response, re.I)
        ),
    )


def validate_missing_design_seed(output: Mapping[str, Any]) -> List[ValidationResult]:
    return _validate_recovery(
        output,
        name="missing_design_seed",
        forbidden_tools=("generate_analogs", "design_molecules"),
        required_response=_looks_like_clarification,
    )


def validate_invalid_retrosynthesis(output: Mapping[str, Any]) -> List[ValidationResult]:
    return _validate_recovery(
        output,
        name="invalid_retrosynthesis_input",
        forbidden_tools=("plan_synthesis",),
        required_response=lambda response: bool(
            re.search(
                r"(invalid|could not|cannot|not valid|provide).*(smiles|structure|molecule)",
                response,
                re.I,
            )
        ),
    )


_VALIDATORS: Dict[str, Validator] = {
    "execution_only": validate_execution_only,
    "seh_analysis": validate_seh_analysis,
    "molecular_generation": validate_molecular_generation,
    "retrosynthesis": validate_retrosynthesis,
    "peptide_design": validate_peptide_design,
    "clarification": validate_clarification,
    "missing_gtm": validate_missing_gtm,
    "missing_design_seed": validate_missing_design_seed,
    "invalid_retrosynthesis": validate_invalid_retrosynthesis,
}


def _frozen_tier_checks(
    validator_name: str,
    output: Mapping[str, Any],
) -> List[ValidationResult]:
    if str(output.get("tier") or "") != "frozen" or validator_name not in {
        "seh_analysis",
        "molecular_generation",
        "retrosynthesis",
        "peptide_design",
    }:
        return []
    forbidden = sorted(set(_tool_names(output)).intersection({"fetch_compounds"}))
    return [
        _check(
            "frozen_fixture_used_without_live_retrieval",
            not forbidden,
            "no live data retrieval" if not forbidden else f"forbidden calls={forbidden}",
            category="frozen_fixture_violation",
        )
    ]


def evaluate_run(validator_name: str, output: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate a run and return task success plus structured evidence."""
    if validator_name not in _VALIDATORS:
        raise ValueError(
            f"Unknown reliability validator '{validator_name}'. "
            f"Available validators: {', '.join(sorted(_VALIDATORS))}"
        )
    checks = _VALIDATORS[validator_name](output)
    checks.extend(_frozen_tier_checks(validator_name, output))
    required = [check for check in checks if check.severity == "required"]
    task_success = bool(required) and all(check.passed for check in required)
    failure_categories = sorted(
        {check.category for check in checks if not check.passed and check.category is not None}
    )
    scientific_outcome: Dict[str, Any] = {}
    if validator_name == "retrosynthesis":
        state = _state(output)
        plan = state.get("synplanner_plan")
        plan = plan if isinstance(plan, dict) else {}
        routes = plan.get("routes") if isinstance(plan.get("routes"), list) else []
        route_records = _memory_collection(output, "routes")
        scientific_outcome = {
            "route_found": bool(routes or route_records),
            "route_count": len(routes) or len(route_records),
        }
    return {
        "validator": validator_name,
        "task_success": task_success,
        "checks": [check.to_dict() for check in checks],
        "failure_categories": failure_categories,
        "scientific_outcome": scientific_outcome,
    }
