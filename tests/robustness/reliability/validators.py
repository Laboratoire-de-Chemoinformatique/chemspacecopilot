"""Content-based acceptance checks, not a substitute for the blinded human review.

Inputs may be reused; completed outputs must be new or changed during this step.
Pickled model artifacts are never deserialized by the evaluator.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping

from .models import ValidationResult

Validator = Callable[[Mapping[str, Any]], List[ValidationResult]]
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
_SEH_PARENT = "CCC(C)C(=O)N1CCC(NC(=O)Nc2ccc(C(F)(C(F)(F)F)C(F)(F)F)cc2)CC1"


def _check(name, passed, evidence, *, category=None, severity="required"):
    return ValidationResult(
        name=name,
        passed=bool(passed),
        evidence=evidence,
        category=category if not passed else None,
        severity=severity,
    )


def _state(output):
    state = output.get("session_state")
    return state if isinstance(state, dict) else {}


def _response(output):
    return str(output.get("response") or "")


def _tool_calls(output):
    telemetry = output.get("telemetry")
    calls = telemetry.get("tool_calls") if isinstance(telemetry, dict) else None
    calls = calls if isinstance(calls, list) else output.get("tool_calls", [])
    return [call for call in calls or [] if isinstance(call, dict)]


def _tool_names(output):
    return [str(call.get("tool_name") or "") for call in _tool_calls(output)]


def _has_tool(output, names):
    return any(
        call.get("tool_name") in names and not call.get("error") for call in _tool_calls(output)
    )


def _memory_collection(output, collection):
    memory = _state(output).get("session_objects")
    records = memory.get(collection, {}) if isinstance(memory, dict) else {}
    return (
        [record for record in records.values() if isinstance(record, dict)]
        if isinstance(records, dict)
        else []
    )


def _paths(value):
    """Find artifact pointers, including report paths nested under format names."""
    found = set()

    def visit(item, depth=0):
        if depth > 12:
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
        elif isinstance(item, str) and (
            item.startswith("s3://")
            or re.search(
                r"\.(csv|tsv|parquet|json|md|html|txt|png|svg|pdf|pkl|gz|fasta)$", item, re.I
            )
        ):
            found.add(item)

    visit(value)
    return found


def _rewritten(output, path):
    previous = (output.get("artifact_baseline_mtime_ns") or {}).get(path)
    if previous is None or path.startswith("s3://"):
        return False
    try:
        changed = Path(path).expanduser().stat().st_mtime_ns != previous
    except OSError:
        return False
    writers = (
        "save_",
        "create_",
        "generate_",
        "design_",
        "sample_",
        "gtm_optimization",
        "plan_synthesis",
        "load_and_prep_data",
        "load_gtm_get_density_matrix",
    )
    return changed and any(
        not call.get("error") and str(call.get("tool_name", "")).startswith(writers)
        for call in _tool_calls(output)
    )


def _artifact_bytes(output, path, *, fresh=False):
    if not isinstance(path, str) or path.startswith(("http://", "https://")):
        return None
    cache = output.get("_artifact_cache", {})
    if path not in cache:
        try:
            if path.startswith("s3://") or not Path(path).expanduser().is_file():
                from cs_copilot.storage import S3

                handle = S3.open(path, "rb")
            else:
                handle = Path(path).expanduser().open("rb")
            with handle as stream:
                data = stream.read(_MAX_ARTIFACT_BYTES + 1)
            cache[path] = (
                data if isinstance(data, bytes) and 0 < len(data) <= _MAX_ARTIFACT_BYTES else None
            )
        except Exception:
            cache[path] = None
    data = cache[path]
    if data and fresh:
        baseline = output.get("artifact_baseline") or {}
        previous = baseline.get(path)
        if previous is not None:
            if previous == hashlib.sha256(data).hexdigest() and not _rewritten(output, path):
                return None
        elif path in _paths(output.get("initial_session_state") or {}):
            # Unknown baseline cannot establish that an inherited output was rewritten.
            return None
    return data


def _new_records(output, collection):
    initial = {"session_state": output.get("initial_session_state") or {}}
    old = {
        json.dumps(record, sort_keys=True, default=str)
        for record in _memory_collection(initial, collection)
    }
    return [
        record
        for record in _memory_collection(output, collection)
        if json.dumps(record, sort_keys=True, default=str) not in old
        or any(_artifact_bytes(output, path, fresh=True) for path in _paths(record))
    ]


def _all_paths(output):
    return sorted(_paths(_state(output)) | _paths(output.get("generated_files") or {}))


def _table(output, path, *, fresh=False):
    if not str(path).lower().endswith((".csv", ".tsv", ".parquet")):
        return None
    data = _artifact_bytes(output, path, fresh=fresh)
    if not data:
        return None
    try:
        import pandas as pd

        if path.lower().endswith(".parquet"):
            return pd.read_parquet(io.BytesIO(data))
        if path.lower().endswith((".csv", ".tsv")):
            header = data.splitlines()[0] if data.splitlines() else b""
            return pd.read_csv(io.BytesIO(data), sep="\t" if b"\t" in header else ",")
    except Exception:
        pass
    return None


def _canonical_smiles(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        from rdkit import Chem, rdBase

        params = Chem.SmilesParserParams()
        params.parseName = False
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(value, params)
        return Chem.MolToSmiles(mol) if mol is not None and mol.GetNumAtoms() else None
    except (ValueError, TypeError):
        return None


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def _model_artifact(output, path):
    """Inspect only a GTM pickle's header; never execute it or read a large model fully.

    This supports model identity, not quality: numerical projection evidence is
    required separately. The tabular artifact size cap does not apply to models.
    """
    try:
        if path.startswith("s3://") or not Path(path).expanduser().is_file():
            from cs_copilot.storage import S3

            handle = S3.open(path, "rb")
        else:
            handle = Path(path).expanduser().open("rb")
        with handle as stream:
            if path.endswith(".gz"):
                with gzip.GzipFile(fileobj=stream) as zipped:
                    header = zipped.read(65536)
            else:
                header = stream.read(65536)
        return (
            header.startswith((b"\x80", b"c"))
            and b"GTM" in header
            and b"chemography" in header.lower()
        )
    except Exception:
        return False


def _numeric_table(table, columns):
    if table is None or table.empty:
        return False
    return any(
        any(_finite(value) for value in table[column])
        for column in table.columns
        if columns(str(column).lower())
    )


def _result_evidence(output, names):
    """A call attempt or a 'done' string is not a scientific result."""
    return [
        str(call.get("result_preview") or call.get("result") or "")
        for call in _tool_calls(output)
        if call.get("tool_name") in names and not call.get("error")
    ]


def _report_texts(output):
    paths = set()
    for record in _new_records(output, "reports"):
        paths.update(_paths(record))
    for path in _all_paths(output):
        if "report" in path.lower():
            paths.add(path)
    texts = []
    for path in sorted(paths):
        if not path.lower().endswith((".md", ".html", ".txt", ".pdf")):
            continue
        data = _artifact_bytes(output, path, fresh=True)
        if data:
            if path.lower().endswith(".pdf"):
                try:
                    from pypdf import PdfReader

                    text = "\n".join(
                        page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages
                    )
                except Exception:
                    continue
            else:
                text = data.decode("utf-8", errors="replace")
            if len(re.sub(r"<[^>]*>", "", text).strip()) >= 80:
                texts.append(text)
    return texts


def _execution_checks(output):
    status = str(output.get("status") or "unknown")
    categories = {
        "timeout": "timeout",
        "fixture_error": "fixture_failure",
        "prerequisite_error": "prerequisite_failure",
        "failed": "agent_exception",
    }
    calls = _tool_calls(output)
    failed = [call for call in calls if call.get("error")]
    telemetry = output.get("telemetry")
    telemetry_status = (
        telemetry.get("telemetry_status", output.get("telemetry_status", "unavailable"))
        if isinstance(telemetry, dict)
        else output.get("telemetry_status", "unavailable")
    )
    return [
        _check(
            "execution_completed",
            status == "success",
            f"execution status={status}",
            category=categories.get(status, "execution_failure"),
        ),
        _check(
            "no_failed_tool_calls",
            not failed and telemetry_status == "complete",
            f"observed failed calls={len(failed)}; telemetry={telemetry_status}",
            category="tool_exception" if failed else "telemetry_unavailable",
            severity="diagnostic",
        ),
    ]


def validate_execution_only(output):
    return _execution_checks(output)


def validate_seh_analysis(output):
    checks = _execution_checks(output)
    tables = [(path, _table(output, path)) for path in _all_paths(output)]
    datasets = [
        (path, table)
        for path, table in tables
        if table is not None
        and not table.empty
        and any(
            str(column).lower() in {"smiles", "smi", "canonical_smiles"} for column in table.columns
        )
    ]
    valid_datasets = []
    for path, table in datasets:
        col = next(
            col
            for col in table.columns
            if str(col).lower() in {"smiles", "smi", "canonical_smiles"}
        )
        if any(_canonical_smiles(value) for value in table[col]):
            valid_datasets.append((path, table))
    descriptors = False
    for _, table in tables:
        if table is None or table.empty:
            continue
        for col in table.columns:
            if any(
                token in str(col).lower()
                for token in ("fingerprint", "embedding", "descriptor_vector")
            ):
                for value in table[col]:
                    try:
                        vector = json.loads(value) if isinstance(value, str) else list(value)
                        descriptors |= len(vector) > 1 and all(_finite(item) for item in vector)
                    except (ValueError, TypeError):
                        continue
    new_tables = [(path, _table(output, path, fresh=True)) for path, _ in tables]
    density = any(
        _numeric_table(table, lambda col: col in {"density", "filtered_density"})
        for _, table in new_tables
    )
    density |= any(
        _finite(record.get("density")) or _finite(record.get("filtered_density"))
        for record in _new_records(output, "nodes")
    )
    activity = any(
        _numeric_table(
            table,
            lambda col: col in {"reg_density", "filtered_reg_density", "activity_density"}
            or col.endswith("_prob"),
        )
        for _, table in new_tables
    )
    model_paths = {
        path
        for record in _memory_collection(output, "maps")
        for path in _paths(record)
        if path.endswith((".pkl", ".pkl.gz"))
    }
    model_paths.update(path for path in _all_paths(output) if path.endswith((".pkl", ".pkl.gz")))
    model = any(_model_artifact(output, path) for path in model_paths)
    fitted = any(
        getattr(value, "weights", None) is not None and callable(getattr(value, "project", None))
        for value in _state(output).values()
    )
    scaffold_texts = _result_evidence(output, ("analyze_scaffolds_in_nodes",))
    scaffold_texts.extend(
        json.dumps(record, default=str) for record in _new_records(output, "analyses")
    )
    for _path, table in new_tables:
        if table is not None and any("scaffold" in str(col).lower() for col in table.columns):
            scaffold_texts.append(table.to_string())
    scaffold = any(
        re.search(r"scaffold|chemotype", text, re.I)
        and re.search(r"\d", text)
        and re.search(r"count|frequency|scaffold_smi|smiles", text, re.I)
        for text in scaffold_texts
    )
    reports = _report_texts(output)
    narrative = "\n".join([_response(output), *reports])
    known_ids = {
        str(value)
        for _, table in valid_datasets
        for col in table.columns
        if "assay" in str(col).lower()
        for value in table[col].dropna()
    }
    count_anchors = {str(len(table)) for _, table in valid_datasets}
    grounded = any(
        re.search(r"(?<!\d)" + re.escape(anchor) + r"(?!\d)", narrative)
        for anchor in known_ids | count_anchors
    )
    coverage = bool(
        re.search(r"assay", narrative, re.I)
        and re.search(r"active", narrative, re.I)
        and re.search(
            r"inactive|class separation|classification.*(unavailable|not possible)", narrative, re.I
        )
        and grounded
    )
    checks.extend(
        [
            _check(
                "seh_data_source_resolved",
                bool(valid_datasets)
                and (output.get("tier") != "live" or _has_tool(output, ("fetch_compounds",))),
                f"readable molecular datasets={len(valid_datasets)}",
                category="missing_artifact",
            ),
            _check(
                "clean_dataset_registered",
                bool(valid_datasets),
                f"datasets={len(valid_datasets)}",
                category="missing_artifact",
            ),
            _check(
                "descriptor_artifact_registered",
                descriptors,
                "readable finite descriptor vectors required",
                category="missing_artifact",
            ),
            _check(
                "gtm_map_registered",
                (model or fitted) and density and activity,
                f"model={bool(model or fitted)}, numerical density={density}, numerical activity={activity}",
                category="missing_artifact",
            ),
            _check(
                "density_and_activity_evidence",
                density and activity,
                "new numerical landscape evidence required",
                category="missing_task_requirement",
            ),
            _check(
                "chemotype_or_scaffold_analysis",
                scaffold,
                "scaffold table/count evidence required",
                category="missing_task_requirement",
            ),
            _check(
                "assay_and_class_separation_reported",
                coverage,
                "narrative must use observed dataset counts/assay identifiers",
                category="unsupported_claim",
            ),
            _check(
                "report_available",
                bool(reports)
                and any(
                    re.search(r"density", text, re.I) and re.search(r"activity", text, re.I)
                    for text in reports
                ),
                f"readable new reports={len(reports)}",
                category="missing_artifact",
            ),
        ]
    )
    return checks


def _molecular_candidates(output):
    return _memory_collection(output, "candidate_sets")


def _candidate_payload(output, record, *, fresh=True):
    """Prefer full artifacts; never treat a compact preview or count as candidates."""
    for key in ("artifact_path", "csv_path"):
        path = record.get(key)
        if not path:
            continue
        data = _artifact_bytes(output, path, fresh=fresh)
        if not data:
            continue
        try:
            if str(path).endswith(".json"):
                payload = json.loads(data)
                if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
                    return payload["candidates"], payload.get("metadata") or {}
            table = _table(output, path, fresh=fresh)
            if table is not None:
                return table.to_dict("records"), {}
        except (ValueError, TypeError):
            continue
    if _paths(record):
        return [], {}  # Unreadable artifacts must not silently fall back to a preview.
    if isinstance(record.get("candidates"), list):
        return record["candidates"], {}
    compounds = _state(output).get("session_objects", {}).get("compounds", {})
    return [
        compounds[item]
        for item in record.get("compound_ids", [])
        if item in compounds and isinstance(compounds[item], dict)
    ], {}


def _smiles_rows(rows):
    return [
        (
            row
            if isinstance(row, str)
            else str(row.get("smiles") or row.get("smi") or row.get("canonical_smiles") or "")
        )
        for row in rows
        if isinstance(row, (str, dict))
    ]


def _requested_count(output):
    explicit = output.get("requested_candidate_count")
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    # Only the unambiguous immediate count-noun pattern, not other numbers in prose.
    matches = re.findall(
        r"\b(\d+)\s+(?:(?:valid|new|generated|autoencoder|candidate)\s+){0,3}"
        r"(?:analogues?|analogs?|candidates?|molecules?|peptides?)\b",
        str(output.get("prompt", "")),
        re.I,
    )
    return int(matches[0]) if len(set(matches)) == 1 else 1


def _molecular_evidence(output, *, new=True):
    records = _new_records(output, "candidate_sets") if new else _molecular_candidates(output)
    evidence = []
    for record in records:
        rows, metadata = _candidate_payload(output, record, fresh=new)
        raw = _smiles_rows(rows)
        valid = [value for value in (_canonical_smiles(item) for item in raw) if value]
        evidence.append((record, metadata, raw, valid))
    return evidence


def validate_molecular_generation(output):
    checks = _execution_checks(output)
    evidence = _molecular_evidence(output)
    expected = _requested_count(output)
    matching = []
    engines = []
    for record, metadata, _raw, _valid in evidence:
        seed = record.get("seed_smiles") or metadata.get("seed_smiles")
        parent = record.get("seed_compound_id") or metadata.get("seed_compound_id")
        combined = json.dumps([record, metadata])
        provenance = _canonical_smiles(seed) == _canonical_smiles(_SEH_PARENT) and (
            str(parent).upper() == "CHEMBL3327073" or "CHEMBL3327073" in combined.upper()
        )
        matching.append(provenance)
        engines.append(
            str(record.get("generation_engine") or metadata.get("generation_engine") or "").lower()
        )
    counts = [
        {
            "raw_artifact_count": len(raw),
            "valid_count": len(valid),
            "unique_valid_count": len(set(valid)),
        }
        for _, _, raw, valid in evidence
    ]
    checks.extend(
        [
            _check(
                "molecular_design_tool_used",
                _has_tool(
                    output,
                    (
                        "design_molecules",
                        "generate_analogs",
                        "sample_activity_landscape_nodes",
                        "sample_molecules",
                    ),
                ),
                "successful generation call required",
                category="missing_required_call",
            ),
            _check(
                "candidate_set_registered",
                bool(evidence),
                f"new candidate sets={len(evidence)}",
                category="missing_artifact",
            ),
            _check(
                "valid_candidates_returned",
                any(len(set(valid)) >= expected for _, _, _, valid in evidence),
                f"required={expected}; measured={counts}",
                category="missing_scientific_outcome",
            ),
            _check(
                "parent_provenance_present",
                any(matching),
                "candidate provenance must identify CHEMBL3327073 and its actual seed structure",
                category="missing_provenance",
            ),
            _check(
                "candidate_count_and_provenance_linked",
                any(
                    ok and len(set(item[3])) >= expected
                    for ok, item in zip(matching, evidence, strict=True)
                ),
                "the same candidate set must satisfy count and provenance",
                category="missing_scientific_outcome",
            ),
        ]
    )
    if "autoencoder" in str(output.get("prompt", "")).lower():
        checks.append(
            _check(
                "requested_generation_engine_used",
                any(
                    provenance and engine == "autoencoder" and len(set(item[3])) >= expected
                    for provenance, engine, item in zip(matching, engines, evidence, strict=True)
                ),
                f"recorded engines={engines}",
                category="missing_provenance",
            )
        )
    return checks


def _plan(output):
    plan = _state(output).get("synplanner_plan")
    return plan if isinstance(plan, dict) else {}


def validate_retrosynthesis(output):
    checks = _execution_checks(output)
    plan = _plan(output)
    routes = plan.get("routes") if isinstance(plan.get("routes"), list) else []
    attempts = plan.get("attempts") if isinstance(plan.get("attempts"), list) else []
    completed = [
        item
        for item in attempts
        if isinstance(item, dict)
        and not item.get("error")
        and item.get("stop_reason") in {"routes_found", "no_routes"}
    ]
    old_plan = (output.get("initial_session_state") or {}).get("synplanner_plan")
    fresh = json.dumps(plan, sort_keys=True, default=str) != json.dumps(
        old_plan, sort_keys=True, default=str
    )
    if plan.get("plan_path"):
        fresh = bool(_artifact_bytes(output, plan["plan_path"], fresh=True))
    target = _canonical_smiles(plan.get("smiles"))
    candidates = [
        smiles for _, _, _, valid in _molecular_evidence(output, new=False) for smiles in valid
    ]
    target_matches = bool(target) and (not candidates or target in candidates)
    if "first valid" in str(output.get("prompt", "")).lower() and candidates:
        target_matches = target == candidates[0]
    response = _response(output)
    no_route = bool(re.search(r"no (?:\w+ )?routes?|0\s+routes?", response, re.I))
    route_details = all(
        isinstance(route, dict) and isinstance(route.get("steps"), list) and bool(route["steps"])
        for route in routes
    )
    outcome = (bool(routes) and route_details and bool(re.search(r"route", response, re.I))) or (
        not routes and bool(completed) and no_route
    )
    if routes and re.search(
        r"route (?:length|count)|number of routes", str(output.get("prompt", "")), re.I
    ):
        outcome &= bool(re.search(rf"\b{len(routes)}\b", response)) and any(
            re.search(rf"\b{len(route.get('steps', []))}\s*(?:-|\s)?steps?", response, re.I)
            for route in routes
        )
    checks.extend(
        [
            _check(
                "synplanner_search_executed",
                fresh and bool(completed) and _has_tool(output, ("plan_synthesis",)),
                f"completed search attempts={len(completed)}, new outcome={fresh}",
                category="missing_required_call",
            ),
            _check(
                "target_structure_resolved",
                target_matches,
                "valid target structure must match selected candidate when supplied",
                category="missing_provenance",
            ),
            _check(
                "route_outcome_reported",
                outcome,
                f"routes={len(routes)}, completed no-route={bool(completed) and not routes and no_route}",
                category="missing_scientific_outcome",
            ),
        ]
    )
    return checks


def validate_peptide_design(output):
    checks = _execution_checks(output)
    old = output.get("initial_session_state") or {}
    pointers = [
        value
        for key, value in _state(output).items()
        if isinstance(value, dict)
        and (value.get("peptide_candidate_set_id") or "peptide" in key.lower())
        and (
            json.dumps(value, sort_keys=True, default=str)
            != json.dumps(old.get(key), sort_keys=True, default=str)
            or any(_artifact_bytes(output, path, fresh=True) for path in _paths(value))
        )
    ]
    pointers.extend(
        record
        for record in _new_records(output, "analyses")
        if record.get("analysis_type") == "peptide_design"
    )
    sequences = []
    for pointer in pointers:
        rows, _ = _candidate_payload(output, pointer)
        for row in rows:
            seq = (
                row
                if isinstance(row, str)
                else row.get("sequence", "") if isinstance(row, dict) else ""
            )
            seq = re.sub(r"\s+", "", str(seq)).upper()
            if len(seq) >= 2 and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", seq):
                sequences.append(seq)
    tables = [(path, _table(output, path, fresh=True)) for path in _all_paths(output)]
    landscape = any(_numeric_table(table, lambda col: col.endswith("_prob")) for _, table in tables)
    # A frozen activity input is usable only when a successful current-step tool uses it.
    if not landscape and _has_tool(
        output, ("sample_activity_landscape_nodes", "create_peptide_activity_landscapes")
    ):
        landscape = any(
            _numeric_table(_table(output, path), lambda col: col.endswith("_prob"))
            for path in _all_paths(output)
        )
    reports = _report_texts(output)
    analysis_text = "\n".join(
        reports + [json.dumps(record, default=str) for record in _new_records(output, "analyses")]
    )
    metrics = bool(
        re.search(r"similar\w*[^\n]{0,100}\d", analysis_text, re.I)
        and re.search(r"unique\w*[^\n]{0,100}\d", analysis_text, re.I)
    )
    grounded = bool(sequences) and any(
        seq in re.sub(r"\s+", "", analysis_text).upper() for seq in sequences
    )
    logos = []
    for path in _all_paths(output):
        if "logo" not in path.lower():
            continue
        data = _artifact_bytes(output, path, fresh=True)
        if data and (
            data.startswith(b"\x89PNG\r\n\x1a\n")
            or b"<svg" in data[:1000]
            or data.startswith(b"%PDF")
        ):
            logos.append(path)
    checks.extend(
        [
            _check(
                "peptide_activity_landscape_used",
                landscape,
                "numerical activity landscape evidence required",
                category="missing_artifact",
            ),
            _check(
                "peptide_candidates_available",
                len(set(sequences)) >= _requested_count(output),
                f"parsed unique peptide sequences={len(set(sequences))}",
                category="missing_scientific_outcome",
            ),
            _check(
                "similarity_and_uniqueness_analyzed",
                metrics and grounded,
                "quantitative analysis must refer to generated sequences",
                category="unsupported_claim",
            ),
            _check(
                "sequence_logo_available",
                bool(logos),
                f"readable new logos={len(logos)}",
                category="missing_artifact",
            ),
            _check(
                "peptide_report_available",
                bool(reports) and grounded,
                f"readable new reports={len(reports)}",
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
    output = dict(output)
    output["_artifact_cache"] = {}
    checks = _VALIDATORS[validator_name](output)
    checks.extend(_frozen_tier_checks(validator_name, output))
    required = [check for check in checks if check.severity == "required"]
    task_success = bool(required) and all(check.passed for check in required)
    failure_categories = sorted(
        {check.category for check in checks if not check.passed and check.category is not None}
    )
    scientific_outcome: Dict[str, Any] = {}
    if validator_name == "molecular_generation":
        scientific_outcome = {
            "requested_candidate_count": _requested_count(output),
            "candidate_sets": [
                {
                    "candidate_set_id": record.get("id") or record.get("candidate_set_id"),
                    "stored_candidate_count": len(raw),
                    "valid_candidate_count": len(valid),
                    "unique_valid_candidate_count": len(set(valid)),
                    "reported_attempt_count": record.get("count_attempted")
                    or metadata.get("count_attempted"),
                }
                for record, metadata, raw, valid in _molecular_evidence(output)
            ],
        }
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
