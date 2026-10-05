"""Deterministic scientific facts on disk, selectively retrieved for reporting."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

import pandas as pd

from cs_copilot.storage import S3

ID_COLUMNS = {
    "distinct_assays": ("assay_chembl_id", "assay_chembl_ids", "assay_id", "assay_ids"),
    "distinct_documents": (
        "document_chembl_id",
        "document_chembl_ids",
        "document_id",
        "document_ids",
    ),
}
_MISSING = {"", "nan", "none", "null", "<na>"}


def atomic_ids(values: Any) -> set[str]:
    """Union atomic IDs; a serialized combination is never an entity."""
    result = set()
    for value in values:
        if pd.isna(value):
            continue
        result.update(
            token
            for part in str(value).split("|")
            if (token := part.strip()) and token.lower() not in _MISSING
        )
    return result


def coverage_counts(df: pd.DataFrame) -> dict[str, Any]:
    counts: dict[str, Any] = {"row_count": len(df)}
    for name, aliases in ID_COLUMNS.items():
        columns = [column for column in aliases if column in df]
        sets = [atomic_ids(df[column]) for column in columns]
        if not columns or any(ids != sets[0] for ids in sets[1:]):
            counts[name] = {
                "value": None,
                "status": "unresolved",
                "reason": (
                    "conflicting identifier columns" if columns else "identifier column missing"
                ),
            }
        else:
            counts[name] = {
                "value": len(sets[0]),
                "status": "verified",
                "columns": columns,
                "missing_rows": sum(not atomic_ids([value]) for value in df[columns[0]]),
                "method": "distinct nonmissing atomic IDs; split pipe-joined cells",
            }
    return counts


def scaffold_summary(df: pd.DataFrame, nodes: list[int] | None = None) -> dict[str, Any]:
    """Exact Murcko assignments per distinct canonical isomeric SMILES, including acyclic."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    selection = {"node_index": sorted(set(nodes))} if nodes is not None else {"all": True}
    selected = df if nodes is None else df.loc[df["node_index"].isin(nodes)]
    base = {
        "selection": selection,
        "unit": "distinct canonical isomeric SMILES",
        "definition": "exact achiral RDKit Murcko scaffold assignment; empty scaffold includes acyclic molecules",
    }
    if "smiles" not in selected:
        return {**base, "status": "unresolved", "reason": "smiles column missing", "rows": []}
    molecules = {}
    invalid = 0
    for smi in selected["smiles"].drop_duplicates():
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            invalid += 1
        else:
            molecules[Chem.MolToSmiles(mol)] = mol
    if invalid:
        return {
            **base,
            "status": "unresolved",
            "reason": "invalid or missing structures",
            "invalid_structures": invalid,
            "rows": [],
        }
    counts = Counter(
        MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        for mol in molecules.values()
    )
    return {
        **base,
        "status": "verified",
        "population_size": len(molecules),
        "rows": [
            {"scaffold_smiles": smi, "count": count}
            for smi, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        ],
    }


def source_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with S3.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_dataset_evidence(
    dataset_path: str,
    populations: dict[str, tuple[pd.DataFrame, str, str]],
    *,
    evidence_path: str | None = None,
    scaffold_nodes: list[int] | None = None,
) -> dict[str, str]:
    """Persist full facts and ID sets; return only a small versioned reference.

    Each population supplies its exact dataframe, backing artifact, and label.
    Callers must persist selected/derived frames before registering them.
    """
    sources = {path: source_sha256(path) for _, path, _ in populations.values()}
    version = sources.get(dataset_path) or source_sha256(dataset_path)
    sources[dataset_path] = version
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "dataset_version": version,
        "dataset_path": dataset_path,
        "sources": sources,
        "populations": {},
    }
    for name, (frame, path, label) in populations.items():
        if scaffold_nodes is not None:
            frame = frame.loc[frame["node_index"].isin(scaffold_nodes)]
        counts = coverage_counts(frame)
        identifiers = {
            metric: sorted(atomic_ids(frame[columns[0]]))
            for metric, aliases in ID_COLUMNS.items()
            if (columns := [column for column in aliases if column in frame])
            and counts[metric]["status"] == "verified"
        }
        population: dict[str, Any] = {
            "label": label,
            "source": path,
            "coverage": counts,
            "identifiers": identifiers,
        }
        if name == "clean" or scaffold_nodes is not None:
            population["scaffolds"] = scaffold_summary(frame, nodes=scaffold_nodes)
            scaffolds = population["scaffolds"]
            counts["distinct_structures"] = {
                "value": scaffolds.get("population_size"),
                "status": scaffolds["status"],
                "method": "distinct canonical isomeric SMILES",
            }
        evidence["populations"][name] = population
    # Merging may reduce rows, but must not lose IDs from contributing records.
    if "clean" in populations and "contributing" in populations:
        clean, contributing = (evidence["populations"][name] for name in ("clean", "contributing"))
        for metric in ID_COLUMNS:
            if contributing["coverage"][metric]["status"] == "unresolved":
                clean["coverage"][metric] = {
                    "value": None,
                    "status": "unresolved",
                    "reason": "contributing identifier evidence is unresolved",
                }
                continue
            if (
                metric in clean["identifiers"]
                and metric in contributing["identifiers"]
                and clean["identifiers"][metric] != contributing["identifiers"][metric]
            ):
                clean["coverage"][metric] = {
                    "value": None,
                    "status": "unresolved",
                    "reason": "clean and contributing identifier sets conflict",
                }
    path = evidence_path or dataset_path + ".evidence.json"
    with S3.open(path, "w") as handle:
        json.dump(evidence, handle, sort_keys=True, allow_nan=False)
    return {"dataset_version": version, "evidence_path": S3.path(path), "status": "verified"}


def get_report_evidence(
    evidence_path: str,
    section: str = "coverage",
    population: str = "clean",
    offset: int = 0,
    limit: int = 10,
) -> str:
    """Read scoped scientific facts without loading full evidence into agent context.

    Args:
        evidence_path: Evidence JSON artifact from dataset preparation or scaffold analysis.
        section: coverage, scaffolds, or identifiers (explicit audit only).
        population: Named population: clean, retained, retrieved, search, contributing, or selected.
        offset: Zero-based pagination offset for scaffolds or identifier details.
        limit: Maximum returned rows, 1 to 50. Totals always use the full population.

    Missing, changed, or conflicting evidence remains explicitly unresolved. Full
    tables and ID lists remain on disk; neither these results nor repeated reads
    are appended to session metadata. Fetch once per report section and reuse it.
    """
    if section not in {"coverage", "scaffolds", "identifiers"}:
        raise ValueError("section must be coverage, scaffolds, or identifiers")
    if not 1 <= limit <= 50 or offset < 0:
        raise ValueError("limit must be 1..50 and offset must be nonnegative")
    base: dict[str, Any] = {
        "evidence_path": evidence_path,
        "population": population,
        "section": section,
    }
    try:
        with S3.open(evidence_path, "r") as handle:
            artifact = json.load(handle)
        if artifact.get("schema_version") != 1:
            raise ValueError("unsupported evidence schema")
        if (
            not artifact["sources"]
            or artifact["sources"].get(artifact["dataset_path"]) != artifact["dataset_version"]
        ):
            raise ValueError("dataset source/version missing or inconsistent")
        for path, expected in artifact["sources"].items():
            if source_sha256(path) != expected:
                return json.dumps(
                    {**base, "status": "unresolved", "reason": "stale source", "source": path}
                )
        data = artifact["populations"].get(population)
        if data is None:
            return json.dumps(
                {
                    **base,
                    "status": "unresolved",
                    "reason": "population unavailable",
                    "available_populations": list(artifact["populations"]),
                }
            )
        if data["source"] not in artifact["sources"]:
            raise ValueError("population source missing from source hashes")
        base.update(
            status="verified",
            dataset_version=artifact["dataset_version"],
            population_label=data["label"],
            source=data["source"],
        )
        if section == "coverage":
            coverage = data["coverage"]
            if type(coverage["row_count"]) is not int or coverage["row_count"] < 0:
                raise ValueError("invalid population row count")
            for metric in ID_COLUMNS:
                value = coverage[metric]
                if value["status"] not in {"verified", "unresolved"}:
                    raise ValueError("invalid metric status")
                if value["status"] == "verified":
                    if (
                        type(value["value"]) is not int
                        or value["value"] < 0
                        or value["value"] != len(set(data["identifiers"][metric]))
                    ):
                        raise ValueError("coverage conflicts with supporting identifiers")
            base["coverage"] = coverage
        else:
            if section == "scaffolds":
                facts = data.get("scaffolds")
                if facts is None:
                    return json.dumps(
                        {**base, "status": "unresolved", "reason": "scaffolds unavailable"}
                    )
                if facts["status"] == "verified":
                    if (
                        type(facts["population_size"]) is not int
                        or facts["population_size"] < 0
                        or not isinstance(facts["selection"], dict)
                        or not isinstance(facts["unit"], str)
                        or not isinstance(facts["definition"], str)
                        or any(
                            not isinstance(row["scaffold_smiles"], str)
                            or type(row["count"]) is not int
                            or row["count"] <= 0
                            for row in facts["rows"]
                        )
                        or sum(row["count"] for row in facts["rows"]) != facts["population_size"]
                    ):
                        raise ValueError("invalid scaffold population or frequencies")
                elif facts["status"] != "unresolved" or not facts.get("reason"):
                    raise ValueError("invalid scaffold status")
                base.update({key: value for key, value in facts.items() if key != "rows"})
                rows = facts["rows"]
            else:
                rows = [
                    {"metric": metric, "id": identifier}
                    for metric, ids in data["identifiers"].items()
                    for identifier in ids
                ]
            end = offset + limit
            base.update(
                rows=rows[offset:end],
                total_rows=len(rows),
                offset=offset,
                partial=offset > 0 or end < len(rows),
                next_offset=end if end < len(rows) else None,
            )
        return json.dumps(base, sort_keys=True, allow_nan=False)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return json.dumps(
            {**base, "status": "unresolved", "reason": f"evidence unavailable or invalid: {exc}"}
        )


def evidence_report_section(request: dict[str, Any]) -> dict[str, Any]:
    """Render the requested facts directly; never infer counts from report prose."""
    facts = json.loads(get_report_evidence(**request))
    title = f"Evidence: {facts.get('population_label', facts['population'])}"
    provenance = f"Source evidence: {facts['evidence_path']}"
    if facts["status"] == "unresolved":
        return {"heading": title, "paragraphs": [f"Unresolved: {facts['reason']}", provenance]}
    provenance += f"; source: {facts['source']}; dataset SHA-256: {facts['dataset_version']}"
    if facts["section"] == "coverage":
        coverage = facts["coverage"]
        rows = [["Rows", coverage["row_count"], "rows in named population"]]
        for metric in ("distinct_assays", "distinct_documents", "distinct_structures"):
            if metric not in coverage:
                continue
            value = coverage[metric]
            method = value.get("method", value.get("reason", ""))
            if "missing_rows" in value:
                method += f"; {value['missing_rows']} rows without identifiers"
            rows.append(
                [
                    metric.replace("_", " "),
                    value["value"] if value["status"] == "verified" else "Unresolved",
                    method,
                ]
            )
        table = {"title": title, "columns": ["Metric", "Count", "Method"], "rows": rows}
    elif facts["section"] == "scaffolds":
        table = {
            "title": title,
            "columns": ["Scaffold", "Count", "Population size"],
            "rows": [
                [row["scaffold_smiles"] or "(acyclic)", row["count"], facts["population_size"]]
                for row in facts["rows"]
            ],
        }
        provenance += (
            f"; {facts['definition']}; unit: {facts['unit']}; selection: {facts['selection']}"
        )
        if facts["partial"]:
            provenance += f"; partial table ({facts['offset']} onwards, {facts['total_rows']} total scaffolds)"
    else:
        raise ValueError("Report tables accept coverage or scaffolds; identifiers are for audits")
    return {"heading": title, "paragraphs": [provenance], "tables": [table]}
